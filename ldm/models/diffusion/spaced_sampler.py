from typing import Optional, Tuple, Dict, List, Callable

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_beta_schedule

# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py
def space_timesteps(num_timesteps, section_counts, start_idx = 200):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    num_timesteps = num_timesteps - start_idx
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    try:
        # float64 as default. float64 is not supported by mps device.
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    except:
        # to be compatible with mps
        res = torch.from_numpy(arr.astype(np.float32)).to(device=timesteps.device)[timesteps].float()
        
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class SpacedSampler:
    """
    Implementation for spaced sampling schedule proposed in IDDPM. This class is designed
    for sampling ControlLDM.
    
    https://arxiv.org/pdf/2102.09672.pdf
    """
    
    def __init__(
        self,
        model,
        schedule: str="linear",
        var_type: str="fixed_small"
    ) -> "SpacedSampler":
        self.model = model
        self.original_num_steps = model.num_timesteps
        self.schedule = schedule
        self.var_type = var_type

    def make_schedule(self, num_steps: int) -> None:
        """
        Initialize sampling parameters according to `num_steps`.
        
        Args:
            num_steps (int): Sampling steps.

        Returns:
            None
        """
        # NOTE: this schedule, which generates betas linearly in log space, is a little different
        # from guided diffusion.
        original_betas = make_beta_schedule(
            self.schedule, self.original_num_steps, linear_start=self.model.linear_start,
            linear_end=self.model.linear_end
        )
        original_alphas = 1.0 - original_betas
        original_alphas_cumprod = np.cumprod(original_alphas, axis=0)
        
        # calcualte betas for spaced sampling
        # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py
        used_timesteps = space_timesteps(self.original_num_steps, str(num_steps))
        # print(f"timesteps used in spaced sampler: \n\t{sorted(list(used_timesteps))}")
        
        betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(original_alphas_cumprod):
            if i in used_timesteps:
                # marginal distribution is the same as q(x_{S_t}|x_0)
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        assert len(betas) == num_steps
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        
        self.timesteps = np.array(sorted(list(used_timesteps)), dtype=np.int32) # e.g. [0, 10, 20, ...]
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (num_steps, )
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        """
        Implement the marginal distribution q(x_t|x_0).

        Args:
            x_start (torch.Tensor): Images (NCHW) sampled from data distribution.
            t (torch.Tensor): Timestep (N) for diffusion process. `t` serves as an index
                to get parameters for each timestep.
            noise (torch.Tensor, optional): Specify the noise (NCHW) added to `x_start`.

        Returns:
            x_t (torch.Tensor): The noisy images.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Implement the posterior distribution q(x_{t-1}|x_t, x_0).
        
        Args:
            x_start (torch.Tensor): The predicted images (NCHW) in timestep `t`.
            x_t (torch.Tensor): The sampled intermediate variables (NCHW) of timestep `t`.
            t (torch.Tensor): Timestep (N) of `x_t`. `t` serves as an index to get 
                parameters for each timestep.
        
        Returns:
            posterior_mean (torch.Tensor): Mean of the posterior distribution.
            posterior_variance (torch.Tensor): Variance of the posterior distribution.
            posterior_log_variance_clipped (torch.Tensor): Log variance of the posterior distribution.
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_xstart_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor
    ) -> torch.Tensor:
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def predict_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        cfg_scale: float,
        uncond: Optional[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        if uncond is None or cfg_scale == 1.:
            output = self.model.apply_model(x, t, cond)
            model_output = output[:,:4,...]
            affine_output = output[:,4:,...]
        else:
            # apply classifier-free guidance
            model_cond = self.model.apply_model(x, t, cond)
            model_uncond = self.model.apply_model(x, t, uncond)
            model_output = model_uncond + cfg_scale * (model_cond - model_uncond)
        
        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        return e_t, affine_output
    
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        cond,
        guidance,
        t: torch.Tensor,
        index: torch.Tensor,
        cfg_scale: float,
        uncond
    ) -> torch.Tensor:
        # variance of posterior distribution q(x_{t-1}|x_t, x_0)
        model_variance = {
            "fixed_large": np.append(self.posterior_variance[1], self.betas[1:]),
            "fixed_small": self.posterior_variance
        }[self.var_type]
        model_variance = _extract_into_tensor(model_variance, index, x.shape)
        
        # mean of posterior distribution q(x_{t-1}|x_t, x_0)
        e_t, affine_output = self.predict_noise(
            x, t, cond, cfg_scale, uncond
        )
        pred_x0 = self._predict_xstart_from_eps(x_t=x, t=index, eps=e_t)
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_x0, x_t=x, t=index
        )
        
        # sample x_t from q(x_{t-1}|x_t, x_0)
        noise = torch.randn_like(x)
        nonzero_mask = (
            (index != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        x_prev = model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
        x0_affine = self.model.hdrdecoder(affine_output, guidance)
        return x_prev, pred_x0, x0_affine
    
    @torch.no_grad()
    def sample(
        self,
        steps: int,
        guidance,
        num_samples,
        shape: Tuple[int],
        cond,
        x_T: Optional[torch.Tensor]=None,
        log_every_t = 10,
        cfg_scale: float=1.
    ) -> torch.Tensor:
        self.make_schedule(num_steps=steps)
        
        device = next(self.model.parameters()).device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        
        time_range = np.flip(self.timesteps) # [1000, 950, 900, ...]
        total_steps = len(self.timesteps)
        iterator = tqdm(time_range, desc="Spaced Sampler", total=total_steps)
        
        intermediates = {'x_inter': [img], 'pred_x0': [img], 'x0_affine': [guidance]}
        for i, step in enumerate(iterator):
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            index = torch.full_like(ts, fill_value=total_steps - i - 1)
            index_log = total_steps - i - 1
            outs = self.p_sample(
                img, cond, guidance, ts, index=index,
                cfg_scale=cfg_scale, uncond=None
            )
            img, pred_x0, x0_affine = outs

            if index_log % log_every_t == 0 or index_log == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
                intermediates['x0_affine'].append(x0_affine)
        
        # img_pixel = (self.model.decode_first_stage(img) + 1) / 2
        return img, intermediates, x0_affine
