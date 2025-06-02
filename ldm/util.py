import importlib

import torch
from torch import optim
import numpy as np
import torch.nn.functional as F

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont


def autocast(f):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(enabled=True,
                                     dtype=torch.get_autocast_gpu_dtype(),
                                     cache_enabled=torch.is_autocast_cache_enabled()):
            return f(*args, **kwargs)

    return do_autocast


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x,torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def instantiate_from_config_vq_diffusion(config):
    """the VQ-Diffusion version"""
    if config is None:
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    print(f'instantiate_from_config --- module: {module}, cls: {cls}') # ziqi added
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class AdamWwithEMAandWings(optim.Optimizer):
    # credit to https://gist.github.com/crowsonkb/65f7265353f403714fce3b2595e0b298
    def __init__(self, params, lr=1.e-3, betas=(0.9, 0.999), eps=1.e-8,  # TODO: check hyperparameters before using
                 weight_decay=1.e-2, amsgrad=False, ema_decay=0.9999,   # ema decay to match previous code
                 ema_power=1., param_names=()):
        """AdamW that saves EMA versions of the parameters."""
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError("Invalid ema_decay value: {}".format(ema_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, ema_decay=ema_decay,
                        ema_power=ema_power, param_names=param_names)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            ema_params_with_grad = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            ema_decay = group['ema_decay']
            ema_power = group['ema_power']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of parameter values
                    state['param_exp_avg'] = p.detach().float().clone()

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                ema_params_with_grad.append(state['param_exp_avg'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            optim._functional.adamw(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=False)

            cur_ema_decay = min(ema_decay, 1 - state['step'] ** -ema_power)
            for param, ema_param in zip(params_with_grad, ema_params_with_grad):
                ema_param.mul_(cur_ema_decay).add_(param.float(), alpha=1 - cur_ema_decay)

        return loss
    
def _diff_x(src, r):
    cum_src = src.cumsum(-2)

    left = cum_src[..., r:2*r + 1, :]
    middle = cum_src[..., 2*r + 1:, :] - cum_src[..., :-2*r - 1, :]
    right = cum_src[..., -1:, :] - cum_src[..., -2*r - 1:-r - 1, :]

    output = torch.cat([left, middle, right], -2)

    return output

def _diff_y(src, r):
    cum_src = src.cumsum(-1)

    left = cum_src[..., r:2*r + 1]
    middle = cum_src[..., 2*r + 1:] - cum_src[..., :-2*r - 1]
    right = cum_src[..., -1:] - cum_src[..., -2*r - 1:-r - 1]

    output = torch.cat([left, middle, right], -1)

    return output

def boxfilter2d(src, radius):
    return _diff_y(_diff_x(src, radius), radius)

def batch_det(inputs, rank):
    inputs_v = []
    lines = []
    b, C, H, W = inputs[0].shape
    for input in inputs:
        inputs_v.append(input.reshape(b, C, H*W).transpose(1,2)) 
    for i in range(rank):
        a = torch.stack(inputs_v[i*rank: (i+1)*rank], dim=-1)
        lines.append(a)
    det = torch.cat(lines, dim=-2)
    cov_det = torch.det(det).reshape(b, C, H, W)
    return cov_det

def calculate_colorfulness(img):
    # split the image into its respective RGB components
	(R, G, B) = torch.chunk(img, 3, dim=1)
	# compute rg = R - G
	rg = torch.abs(R - G)
	# compute yb = 0.5 * (R + G) - B
	yb = torch.abs(0.5 * (R + G) - B)
	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (torch.mean(rg, dim=[2,3]), torch.std(rg, dim=[2,3]))
	(ybMean, ybStd) = (torch.mean(yb, dim=[2,3]), torch.std(yb, dim=[2,3]))
	# combine the mean and standard deviations
	stdRoot = torch.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = torch.sqrt((rbMean ** 2) + (ybMean ** 2))
	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)

def calculate_brightness(img):
    # print('img:', img.shape)
    (R, G, B) = torch.chunk(img, 3, dim=1)
    return torch.sqrt(0.241*(R.mean(dim=[2,3])**2) + 0.691*(G.mean(dim=[2,3])**2) + 0.068*(B.mean(dim=[2,3])**2))

def calculate_contrast(img):
    _, c, h, w = img.shape
    #图片矩阵向外扩展一个像素
    img_ext = F.pad(img, (1,1,1,1), 'replicate') 
    img_up = img_ext[:, :, 0:-2, 1:-1]
    img_down = img_ext[:, :, 2:, 1:-1]
    img_right = img_ext[:, :, 1:-1, 0:-2]
    img_left = img_ext[:, :, 1:-1, 2:]
    b = torch.sum((img_up - img)**2 + (img_down - img)**2 + (img_right - img)**2 + (img_left - img)**2, dim=[2,3])
    b = b.sum(dim=1, keepdim=True)

    cg = b/(4*(h-2)*(w-2)+3*(2*(h-2)+2*(w-2))+2*4) #对应上面48的计算公式
    return cg

def calculate_tone(img, eps = 1e-4):
    img = (img + 1.0) / 2.0 * 255
    (R, G, B) = torch.chunk(img, 3, dim=1)

    numerator = (0.23881) * R.mean(dim=[2,3]) +(0.25499) * G.mean(dim=[2,3])+(-0.58291) * B.mean(dim=[2,3])
    denominator = (0.11109) * R.mean(dim=[2,3]) +(-0.85406) * G.mean(dim=[2,3]) +(0.52289) * B.mean(dim=[2,3]) + eps
    if torch.abs(denominator) >=10:
        n = numerator / denominator
    else:
        n = torch.zeros_like(denominator)
    return (449 * n**3 + 3525 * n**2 + 6823.3 * n)

def slicing(grid, guide):
    N, C, H, W = guide.shape
    device = grid.get_device()
    if device >= 0:
        hh, ww = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device)) # H, W
    else:
        hh, ww = torch.meshgrid(torch.arange(H), torch.arange(W)) # H, W
    # To [-1, 1] range for grid_sample
    hh = hh / (H - 1) * 2 - 1
    ww = ww / (W - 1) * 2 - 1
    guide = guide * 2 - 1
    hh = hh[None, :, :, None].repeat(N, 1, 1, 1) # N, H, W, C=1
    ww = ww[None, :, :, None].repeat(N, 1, 1, 1)  # N, H, W, C=1
    guide = guide.permute(0, 2, 3, 1) # N, H, W, C=1

    guide_coords = torch.cat([ww, hh, guide], dim=3) # N, H, W, 3
    # unsqueeze because extra D dimension
    guide_coords = guide_coords.unsqueeze(1) # N, Dout=1, H, W, 3
    sliced = F.grid_sample(grid, guide_coords, align_corners=False, padding_mode="border") # N, C=12, Dout=1, H, W
    sliced = sliced.squeeze(2) # N, C=12, H, W

    return sliced

def apply(sliced, fullres):
    # r' = w1*r + w2*g + w3*b + w4
    rr = fullres * sliced[:, 0:3, :, :] # N, C=3, H, W
    gg = fullres * sliced[:, 4:7, :, :] # N, C=3, H, W
    bb = fullres * sliced[:, 8:11, :, :] # N, C=3, H, W
    rr = torch.sum(rr, dim=1) + sliced[:, 3, :, :] # N, H, W
    gg = torch.sum(gg, dim=1) + sliced[:, 7, :, :] # N, H, W
    bb = torch.sum(bb, dim=1) + sliced[:, 11, :, :] # N, H, W
    output = torch.stack([rr, gg, bb], dim=1) # N, C=3, H, W
    return output