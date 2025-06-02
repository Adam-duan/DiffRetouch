"""make variations of input image"""

import argparse, os
import PIL
import cv2
import torch
import json
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.spaced_sampler import SpacedSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    print(len(pl_sd['optimizer_states'][0]['state']))
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.

def make_batch_sd(
        image,
        label,
        device):

    batch = {
        "image": image.to(device=device),
        "label": label,
    }
    return batch

def retouch(sampler, image, label, seed, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)

    with torch.no_grad(), \
            torch.autocast("cuda"):
        batch = make_batch_sd(image, label=label,
                              device=device)

        c = model.cond_stage_model(batch["label"])
        guidance = batch['image']

        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck == model.masked_image_key:
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.get_first_stage_encoding(
                    model.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        print(c_cat.shape)

        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        shape = [num_samples, model.channels, h // 8, w // 8]
        samples_cfg, intermediates, x0_affine = sampler.sample(
            ddim_steps,
            guidance, 
            num_samples,
            shape,
            cond,
            x_T=start_code,
        )

        result = torch.clamp((x0_affine + 1.0) / 2.0,
                             min=0.0, max=1.0)

        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [Image.fromarray(img.astype(np.uint8)) for img in result]

def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

def predict(sampler, input_image, label, ddim_steps, num_samples, seed):
    # image = pad_image(input_image) # resize to integer multiple of 32
    image = input_image

    result = retouch(
        sampler=sampler,
        image=image,
        label=label,
        seed=seed,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        h=512, w=512
    )

    return result


def main():
    print('start')
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--promptdir",
        type=str,
        nargs="?",
        help="dir to the input image",
    )

    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir to the input image",
        default="test_data/Adobe5K/RAW"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="results/"
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=20,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )

    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )

    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/retouchdiff/adobe.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="diffretouch_models/adobe.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = SpacedSampler(model, var_type="fixed_small")


    prompt_path = os.path.join('test_data/Adobe5K/json_new', opt.promptdir+'.json')
    with open(prompt_path, 'r') as f:
        text_file_content = json.load(f)

    batch_size = opt.n_samples
    outpath = opt.outdir
    sample_path = os.path.join(outpath, f"samples_{opt.promptdir}")
    os.makedirs(sample_path, exist_ok=True)
    # count = 0

    img_name_list = sorted(os.listdir(opt.indir))
    for b in range(len(img_name_list)//batch_size):
        print(b, len(img_name_list))
        img_list = []
        label_list = []
        init_img_name_list = []
        init_img_name = img_name_list[batch_size * b]
        for i in range(batch_size):
            init_img_name = img_name_list[batch_size * b + i]
            # init_img_name = 'a4974.png'
            init_img_name_list.append(init_img_name)
            init_img = os.path.join(opt.indir, init_img_name)
            assert os.path.isfile(init_img)
            init_image = load_img(init_img).to(device)
            img_list.append(init_image)
            label = text_file_content[init_img_name]
            label_list.append(label)
        
        init_image = torch.cat(img_list, dim=0)

        x_samples = predict(sampler, init_image, label_list, opt.ddim_steps, batch_size, opt.seed)


        for j in range(len(x_samples)):
            x_sample = x_samples[j]
            init_img_name = init_img_name_list[j]
            x_sample.save(os.path.join(sample_path, f"{init_img_name}"))


    print(f"Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.")


if __name__ == "__main__":
    main()
