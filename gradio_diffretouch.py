import gradio as gr
import argparse
import sys
import os
import glob
sys.path.append(os.getcwd())
import torch
import numpy as np

from omegaconf import OmegaConf
from PIL import Image

from ldm.util import instantiate_from_config
from ldm.models.diffusion.spaced_sampler import SpacedSampler

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default='diffretouch_models/adobe.ckpt')
parser.add_argument("--config_path", type=str, default='configs/retouchdiff/adobe.yaml')
args = parser.parse_args()

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

def load_img(image):
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

def predict(input_image, colorfulness, brightness, contrast, temperature, ddim_steps=20, num_samples=1, seed=0):

    image = load_img(input_image)
    label = [[colorfulness, brightness, contrast, temperature]]

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



#
Intro= \
"""
## DiffRetouch: Using Diffusion to Retouch on the Shoulder of Experts

[Paper](https://arxiv.org/abs/2407.03757)
"""

config = OmegaConf.load(f"{args.config_path}")
model = load_model_from_config(config, f"{args.ckpt_path}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
sampler = SpacedSampler(model, var_type="fixed_small")

exaple_images = sorted(glob.glob('examples/*.png'))
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown(Intro)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil")
            colorfulness = gr.Slider(label="Colorfulness", minimum=-1.5, maximum=1.5, value=0.0, step=0.1)
            brightness = gr.Slider(label="Brightness", minimum=-1.5, maximum=1.5, value=0.0, step=0.1)
            contrast = gr.Slider(label="Contrast", minimum=-1.5, maximum=1.5, value=0.0, step=0.1)
            temperature = gr.Slider(label="Color Temperature", minimum=-1.5, maximum=1.5, value=0.0, step=0.1)
            gr.Examples(examples=exaple_images, inputs=[input_image])
        with gr.Column():
            result_gallery = gr.Gallery(label="Output", show_label=False, elem_id="gallery")
            with gr.Row():
                run_diffretouch_button = gr.Button(value="Run DiffRetouch")
    
    # label = [colorfulness, brightness, contrast, temperature]

    run_diffretouch_button.click(fn=predict, inputs=[input_image, colorfulness, brightness, contrast, temperature], outputs=[result_gallery])

block.launch()
