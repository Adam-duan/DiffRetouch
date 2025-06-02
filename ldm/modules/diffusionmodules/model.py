# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
from einops import rearrange
from typing import Optional, Any
import torch.nn.functional as F
import cv2

from ldm.modules.attention import MemoryEfficientCrossAttention

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
    print("No module 'xformers'. Proceeding without it.")


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activation=nn.ReLU, batch_norm=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation:
        layers.append(activation())
    return nn.Sequential(*layers)

def fc_layer(in_channels, out_channels, bias=True, activation=nn.ReLU, batch_norm=False):
    layers = [nn.Linear(int(in_channels), int(out_channels), bias=bias)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_channels))
    if activation:
        layers.append(activation())
    return nn.Sequential(*layers)

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

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h
    
class Coefficients(nn.Module):
    def __init__(self, c_in=8):
        super(Coefficients, self).__init__()
        self.relu = nn.ReLU()

        # ===========================Splat===========================
        self.splat1 = conv_layer(c_in, 8,  kernel_size=3, stride=2, padding=1, batch_norm=False)
        self.splat2 = conv_layer(8,    16, kernel_size=3, stride=2, padding=1, batch_norm=False)
        self.splat3 = conv_layer(16,   32, kernel_size=3, stride=1, padding=1, batch_norm=False)
        self.splat4 = conv_layer(32,   64, kernel_size=3, stride=1, padding=1, batch_norm=False)

        # ===========================Global===========================
        # Conv until 4x4
        self.global1 = conv_layer(64, 64, kernel_size=3, stride=2, padding=1, batch_norm=False)
        self.global2 = conv_layer(64, 64, kernel_size=3, stride=2, padding=1, batch_norm=False)
        # Caculate size after flatten for fc layers
        flatten_size = 4*4 * 64 # 4x4 * nchans
        self.global3 = fc_layer(flatten_size, 256, batch_norm=False)
        self.global4 = fc_layer(256,          128, batch_norm=False)
        self.global5 = fc_layer(128,          64,  activation=None)

        # ===========================Local===========================
        self.local1 = conv_layer(64, 64, kernel_size=3, padding=1, batch_norm=False)
        self.local2 = conv_layer(64, 64, kernel_size=3, padding=1, bias=False, activation=None)

        # ===========================predicton===========================
        self.pred = conv_layer(64, 96, kernel_size=1, activation=None) # 64 -> 96

    def forward(self, x):
        N = x.shape[0]
        # ===========================Splat===========================
        x = self.splat1(x) # N, C=8,  H=128, W=128
        x = self.splat2(x) # N, C=16, H=64,  W=64
        x = self.splat3(x) # N, C=32, H=32,  W=32
        x = self.splat4(x) # N, C=64, H=16,  W=16
        splat_out = x # N, C=64, H=16,  W=16

        # ===========================Global===========================
        # convs
        x = self.global1(x) # N, C=64, H=8, W=8
        x = self.global2(x) # N, C=64, H=4, W=4
        # flatten
        x = x.view(N, -1)   # N, C=64, H=4, W=4 -> N, 1024
        # fcs
        x = self.global3(x) # N, 256
        x = self.global4(x) # N, 128
        x = self.global5(x) # N, 64
        global_out = x # N, 64

        # ===========================Local===========================
        x = splat_out
        x = self.local1(x) # N, C=64, H=16,  W=16
        x = self.local2(x) # N, C=64, H=16,  W=16
        local_out = x # N, C=64, H=16, W=16

        # ===========================Fusion===========================
        global_out = global_out[:, :, None, None] # N, 64， 1， 1
        fusion = self.relu(local_out + global_out) # N, C=64, H=16, W=16

        # ===========================Prediction===========================
        x = self.pred(fusion) # N, C=96, H=16, W=16
        x = x.view(N, 12, 8, 16, 16) # N, C=12, D=8, H=16, W=16

        return x
    
class CoefficientsV2(nn.Module):
    def __init__(self, c_in=4):
        super(CoefficientsV2, self).__init__()
        self.relu = nn.ReLU()

        # ===========================Splat===========================
        self.splat1 = conv_layer(c_in, 16,  kernel_size=3, stride=2, padding=1, batch_norm=False)
        self.splat2 = conv_layer(16,  32, kernel_size=3, stride=2, padding=1, batch_norm=False)

        # ===========================predicton===========================
        self.pred1 = conv_layer(32, 64, kernel_size=1, activation=None) # 64 -> 96
        self.pred2 = conv_layer(64, 96, kernel_size=1, activation=None) # 64 -> 96

    def forward(self, x):
        N = x.shape[0]
        # ===========================Splat===========================
        x = self.splat1(x) # N, C=16, H=32, W=32
        x = self.splat2(x) # N, C=32, H=16, W=16

        x = self.pred1(x)
        x = self.pred2(x)
        x = x.view(N, 12, 8, 16, 16) # N, C=12, D=8, H=16, W=16

        return x


class Guide(nn.Module):
    def __init__(self, c_in=3):
        super(Guide, self).__init__()
        # Number of relus/control points for the curve
        self.nrelus = 16
        self.c_in = c_in
        self.M = nn.Parameter(torch.eye(c_in, dtype=torch.float32) + torch.randn(1, dtype=torch.float32) * 1e-4) # (c_in, c_in)
        self.M_bias = nn.Parameter(torch.zeros(c_in, dtype=torch.float32)) # (c_in,)
        # The shifts/thresholds in x of relus
        thresholds = np.linspace(0, 1, self.nrelus, endpoint=False, dtype=np.float32) # (nrelus,)
        thresholds = torch.tensor(thresholds) # (nrelus,)
        thresholds = thresholds[None, None, None, :] # (1, 1, 1, nrelus)
        thresholds = thresholds.repeat(1, 1, c_in, 1) # (1, 1, c_in, nrelus)
        self.thresholds = nn.Parameter(thresholds) # (1, 1, c_in, nrelus)
        # The slopes of relus
        slopes = torch.zeros(1, 1, 1, c_in, self.nrelus, dtype=torch.float32) # (1, 1, 1, c_in, nrelus)
        slopes[:, :, :, :, 0] = 1.0
        self.slopes = nn.Parameter(slopes)

        self.relu = nn.ReLU()
        self.bias = nn.Parameter(torch.tensor(0, dtype=torch.float32))

    def forward(self, x):
        # Permute from (N, C=3, H, W) to (N, H, W, C=3)
        x = x.permute(0, 2, 3, 1) # N, H, W, C=3
        old_shape = x.shape # (N, H, W, C=3)

        x = torch.matmul(x.reshape(-1, self.c_in), self.M) # N*H*W, C=3
        x = x + self.M_bias
        x = x.reshape(old_shape) # N, H, W, C=3
        x = x.unsqueeze(4) # N, H, W, C=3, 1

        x = torch.sum(self.slopes * self.relu(x - self.thresholds), dim=4) # N, H, W, C=3

        x = x.permute(0, 3, 1, 2) # N, C=3, H, W
        x = torch.sum(x, dim=1, keepdim=True) / self.c_in # N, C=1, H, W
        x = x + self.bias # N, C=1, H, W
        x = torch.clamp(x, 0, 1) # N, C=1, H, W

        return x


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class MemoryEfficientAttnBlock(nn.Module):
    """
        Uses xformers efficient implementation,
        see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
        Note: this is a single-head self-attention operation
    """
    #
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.attention_op: Optional[Any] = None

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, 'b c h w -> b (h w) c'), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        out = rearrange(out, 'b (h w) c -> b c h w', b=B, h=H, w=W, c=C)
        out = self.proj_out(out)
        return x+out


class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):
    def forward(self, x, context=None, mask=None):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        out = super().forward(x, context=context, mask=mask)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w, c=c)
        return x + out


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    assert attn_type in ["vanilla", "vanilla-xformers", "memory-efficient-cross-attn", "linear", "none"], f'attn_type {attn_type} unknown'
    if XFORMERS_IS_AVAILBLE and attn_type == "vanilla":
        attn_type = "vanilla-xformers"
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    elif attn_type == "vanilla-xformers":
        print(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
        return MemoryEfficientAttnBlock(in_channels)
    elif type == "memory-efficient-cross-attn":
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        raise NotImplementedError()


class Model(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True, use_linear_attn=False, attn_type="vanilla"):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t=None, context=None):
        #assert x.shape[2] == x.shape[3] == self.resolution
        if context is not None:
            # assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, return_fea=False):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        fea_list = []
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if return_fea:
                if i_level==1 or i_level==2:
                    fea_list.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        if return_fea:
            return h, fea_list
        
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
    
class HDRDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        self.coefficients = Coefficients()
        self.guide = Guide()

    def forward(self, z_enc, z, guidance):
        grid = self.coefficients(torch.cat([z_enc, z], dim=1))
        guide = self.guide(guidance)
        sliced = slicing(grid, guide)
        output = apply(sliced, guidance)
        return output
    
class HDRDecoderV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficients = CoefficientsV2()
        self.guide = Guide()

    def forward(self, z, guidance):
        grid = self.coefficients(z)
        guide = self.guide(guidance)
        # guide_save = guide.clone().detach().cpu().numpy()
        # guide_save = (guide_save - guide_save.min()) / (guide_save.max() - guide_save.min())
        # guide_save = (guide_save[0][0]*255).astype(np.uint8)
        # cv2.imwrite('/mnt/lustrenew/duanzhengpeng/retouch-diff/ABG/guidance.png', guide_save)
        sliced = slicing(grid, guide)
        # sliced_show = sliced.clone().detach().cpu().numpy()
        # for i in range(12):
        #     sliced_save = sliced_show[0][i]
        #     sliced_save = (sliced_save - sliced_save.min()) / (sliced_save.max() - sliced_save.min())
        #     sliced_save = (sliced_save * 255).astype(np.uint8)
        #     cv2.imwrite(f'/mnt/lustrenew/duanzhengpeng/retouch-diff/ABG/sliced_{i}.png', sliced_save)
        output = apply(sliced, guidance)
        return output

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResBlk(nn.Module):
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(ResBlk, self).__init__()
		self.bias1a = nn.Parameter(torch.zeros(1))
		self.actv1 = nn.PReLU()
		self.bias1b = nn.Parameter(torch.zeros(1))
		self.conv1 = conv3x3(inplanes, planes, stride)
		
		self.bias2a = nn.Parameter(torch.zeros(1))
		self.actv2 = nn.PReLU()
		self.bias2b = nn.Parameter(torch.zeros(1))
		self.conv2 = conv3x3(planes, planes)
		self.scale = nn.Parameter(torch.ones(1))

		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.actv1(x + self.bias1a)

		if self.downsample is not None:
			identity = self.downsample(out)

		out = self.conv1(out + self.bias1b)

		out = self.actv2(out + self.bias2a)
		out = self.conv2(out + self.bias2b)
		
		out = out * self.scale

		out += identity

		return out
 
class StyleEncoder(nn.Module):
	def __init__(self, dim):
		super(StyleEncoder, self).__init__()
		self.layers = [4, 4, 4, 4]
		self.planes = [64, 128, 256, 512]

		self.num_layers = sum(self.layers)
		self.inplanes = self.planes[0]

		self.conv1 = nn.Conv2d(3, self.planes[0], kernel_size=7, stride=2, padding=3, bias=False)
		self.bias1 = nn.Parameter(torch.zeros(1))
		self.actv = nn.PReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(ResBlk, self.planes[0], self.layers[0])
		self.layer2 = self._make_layer(ResBlk, self.planes[1], self.layers[1], stride=2)
		self.layer3 = self._make_layer(ResBlk, self.planes[2], self.layers[2], stride=2)
		self.layer4 = self._make_layer(ResBlk, self.planes[3], self.layers[3], stride=2)
		self.gap = nn.AdaptiveAvgPool2d(1)
		self.gmp = nn.AdaptiveMaxPool2d(1)
		self.bias2 = nn.Parameter(torch.zeros(1))

		self.fc = nn.Linear(self.planes[3], dim)

		self._reset_params()

	def _reset_params(self):
		for m in self.modules():
			if isinstance(m, ResBlk):
				nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
				nn.init.constant_(m.conv2.weight, 0)
				if m.downsample is not None:
					nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
			elif isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes:
			downsample = conv1x1(self.inplanes, planes, stride)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.actv(x + self.bias1)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		avg_x = self.gap(x)
		max_x = self.gmp(x)

		x = (max_x + avg_x).flatten(1)
		x = self.fc(x + self.bias2)

		x = F.normalize(x, p=2, dim=1)

		return x


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 1),
                                     ResnetBlock(in_channels=in_channels,
                                                 out_channels=2 * in_channels,
                                                 temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=2 * in_channels,
                                                out_channels=4 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=4 * in_channels,
                                                out_channels=2 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     nn.Conv2d(2*in_channels, in_channels, 1),
                                     Upsample(in_channels, with_conv=True)])
        # end
        self.norm_out = Normalize(in_channels)
        self.conv_out = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1,2,3]:
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution,
                 ch_mult=(2,2), dropout=0.0):
        super().__init__()
        # upsampling
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # upsampling
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class LatentRescaler(nn.Module):
    def __init__(self, factor, in_channels, mid_channels, out_channels, depth=2):
        super().__init__()
        # residual block, interpolate, residual block
        self.factor = factor
        self.conv_in = nn.Conv2d(in_channels,
                                 mid_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.res_block1 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])
        self.attn = AttnBlock(mid_channels)
        self.res_block2 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])

        self.conv_out = nn.Conv2d(mid_channels,
                                  out_channels,
                                  kernel_size=1,
                                  )

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.res_block1:
            x = block(x, None)
        x = torch.nn.functional.interpolate(x, size=(int(round(x.shape[2]*self.factor)), int(round(x.shape[3]*self.factor))))
        x = self.attn(x)
        for block in self.res_block2:
            x = block(x, None)
        x = self.conv_out(x)
        return x


class MergedRescaleEncoder(nn.Module):
    def __init__(self, in_channels, ch, resolution, out_ch, num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 ch_mult=(1,2,4,8), rescale_factor=1.0, rescale_module_depth=1):
        super().__init__()
        intermediate_chn = ch * ch_mult[-1]
        self.encoder = Encoder(in_channels=in_channels, num_res_blocks=num_res_blocks, ch=ch, ch_mult=ch_mult,
                               z_channels=intermediate_chn, double_z=False, resolution=resolution,
                               attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=resamp_with_conv,
                               out_ch=None)
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=intermediate_chn,
                                       mid_channels=intermediate_chn, out_channels=out_ch, depth=rescale_module_depth)

    def forward(self, x):
        x = self.encoder(x)
        x = self.rescaler(x)
        return x


class MergedRescaleDecoder(nn.Module):
    def __init__(self, z_channels, out_ch, resolution, num_res_blocks, attn_resolutions, ch, ch_mult=(1,2,4,8),
                 dropout=0.0, resamp_with_conv=True, rescale_factor=1.0, rescale_module_depth=1):
        super().__init__()
        tmp_chn = z_channels*ch_mult[-1]
        self.decoder = Decoder(out_ch=out_ch, z_channels=tmp_chn, attn_resolutions=attn_resolutions, dropout=dropout,
                               resamp_with_conv=resamp_with_conv, in_channels=None, num_res_blocks=num_res_blocks,
                               ch_mult=ch_mult, resolution=resolution, ch=ch)
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=z_channels, mid_channels=tmp_chn,
                                       out_channels=tmp_chn, depth=rescale_module_depth)

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Upsampler(nn.Module):
    def __init__(self, in_size, out_size, in_channels, out_channels, ch_mult=2):
        super().__init__()
        assert out_size >= in_size
        num_blocks = int(np.log2(out_size//in_size))+1
        factor_up = 1.+ (out_size % in_size)
        print(f"Building {self.__class__.__name__} with in_size: {in_size} --> out_size {out_size} and factor {factor_up}")
        self.rescaler = LatentRescaler(factor=factor_up, in_channels=in_channels, mid_channels=2*in_channels,
                                       out_channels=in_channels)
        self.decoder = Decoder(out_ch=out_channels, resolution=out_size, z_channels=in_channels, num_res_blocks=2,
                               attn_resolutions=[], in_channels=None, ch=in_channels,
                               ch_mult=[ch_mult for _ in range(num_blocks)])

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Resize(nn.Module):
    def __init__(self, in_channels=None, learned=False, mode="bilinear"):
        super().__init__()
        self.with_conv = learned
        self.mode = mode
        if self.with_conv:
            print(f"Note: {self.__class__.__name} uses learned downsampling and will ignore the fixed {mode} mode")
            raise NotImplementedError()
            assert in_channels is not None
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)

    def forward(self, x, scale_factor=1.0):
        if scale_factor==1.0:
            return x
        else:
            x = torch.nn.functional.interpolate(x, mode=self.mode, align_corners=False, scale_factor=scale_factor)
        return x
    
class Decoder_Mix(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", num_fuse_block=2, fusion_w=1.0, **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.fusion_w = fusion_w

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]

            if i_level != self.num_resolutions-1:
                if i_level != 0:
                    fuse_layer = Fuse_sft_block_RRDB(in_ch=block_out, out_ch=block_out, num_block=num_fuse_block)
                    setattr(self, 'fusion_layer_{}'.format(i_level), fuse_layer)

            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, enc_fea):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != self.num_resolutions-1 and i_level != 0:
                cur_fuse_layer = getattr(self, 'fusion_layer_{}'.format(i_level))
                h = cur_fuse_layer(enc_fea[i_level-1], h, self.fusion_w)

            if i_level != 0:
                h = self.up[i_level].upsample(h)
        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = nonlinearity(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = nonlinearity(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in

class Fuse_sft_block_RRDB(nn.Module):
    def __init__(self, in_ch, out_ch, num_block=1, num_grow_ch=32):
        super().__init__()
        self.encode_enc_1 = ResBlock(2*in_ch, in_ch)
        self.encode_enc_2 = make_layer(RRDB, num_block, num_feat=in_ch, num_grow_ch=num_grow_ch)
        self.encode_enc_3 = ResBlock(in_ch, out_ch)

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc_1(torch.cat([enc_feat, dec_feat], dim=1))
        enc_feat = self.encode_enc_2(enc_feat)
        enc_feat = self.encode_enc_3(enc_feat)
        residual = w * enc_feat
        out = dec_feat + residual
        return out

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x
    
@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2*in_ch, out_ch)

        self.scale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out

class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=64, norm=nn.BatchNorm2d):
        super(ConvGuidedFilter, self).__init__()

        self.box_filter = nn.Conv2d(3, 3, kernel_size=radius, stride=radius//2, bias=False, groups=3)
        self.conv_a = nn.Sequential(nn.Conv2d(6, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 3, kernel_size=1, bias=False))
        self.box_filter.weight.data[...] = 1.0

    def forward(self, guide, src):
        _, _, h_hrx, w_hrx = guide.size()

        N = self.box_filter(guide.data.new().resize_((1, 3, h_hrx, w_hrx)).fill_(1.0))
        ## mean_x
        mean_x = self.box_filter(guide)/N
        ## mean_y
        mean_y = self.box_filter(src)/N
        ## cov_xy
        cov_xy = self.box_filter(guide * src)/N - mean_x * mean_y
        ## var_x
        var_x  = self.box_filter(guide * guide)/N - mean_x * mean_x

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A * guide + mean_b
    
class ConvGuidedFilterASPP(nn.Module):
    def __init__(self, radius=[8,16,32,64], norm=nn.BatchNorm2d):
        super(ConvGuidedFilterASPP, self).__init__()

        self.box_filter1 = nn.Conv2d(3, 3, kernel_size=radius[0], stride=radius[0]//2, bias=False, groups=3, padding_mode='replicate')
        self.box_filter2 = nn.Conv2d(3, 3, kernel_size=radius[1], stride=radius[1]//2, bias=False, groups=3, padding_mode='replicate')
        self.box_filter3 = nn.Conv2d(3, 3, kernel_size=radius[2], stride=radius[2]//2, bias=False, groups=3, padding_mode='replicate')
        self.box_filter4 = nn.Conv2d(3, 3, kernel_size=radius[3], stride=radius[3]//2, bias=False, groups=3, padding_mode='replicate')
        self.conv_a = nn.Sequential(nn.Conv2d(24, 48, kernel_size=1, bias=False),
                                    norm(48),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(48, 48, kernel_size=1, bias=False),
                                    norm(48),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(48, 3, kernel_size=1, bias=False))
        self.box_filter1.weight.data[...] = 1.0
        self.box_filter2.weight.data[...] = 1.0
        self.box_filter3.weight.data[...] = 1.0
        self.box_filter4.weight.data[...] = 1.0

    def forward(self, guide, src):
        _, _, h_hrx, w_hrx = guide.size()

        N = self.box_filter1(guide.data.new().resize_((1, 3, h_hrx, w_hrx)).fill_(1.0))
        ## mean_x
        mean_x1 = self.box_filter1(guide)/N
        ## mean_y
        mean_y1 = self.box_filter1(src)/N
        ## cov_xy
        cov_xy1 = self.box_filter1(guide * src)/N - mean_x1 * mean_y1
        ## var_x
        var_x1  = self.box_filter1(guide * guide)/N - mean_x1 * mean_x1

        N = self.box_filter2(guide.data.new().resize_((1, 3, h_hrx, w_hrx)).fill_(1.0))
        ## mean_x
        mean_x2 = self.box_filter2(guide)/N
        ## mean_y
        mean_y2 = self.box_filter2(src)/N
        ## cov_xy
        cov_xy2 = self.box_filter2(guide * src)/N - mean_x2 * mean_y2
        ## var_x
        var_x2  = self.box_filter2(guide * guide)/N - mean_x2 * mean_x2
        cov_xy2 = F.interpolate(cov_xy2, size=cov_xy1.size()[2:], mode='bilinear', align_corners=True) 
        var_x2 = F.interpolate(var_x2, size=var_x1.size()[2:], mode='bilinear', align_corners=True) 

        N = self.box_filter3(guide.data.new().resize_((1, 3, h_hrx, w_hrx)).fill_(1.0))
        ## mean_x
        mean_x3 = self.box_filter3(guide)/N
        ## mean_y
        mean_y3 = self.box_filter3(src)/N
        ## cov_xy
        cov_xy3 = self.box_filter3(guide * src)/N - mean_x3 * mean_y3
        ## var_x
        var_x3  = self.box_filter3(guide * guide)/N - mean_x3 * mean_x3
        cov_xy3 = F.interpolate(cov_xy3, size=cov_xy1.size()[2:], mode='bilinear', align_corners=True) 
        var_x3 = F.interpolate(var_x3, size=var_x1.size()[2:], mode='bilinear', align_corners=True) 

        N = self.box_filter4(guide.data.new().resize_((1, 3, h_hrx, w_hrx)).fill_(1.0))
        ## mean_x
        mean_x4 = self.box_filter4(guide)/N
        ## mean_y
        mean_y4 = self.box_filter4(src)/N
        ## cov_xy
        cov_xy4 = self.box_filter4(guide * src)/N - mean_x4 * mean_y4
        ## var_x
        var_x4  = self.box_filter4(guide * guide)/N - mean_x4 * mean_x4
        cov_xy4 = F.interpolate(cov_xy4, size=cov_xy1.size()[2:], mode='bilinear', align_corners=True) 
        var_x4 = F.interpolate(var_x4, size=var_x1.size()[2:], mode='bilinear', align_corners=True) 

        ## A
        A = self.conv_a(torch.cat([cov_xy1, var_x1, cov_xy2, var_x2, cov_xy3, var_x3, cov_xy4, var_x4], dim=1))
        ## b
        b = mean_y1 - A * mean_x1

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A * guide + mean_b
    
class ConvGuidedFilterASPP2(nn.Module):
    def __init__(self, radius=[4,32,64,128], norm=nn.BatchNorm2d):
        super(ConvGuidedFilterASPP2, self).__init__()

        self.box_filter1 = nn.Conv2d(3, 3, kernel_size=radius[0], stride=radius[0]//2, bias=False, groups=3, padding_mode='replicate')
        self.box_filter2 = nn.Conv2d(3, 3, kernel_size=radius[1], stride=radius[1]//2, bias=False, groups=3, padding_mode='replicate')
        self.box_filter3 = nn.Conv2d(3, 3, kernel_size=radius[2], stride=radius[2]//2, bias=False, groups=3, padding_mode='replicate')
        self.box_filter4 = nn.Conv2d(3, 3, kernel_size=radius[3], stride=radius[3]//2, bias=False, groups=3, padding_mode='replicate')
        self.conv_a = nn.Sequential(nn.Conv2d(24, 48, kernel_size=1, bias=False),
                                    norm(48),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(48, 48, kernel_size=1, bias=False),
                                    norm(48),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(48, 3, kernel_size=1, bias=False))
        self.box_filter1.weight.data[...] = 1.0
        self.box_filter2.weight.data[...] = 1.0
        self.box_filter3.weight.data[...] = 1.0
        self.box_filter4.weight.data[...] = 1.0

    def forward(self, guide, src):
        _, _, h_hrx, w_hrx = guide.size()

        N = self.box_filter1(guide.data.new().resize_((1, 3, h_hrx, w_hrx)).fill_(1.0))
        ## mean_x
        mean_x1 = self.box_filter1(guide)/N
        ## mean_y
        mean_y1 = self.box_filter1(src)/N
        ## cov_xy
        cov_xy1 = self.box_filter1(guide * src)/N - mean_x1 * mean_y1
        ## var_x
        var_x1  = self.box_filter1(guide * guide)/N - mean_x1 * mean_x1

        N = self.box_filter2(guide.data.new().resize_((1, 3, h_hrx, w_hrx)).fill_(1.0))
        ## mean_x
        mean_x2 = self.box_filter2(guide)/N
        ## mean_y
        mean_y2 = self.box_filter2(src)/N
        ## cov_xy
        cov_xy2 = self.box_filter2(guide * src)/N - mean_x2 * mean_y2
        ## var_x
        var_x2  = self.box_filter2(guide * guide)/N - mean_x2 * mean_x2
        cov_xy2 = F.interpolate(cov_xy2, size=cov_xy1.size()[2:], mode='bilinear', align_corners=True) 
        var_x2 = F.interpolate(var_x2, size=var_x1.size()[2:], mode='bilinear', align_corners=True) 

        N = self.box_filter3(guide.data.new().resize_((1, 3, h_hrx, w_hrx)).fill_(1.0))
        ## mean_x
        mean_x3 = self.box_filter3(guide)/N
        ## mean_y
        mean_y3 = self.box_filter3(src)/N
        ## cov_xy
        cov_xy3 = self.box_filter3(guide * src)/N - mean_x3 * mean_y3
        ## var_x
        var_x3  = self.box_filter3(guide * guide)/N - mean_x3 * mean_x3
        cov_xy3 = F.interpolate(cov_xy3, size=cov_xy1.size()[2:], mode='bilinear', align_corners=True) 
        var_x3 = F.interpolate(var_x3, size=var_x1.size()[2:], mode='bilinear', align_corners=True) 

        N = self.box_filter4(guide.data.new().resize_((1, 3, h_hrx, w_hrx)).fill_(1.0))
        ## mean_x
        mean_x4 = self.box_filter4(guide)/N
        ## mean_y
        mean_y4 = self.box_filter4(src)/N
        ## cov_xy
        cov_xy4 = self.box_filter4(guide * src)/N - mean_x4 * mean_y4
        ## var_x
        var_x4  = self.box_filter4(guide * guide)/N - mean_x4 * mean_x4
        cov_xy4 = F.interpolate(cov_xy4, size=cov_xy1.size()[2:], mode='bilinear', align_corners=True) 
        var_x4 = F.interpolate(var_x4, size=var_x1.size()[2:], mode='bilinear', align_corners=True) 

        ## A
        A = self.conv_a(torch.cat([cov_xy1, var_x1, cov_xy2, var_x2, cov_xy3, var_x3, cov_xy4, var_x4], dim=1))
        ## b
        b = mean_y1 - A * mean_x1

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A * guide + mean_b

class GuidedFilterColor(nn.Module):
    def __init__(self, radius=32, norm=nn.BatchNorm2d):
        super(GuidedFilterColor, self).__init__()

        self.box_filter_3 = nn.Conv2d(3, 3, kernel_size=radius, stride=radius//2, bias=False, groups=3)
        self.box_filter_1 = nn.Conv2d(1, 1, kernel_size=radius, stride=radius//2, bias=False)
        self.conv_a = nn.Sequential(nn.Conv2d(6, 32, kernel_size=1, padding=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=1, padding=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 3, kernel_size=1, padding=1, bias=False))
        self.box_filter_3.weight.data[...] = 1.0

    def forward(self, guide, src, eps=1e-4):
        _, _, h_hrx, w_hrx = guide.size()

        guide_r, guide_g, guide_b = torch.chunk(guide, 3, 1) # b x 1 x H x W
        ones = torch.ones_like(guide_r)
        # N = boxfilter2d(ones, radius)
        N = self.box_filter_1(ones)

        mean_I = self.box_filter_3(guide) / N # b x 3 x H x W
        mean_I_r, mean_I_g, mean_I_b = torch.chunk(mean_I, 3, 1) # b x 1 x H x W

        mean_p = self.box_filter_3(src) / N # b x C x H x W

        mean_Ip_r = self.box_filter_3(guide_r * src) / N # b x C x H x W
        mean_Ip_g = self.box_filter_3(guide_g * src) / N # b x C x H x W
        mean_Ip_b = self.box_filter_3(guide_b * src) / N # b x C x H x W

        cov_Ip_r = mean_Ip_r - mean_I_r * mean_p # b x C x H x W
        cov_Ip_g = mean_Ip_g - mean_I_g * mean_p # b x C x H x W
        cov_Ip_b = mean_Ip_b - mean_I_b * mean_p # b x C x H x W

        var_I_rr = self.box_filter_1(guide_r * guide_r) / N - mean_I_r * mean_I_r + eps # b x 1 x H x W
        var_I_rg = self.box_filter_1(guide_r * guide_g) / N - mean_I_r * mean_I_g # b x 1 x H x W
        var_I_rb = self.box_filter_1(guide_r * guide_b) / N - mean_I_r * mean_I_b # b x 1 x H x W
        var_I_gg = self.box_filter_1(guide_g * guide_g) / N - mean_I_g * mean_I_g + eps # b x 1 x H x W
        var_I_gb = self.box_filter_1(guide_g * guide_b) / N - mean_I_g * mean_I_b # b x 1 x H x W
        var_I_bb = self.box_filter_1(guide_b * guide_b) / N - mean_I_b * mean_I_b + eps # b x 1 x H x W

        # determinant
        cov_det = var_I_rr * var_I_gg * var_I_bb \
            + var_I_rg * var_I_gb * var_I_rb \
                + var_I_rb * var_I_rg * var_I_gb \
                    - var_I_rb * var_I_gg * var_I_rb \
                        - var_I_rg * var_I_rg * var_I_bb \
                            - var_I_rr * var_I_gb * var_I_gb # b x 1 x H x W

        # inverse
        inv_var_I_rr = (var_I_gg * var_I_bb - var_I_gb * var_I_gb) / cov_det # b x 1 x H x W
        inv_var_I_rg = - (var_I_rg * var_I_bb - var_I_rb * var_I_gb) / cov_det # b x 1 x H x W
        inv_var_I_rb = (var_I_rg * var_I_gb - var_I_rb * var_I_gg) / cov_det # b x 1 x H x W
        inv_var_I_gg = (var_I_rr * var_I_bb - var_I_rb * var_I_rb) / cov_det # b x 1 x H x W
        inv_var_I_gb = - (var_I_rr * var_I_gb - var_I_rb * var_I_rg) / cov_det # b x 1 x H x W
        inv_var_I_bb = (var_I_rr * var_I_gg - var_I_rg * var_I_rg) / cov_det # b x 1 x H x W

        inv_sigma = torch.stack([
            torch.stack([inv_var_I_rr, inv_var_I_rg, inv_var_I_rb], 1),
            torch.stack([inv_var_I_rg, inv_var_I_gg, inv_var_I_gb], 1),
            torch.stack([inv_var_I_rb, inv_var_I_gb, inv_var_I_bb], 1)
        ], 1).squeeze(-3) # b x 3 x 3 x H x W

        cov_Ip = torch.stack([cov_Ip_r, cov_Ip_g, cov_Ip_b], 1) # b x 3 x C x H x W

        A = torch.einsum("bichw,bijhw->bjchw", (cov_Ip, inv_sigma))
        b = mean_p - A[:, 0] * mean_I_r - A[:, 1] * mean_I_g - A[:, 2] * mean_I_b # b x C x H x W

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A * guide + mean_b
    
class DenseGuidedFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = nn.Sequential(nn.Conv2d(6, 32, kernel_size=3, padding=1),
                                    ResBlock(32, 32),
                                    Normalize(32),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(32, 6, kernel_size=3, padding=1))

    def forward(self, x_lr, y_lr, x_hr):
        mean_A, mean_b = torch.chunk(self.conv_a(torch.cat([x_lr, y_lr], dim=1)), 2, dim=1)

        return mean_A * x_hr + mean_b
    