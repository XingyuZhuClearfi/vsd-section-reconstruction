"""Denoising diffusion U-Net used as the 2D judge, with the noise schedule, the inpainting conditioning (known / band / shell
masks, boundary-weighted loss) and the training-mask sampler.  Extracted from the original training script; the K-phase
versions of the conditioning and loss are in multiphase_common.py."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        scale = math.log(10000.0) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=t.device) * -scale)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.temb = nn.Linear(temb_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).reshape(batch, 3, channels, height * width).unbind(1)
        attn = torch.softmax(torch.einsum("bcn,bcm->bnm", q, k) / (channels ** 0.5), dim=-1)
        out = torch.einsum("bnm,bcm->bcn", attn, v).reshape(batch, channels, height, width)
        return x + self.proj(out)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim, use_attn):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels, temb_dim)
        self.attn = AttentionBlock(out_channels) if use_attn else None

    def forward(self, x, temb):
        x = self.res(x, temb)
        if self.attn is not None:
            x = self.attn(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, base=64, ch_mult=(1, 2, 4, 4), num_res_blocks=2, attn_res=(16,), img_res=128):
        super().__init__()
        temb_dim = base * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base),
            nn.Linear(base, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )
        self.init_conv = nn.Conv2d(5, base, 3, padding=1)

        self.down_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels = [base]
        cur = base
        res = img_res
        for level, mult in enumerate(ch_mult):
            out = base * mult
            stage = nn.ModuleList()
            for _ in range(num_res_blocks):
                stage.append(Block(cur, out, temb_dim, res in attn_res))
                cur = out
                skip_channels.append(cur)
            self.down_stages.append(stage)
            if level != len(ch_mult) - 1:
                self.downsamples.append(Downsample(cur))
                skip_channels.append(cur)
                res //= 2
            else:
                self.downsamples.append(None)

        self.mid1 = ResBlock(cur, cur, temb_dim)
        self.mid_attn = AttentionBlock(cur)
        self.mid2 = ResBlock(cur, cur, temb_dim)

        self.up_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mult))):
            out = base * mult
            stage = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                stage.append(Block(cur + skip_channels.pop(), out, temb_dim, res in attn_res))
                cur = out
            self.up_stages.append(stage)
            if level != 0:
                self.upsamples.append(Upsample(cur))
                res *= 2
            else:
                self.upsamples.append(None)

        self.out_norm = nn.GroupNorm(8, cur)
        self.out_conv = nn.Conv2d(cur, 1, 3, padding=1)

    def forward(self, x, cond, t):
        temb = self.time_mlp(t)
        x = self.init_conv(torch.cat([x, cond], dim=1))
        skips = [x]
        for stage, downsample in zip(self.down_stages, self.downsamples):
            for block in stage:
                x = block(x, temb)
                skips.append(x)
            if downsample is not None:
                x = downsample(x)
                skips.append(x)
        x = self.mid1(x, temb)
        x = self.mid_attn(x)
        x = self.mid2(x, temb)
        for stage, upsample in zip(self.up_stages, self.upsamples):
            for block in stage:
                x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, temb)
            if upsample is not None:
                x = upsample(x)
        return self.out_conv(F.silu(self.out_norm(x)))


def make_schedule(timesteps, device):
    betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas_cumprod


def q_sample(x0, t, noise, alphas_cumprod):
    a = alphas_cumprod[t].view(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1.0 - a).sqrt() * noise


def dilate_mask(mask, radius):
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)


def build_hard_band(known, band_radius=4):
    core = known
    band = dilate_mask(known, band_radius)
    shell = (band - core).clamp(0.0, 1.0)
    return core, band, shell


def build_conditioning(x0, known, band_radius=4):
    core, band, shell = build_hard_band(known, band_radius=band_radius)
    observed = x0 * core
    return torch.cat([observed, core, band, shell], dim=1)


def diffusion_loss(prediction, target_noise, known, band_radius=4, boundary_loss_weight=1.5):
    if boundary_loss_weight <= 0 or known.max() <= 0:
        return F.mse_loss(prediction, target_noise)
    _, _, shell = build_hard_band(known, band_radius=band_radius)
    weight = 1.0 + boundary_loss_weight * shell
    return ((prediction - target_noise).pow(2) * weight).sum() / weight.sum().clamp_min(1.0)


def case_to_mask(case_name, masks, real):
    if case_name == "all_unknown":
        return torch.ones_like(real)
    if case_name == "all_known":
        return torch.zeros_like(real)
    if not case_name.startswith("mask_"):
        raise ValueError(f"Unknown case: {case_name}")
    mask_index = int(case_name.split("_", 1)[1])
    return masks[mask_index:mask_index + 1].to(real.device)


def resize_masks(masks, image_size, device):
    masks = masks.to(device)
    if masks.shape[-1] == image_size and masks.shape[-2] == image_size:
        return masks
    return F.interpolate(masks, size=(image_size, image_size), mode="nearest")


def sample_training_mask(batch_size, masks, device, all_unknown_prob=0.25, all_known_prob=0.10, low_known_prob=0.0, low_known_max_ratio=0.10):
    indices = torch.randint(0, masks.shape[0], (batch_size,), device=device)
    sampled = masks[indices].to(device).clone()
    selector = torch.rand(batch_size, 1, 1, 1, device=device)
    all_known = selector < all_known_prob
    all_unknown = (selector >= all_known_prob) & (selector < all_known_prob + all_unknown_prob)
    low_known = (selector >= all_known_prob + all_unknown_prob) & (selector < all_known_prob + all_unknown_prob + low_known_prob)
    if low_known_prob > 0:
        known_ratios = (1.0 - masks.to(device)).flatten(1).mean(dim=1)
        low_indices = torch.nonzero((known_ratios > 0) & (known_ratios <= low_known_max_ratio), as_tuple=False).flatten()
        if low_indices.numel() > 0:
            low_sampled_indices = low_indices[torch.randint(0, low_indices.numel(), (batch_size,), device=device)]
            low_sampled = masks[low_sampled_indices].to(device)
            sampled = torch.where(low_known, low_sampled, sampled)
    sampled = torch.where(all_known, torch.zeros_like(sampled), sampled)
    sampled = torch.where(all_unknown, torch.ones_like(sampled), sampled)
    return sampled


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for ep, p in zip(ema_model.parameters(), model.parameters()):
        ep.mul_(decay).add_(p, alpha=1.0 - decay)
    for eb, b in zip(ema_model.buffers(), model.buffers()):
        eb.copy_(b)
