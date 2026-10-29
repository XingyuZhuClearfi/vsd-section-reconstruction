"""Shared pieces for the K-phase generalisation (new files only; the binary pipeline is untouched).

Representation: a K-phase image is stored as integer labels 0..K-1 and fed to networks as K one-hot
channels scaled to [-1, 1] (x = 2*onehot - 1).  For K = 2 with channel 1 = -channel 0 this reduces
exactly to the binary +-1 encoding used before.

Judge: the same UNet as the binary judge, with only the first and last convolutions widened:
  input  = K noisy channels + conditioning (K observed channels, core, band, shell) = 2K + 3
  output = K noise channels
For K = 1 these are 5 and 1, i.e. the original network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_diffusion import UNet, build_hard_band


class UNetMP(UNet):
    def __init__(self, n_phases, base=64, **kw):
        super().__init__(base=base, **kw)
        self.n_phases = n_phases
        self.init_conv = nn.Conv2d(2 * n_phases + 3, base, 3, padding=1)
        self.out_conv = nn.Conv2d(self.out_norm.num_channels, n_phases, 3, padding=1)


def labels_to_x(labels, n_phases):
    """labels (B,1,H,W) int -> (B,K,H,W) in [-1,1]."""
    oh = F.one_hot(labels[:, 0].long(), n_phases).permute(0, 3, 1, 2).float()
    return oh * 2 - 1


def build_conditioning_mp(x0, known, band_radius=4):
    core, band, shell = build_hard_band(known, band_radius=band_radius)
    return torch.cat([x0 * core, core, band, shell], dim=1)


def zero_cond(batch, n_phases, h, w, device):
    return torch.zeros(batch, n_phases + 3, h, w, device=device)


def diffusion_loss_mp(pred, noise, known, band_radius=4, boundary_loss_weight=1.5):
    if boundary_loss_weight <= 0 or known.max() <= 0:
        return F.mse_loss(pred, noise)
    _, _, shell = build_hard_band(known, band_radius=band_radius)
    weight = (1.0 + boundary_loss_weight * shell).expand_as(pred)
    return ((pred - noise).pow(2) * weight).sum() / weight.sum().clamp_min(1.0)


def hard_onehot(x):
    """x (..., K, ...) along dim=1 -> 2*onehot(argmax)-1, same shape."""
    idx = x.argmax(dim=1, keepdim=True)
    return torch.zeros_like(x).scatter_(1, idx, 1.0) * 2 - 1


def ste_onehot(x):
    """straight-through: forward = hard one-hot (+-1), backward = identity (K = 2 reduces to sign STE)."""
    return x + (hard_onehot(x) - x).detach()


@torch.no_grad()
def ddim_sample_mp(model, n, n_phases, size, alphas, steps, device):
    T = alphas.shape[0]
    ts = torch.linspace(0, T - 1, steps, device=device).round().long().tolist()
    x = torch.randn(n, n_phases, size, size, device=device)
    cond = zero_cond(n, n_phases, size, size, device)
    for i in reversed(range(len(ts))):
        t = ts[i]
        a_t = alphas[t]
        a_prev = alphas[ts[i - 1]] if i > 0 else torch.ones((), device=device)
        eps = model(x, cond, torch.full((n,), t, device=device, dtype=torch.long))
        x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return x.argmax(dim=1)  # labels
