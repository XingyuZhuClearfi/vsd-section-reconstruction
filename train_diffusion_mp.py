"""K-phase version of train_diffusion.py: same schedule, same inpainting-style training (random masks,
25% all-unknown, 10% all-known, boundary-weighted loss), same UNet except the first/last convolution.
Data: uint8 label tensor (N,1,S,S), S = 128 by default with values 0..K-1 (built by build_rock_dataset_mp.py)."""

import argparse
import copy
import os

import numpy as np
import torch
import gpu_guard  # noqa: F401  (caps CUDA memory below VRAM; see gpu_guard.py)
from PIL import Image

from multiphase_common import UNetMP, labels_to_x, build_conditioning_mp, diffusion_loss_mp, ddim_sample_mp
from train_diffusion import make_schedule, q_sample, resize_masks, sample_training_mask, update_ema
from util.mask import gen_masks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--n-phases", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--save-every", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--all-unknown-prob", type=float, default=0.25)
    p.add_argument("--all-known-prob", type=float, default=0.10)
    p.add_argument("--hard-band-radius", type=int, default=4)
    p.add_argument("--boundary-loss-weight", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=410000)
    return p.parse_args()


def save_samples(model, args, alphas, device, path, size=128):
    model.eval()
    lab = ddim_sample_mp(model, 8, args.n_phases, size, alphas, 50, device).cpu().numpy()
    model.train()
    grey = np.linspace(255, 0, args.n_phases).astype(np.uint8)
    tiles = [grey[l] for l in lab]
    sheet = np.full((size, 8 * (size + 3)), 60, np.uint8)
    for i, t in enumerate(tiles):
        sheet[:, i * (size + 3):i * (size + 3) + size] = t
    Image.fromarray(sheet).save(path)
    fr = np.bincount(lab.reshape(-1), minlength=args.n_phases) / lab.size
    print("sample phase fractions", np.round(fr, 3).tolist(), flush=True)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.checkpoint, exist_ok=True)
    data = torch.load(args.data).to(torch.uint8)
    fr = np.bincount(data.numpy().reshape(-1), minlength=args.n_phases) / data.numel()
    print("training data", tuple(data.shape), "phase fractions", np.round(fr, 3).tolist(), flush=True)
    alphas = make_schedule(args.timesteps, dev)
    model = UNetMP(args.n_phases, base=args.base_channels).to(dev)
    ema = copy.deepcopy(model)
    for q in ema.parameters():
        q.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    size = data.shape[-1]  # image size of the data (128 unless built with --crop)
    masks = resize_masks(gen_masks(), size, dev)
    n = data.shape[0]
    for step in range(args.steps):
        idx = torch.randint(0, n, (args.batch_size,))
        x0 = labels_to_x(data[idx].to(dev), args.n_phases)
        t = torch.randint(0, args.timesteps, (x0.shape[0],), device=dev)
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, noise, alphas)
        mask = sample_training_mask(x0.shape[0], masks, dev, all_unknown_prob=args.all_unknown_prob, all_known_prob=args.all_known_prob)
        known = 1 - mask
        cond = build_conditioning_mp(x0, known, band_radius=args.hard_band_radius)
        loss = diffusion_loss_mp(model(xt, cond, t), noise, known, band_radius=args.hard_band_radius, boundary_loss_weight=args.boundary_loss_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        update_ema(ema, model, args.ema_decay)
        if step == 0 or (step + 1) % 100 == 0:
            print(f"step {step + 1}/{args.steps} loss {loss.item():.4f}", flush=True)
        if args.save_every > 0 and (step + 1) % args.save_every == 0 and step + 1 < args.steps:
            os.makedirs(os.path.join(args.checkpoint, f"step_{step + 1}"), exist_ok=True)
            torch.save(ema.state_dict(), os.path.join(args.checkpoint, f"step_{step + 1}", "ema.pth"))
            save_samples(ema, args, alphas, dev, os.path.join(args.output, f"samples_step{step + 1}.png"), size)
    torch.save(ema.state_dict(), os.path.join(args.checkpoint, "ema.pth"))
    torch.save(model.state_dict(), os.path.join(args.checkpoint, "model.pth"))
    save_samples(ema, args, alphas, dev, os.path.join(args.output, "samples_final.png"), size)
    print("done", flush=True)


if __name__ == "__main__":
    main()
