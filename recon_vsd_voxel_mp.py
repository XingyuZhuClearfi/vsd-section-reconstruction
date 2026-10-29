"""K-phase version of recon_vsd_voxel.py (final recipe only: periodic cell + central crop, shared fake forward,
bf16, adaptive phase-fraction multipliers, power mitigations).  The binary script is untouched.

Generalisation, term by term (K = 2 reproduces the binary method):
  variable      u (K,S,S,S);  volume = 2*onehot(argmax_k u_k) - 1 with straight-through gradient (binary: sign(u))
  sections      random planes through the periodic cell, trilinear interpolation of all K channels with the
                same coordinates, then straight-through argmax (binary: interpolate, then sign)
  judge         K-channel 2D diffusion UNet (multiphase_common.UNetMP); VSD gradient (1-abar)(eps_real - eps_fake)
                on all K channels
  constraint    phase fractions f_k = mean((vol_k+1)/2), one adaptive multiplier per phase (binary: porosity)
"""

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import gpu_guard  # noqa: F401  (caps CUDA memory below VRAM; see gpu_guard.py)
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from multiphase_common import UNetMP, hard_onehot, ste_onehot, zero_cond
from train_diffusion import make_schedule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="uint8 label tensor used for the judge (for the target phase fractions)")
    p.add_argument("--model", required=True)
    p.add_argument("--n-phases", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--size", type=int, default=192)
    p.add_argument("--crop-out", type=int, default=128)
    p.add_argument("--iters", type=int, default=25000)
    p.add_argument("--sections", type=int, default=48)
    p.add_argument("--lr-u", type=float, default=0.05)
    p.add_argument("--lr-fake", type=float, default=3e-4)
    p.add_argument("--t-min", type=float, default=0.02)
    p.add_argument("--t-max", type=float, default=0.7)
    p.add_argument("--w-frac", type=float, default=20.0)
    p.add_argument("--frac-lambda-rate", type=float, default=2e-6)
    p.add_argument("--frac-lambda-max", type=float, default=5e-5)
    p.add_argument("--u-clamp", type=float, default=2.0)
    p.add_argument("--ema", type=float, default=0.995)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--max-chunk-pixels", type=int, default=0, help="if >0: chunk = floor(budget / (H*W)) sections, overrides --micro-batch")
    p.add_argument("--micro-sleep-ms", type=float, default=5.0)
    p.add_argument("--ramp-steps", type=int, default=600)
    p.add_argument("--duty", type=float, default=0.85)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--snapshot-steps", default="", help="comma list of steps at which argmax(u_ema) of the output crop is saved to <output>/snap_XXXXX/labels.npy")
    p.add_argument("--seed", type=int, default=2001)
    p.add_argument("--grad-mode", choices=["vsd", "sds"], default="vsd",
                   help="ablation: sds = (1-abar)(eps_real - noise) with no online fake model (frozen-score distillation)")
    p.add_argument("--judge-tile", type=int, default=0,
                   help="ablation: if >0, cut each S x S section into W x W tiles (random roll) so the judge sees only its training window")
    p.add_argument("--plane-mode", choices=["any", "axis"], default="any",
                   help="ablation: axis = judge only axis-aligned full planes (random axis and position, no in-plane rotation)")
    p.add_argument("--fake-train", choices=["all", "decoder", "lora"], default="all",
                   help="which parameters of the fake (current-distribution) model are trained: all (default), decoder = mid + up path + out, lora = frozen copy + low-rank adapters on every conv")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--stop-every", type=int, default=0, help="if >0: every N steps compare section statistics of u_ema with the training crops (logged as 'stop')")
    p.add_argument("--stop-patience", type=int, default=0, help="if >0: stop when the section distance has not improved for this many checks; output = u_ema at the best check")
    p.add_argument("--stop-min-steps", type=int, default=2000)
    p.add_argument("--stop-tol", type=float, default=0.02, help="relative improvement of the distance that counts as progress")
    return p.parse_args()


class LoRAConv(nn.Module):
    """frozen conv + trainable low-rank update B(A(x)); B starts at zero so the wrapped layer is initially unchanged."""

    def __init__(self, conv, rank):
        super().__init__()
        self.conv = conv
        self.down = nn.Conv2d(conv.in_channels, rank, conv.kernel_size, stride=conv.stride, padding=conv.padding, bias=False)
        self.up = nn.Conv2d(rank, conv.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.conv(x) + self.up(self.down(x))


def set_fake_trainable(fake, mode, rank):
    """returns the list of trainable parameters of the fake model for the chosen mode."""
    for q in fake.parameters():
        q.requires_grad_(False)
    if mode == "all":
        for q in fake.parameters():
            q.requires_grad_(True)
    elif mode == "decoder":
        for m in [fake.mid1, fake.mid_attn, fake.mid2, fake.up_stages, fake.upsamples, fake.out_norm, fake.out_conv]:
            for q in m.parameters():
                q.requires_grad_(True)
    elif mode == "lora":
        for name, m in list(fake.named_modules()):
            for cname, c in list(m.named_children()):
                if isinstance(c, nn.Conv2d) and c.kernel_size == (3, 3):
                    setattr(m, cname, LoRAConv(c, rank).to(c.weight.device))
        for m in fake.modules():
            if isinstance(m, LoRAConv):
                for q in list(m.down.parameters()) + list(m.up.parameters()):
                    q.requires_grad_(True)
    params = [q for q in fake.parameters() if q.requires_grad]
    print(f"fake model: train mode {mode}, trainable parameters {sum(q.numel() for q in params) / 1e6:.2f}M of {sum(q.numel() for q in fake.parameters()) / 1e6:.2f}M", flush=True)
    return params


def section_statistics(lab, K, rmax=32):
    """2D statistics of a batch of label images lab (n,H,W) long, as one vector: per phase [chord, acov(1..rmax)], per pair iface."""
    out = []
    for k in range(K):
        m = (lab == k)
        phi = m.float().mean().clamp_min(1e-6)
        chords, acov = [], []
        for ax in (1, 2):
            mm = m.movedim(ax, -1).float()
            d = torch.diff(torch.cat([torch.zeros_like(mm[..., :1]), mm, torch.zeros_like(mm[..., :1])], dim=-1), dim=-1)
            chords.append(mm.sum() / (d == 1).sum().clamp_min(1))
            acov.append(torch.stack([(mm[..., :-r] * mm[..., r:]).mean() for r in range(1, rmax + 1)]))
        out.append(torch.stack(chords).mean()[None])
        out.append(((torch.stack(acov).mean(0) - phi ** 2) / (phi - phi ** 2).clamp_min(1e-6)))
    for i in range(K):
        for j in range(i + 1, K):
            c = 0.0
            for ax in (1, 2):
                a = lab.movedim(ax, -1)[..., :-1]
                b = lab.movedim(ax, -1)[..., 1:]
                c = c + (((a == i) & (b == j)) | ((a == j) & (b == i))).float().sum()
            out.append((c / lab.numel())[None])
    return torch.cat(out)


def section_distance(v, ref, K, rmax=32):
    """distance between two statistics vectors: mean over phases of max |acov diff| + relative chord error + relative interface error."""
    i, acov_err, chord_err = 0, [], []
    for k in range(K):
        chord_err.append((v[i] - ref[i]).abs() / ref[i].clamp_min(1e-6)); i += 1
        acov_err.append((v[i:i + rmax] - ref[i:i + rmax]).abs().max()); i += rmax
    iface_err = ((v[i:] - ref[i:]).abs() / ref[i:].clamp_min(1e-6)).mean()
    a, c = torch.stack(acov_err).mean(), torch.stack(chord_err).mean()
    return float(a + c + iface_err), float(a), float(c), float(iface_err)


def periodic_sections_mp(vol, n, dev, plane_mode="any"):
    """vol (K,S,S,S) -> (n,K,S,S): random full planes (uniform normal, in-plane rotation, origin) through the
    periodic cell, trilinear interpolation of every channel with the same coordinates.
    plane_mode="axis": ablation, planes normal to a random coordinate axis, no in-plane rotation (axis slices only)."""
    K, S = vol.shape[0], vol.shape[-1]
    if plane_mode == "axis":
        ax = torch.randint(0, 3, (n,), device=dev)
        eye = torch.eye(3, device=dev)
        nrm, e1, e2 = eye[ax], eye[(ax + 1) % 3], eye[(ax + 2) % 3]
    else:
        nrm = F.normalize(torch.randn(n, 3, device=dev), dim=1)
        e1 = torch.randn(n, 3, device=dev)
        e1 = F.normalize(e1 - (e1 * nrm).sum(1, keepdim=True) * nrm, dim=1)
        e2 = torch.cross(nrm, e1, dim=1)
    origin = torch.rand(n, 3, device=dev) * S
    r = torch.arange(S, device=dev, dtype=torch.float32) - S / 2 + 0.5
    coords = origin[:, None, None, :] + r[None, :, None, None] * e1[:, None, None, :] + r[None, None, :, None] * e2[:, None, None, :]
    f = torch.floor(coords)
    w = coords - f
    f = f.long()
    flat = vol.reshape(K, -1)
    out = torch.zeros(K, n, S, S, device=dev)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix = (f[..., 0] + dx) % S
                iy = (f[..., 1] + dy) % S
                iz = (f[..., 2] + dz) % S
                wt = (w[..., 0] if dx else 1 - w[..., 0]) * (w[..., 1] if dy else 1 - w[..., 1]) * (w[..., 2] if dz else 1 - w[..., 2])
                out = out + wt[None] * flat[:, (ix * S + iy) * S + iz]
    return out.permute(1, 0, 2, 3)


def save_labels_png(lab2d_list, K, path):
    grey = np.linspace(255, 0, K).astype(np.uint8)
    S = lab2d_list[0].shape[0]
    sheet = np.full((S, len(lab2d_list) * (S + 3)), 60, np.uint8)
    for i, l in enumerate(lab2d_list):
        sheet[:, i * (S + 3):i * (S + 3) + S] = grey[l]
    Image.fromarray(sheet).save(path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    os.makedirs(args.output, exist_ok=True)
    K, S = args.n_phases, args.size
    data = torch.load(args.data).to(torch.uint8)
    target = torch.tensor(np.bincount(data.numpy().reshape(-1), minlength=K) / data.numel(), dtype=torch.float32, device=dev)
    print("target phase fractions", [round(float(x), 4) for x in target], flush=True)
    alphas = make_schedule(args.timesteps, dev)
    real = UNetMP(K).to(dev)
    real.load_state_dict(torch.load(args.model, map_location=dev))
    real.eval()
    for q in real.parameters():
        q.requires_grad_(False)
    fake = copy.deepcopy(real)
    fake.train()
    fake_params = set_fake_trainable(fake, args.fake_train, args.lora_rank)
    opt_fake = torch.optim.Adam(fake_params, lr=args.lr_fake, fused=True)
    # reference section statistics of the training crops (for --stop-every): a fixed random sample of crops
    stop_ref, stop_best, stop_best_it, u_best = None, None, 0, None
    if args.stop_every > 0:
        g = torch.Generator().manual_seed(123)
        ref_idx = torch.randperm(data.shape[0], generator=g)[:512]
        stop_ref = section_statistics(data[ref_idx, 0].long().to(dev), K)
        print("stop criterion: reference statistics from 512 training crops", flush=True)
    u = torch.randn(K, S, S, S, device=dev).clamp(-args.u_clamp, args.u_clamp).requires_grad_(True)
    u0_labels = u.detach().argmax(0)
    opt_u = torch.optim.Adam([u], lr=args.lr_u, fused=True)
    u_ema = u.detach().clone()
    lam = torch.zeros(K, device=dev)
    log, it0, start = [], 0, time.time()
    ckpt_path = os.path.join(args.output, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        with torch.no_grad():
            u.copy_(ck["u"]); u_ema.copy_(ck["u_ema"]); lam.copy_(ck["lam"])
        opt_u.load_state_dict(ck["opt_u"]); fake.load_state_dict(ck["fake"]); opt_fake.load_state_dict(ck["opt_fake"])
        torch.set_rng_state(ck["rng_cpu"].cpu()); torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        it0, log, start = ck["iter"], ck["log"], start - ck["elapsed"]
        print(f"resumed at iter {it0}", flush=True)

    def save_ckpt(it):
        tmp = ckpt_path + ".tmp"
        torch.save({"u": u.detach(), "u_ema": u_ema, "lam": lam, "opt_u": opt_u.state_dict(), "fake": fake.state_dict(), "opt_fake": opt_fake.state_dict(),
                    "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(), "iter": it, "log": log, "elapsed": time.time() - start, "args": vars(args)}, tmp)
        os.replace(tmp, ckpt_path)

    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    sections_full, n_sec = args.sections, None
    assert args.max_chunk_pixels > 0 or args.micro_batch <= args.sections, \
        f"--micro-batch ({args.micro_batch}) must not exceed --sections ({args.sections}): the ramp would silently judge micro-batch sections per step"
    snap_steps = set(int(x) for x in args.snapshot_steps.split(",") if x.strip())
    for it in range(it0 + 1, args.iters + 1):
        t_step0 = time.time()
        stage = min(3, 1 + int(3 * (it - 1) / args.ramp_steps)) if args.ramp_steps > 0 and it <= args.ramp_steps else 3
        new_n = max(args.micro_batch, sections_full * stage // 3)
        if new_n != n_sec:
            n_sec = new_n
            torch.cuda.empty_cache()
        vol = ste_onehot(u[None])[0]  # (K,S,S,S) in {-1,+1}, one +1 per voxel
        sec = periodic_sections_mp(vol, n_sec, dev, args.plane_mode)
        x0_all = ste_onehot(sec)
        if args.judge_tile > 0:
            Wt = args.judge_tile; m = x0_all.shape[-1] // Wt
            ri, rj = [int(v) for v in torch.randint(0, x0_all.shape[-1], (2,))]
            x0_all = torch.roll(x0_all, shifts=(ri, rj), dims=(-2, -1))[..., :m * Wt, :m * Wt]
            x0_all = x0_all.reshape(x0_all.shape[0], K, m, Wt, m, Wt).permute(0, 2, 4, 1, 3, 5).reshape(-1, K, Wt, Wt)
        n_items = x0_all.shape[0]
        frac = ((vol + 1) / 2).mean(dim=(1, 2, 3))  # (K,)
        opt_u.zero_grad(set_to_none=True)
        opt_fake.zero_grad(set_to_none=True)
        cs = max(1, args.max_chunk_pixels // (x0_all.shape[-1] * x0_all.shape[-2])) if args.max_chunk_pixels > 0 else args.micro_batch
        chunks = [x0_all[i:i + cs] for i in range(0, n_items, cs)]
        loss_fake = 0.0
        for x0 in chunks:
            torch.cuda.synchronize()
            time.sleep(args.micro_sleep_ms / 1000.0)
            b = x0.shape[0]
            t = torch.randint(int(args.t_min * args.timesteps), int(args.t_max * args.timesteps), (b,), device=dev)
            a_t = alphas[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x0)
            x_t = a_t.sqrt() * x0.detach() + (1 - a_t).sqrt() * noise
            cond = zero_cond(b, K, x0.shape[-2], x0.shape[-1], dev)
            with torch.no_grad(), amp:
                e_real = real(x_t, cond, t).float()
            if args.grad_mode == "sds":
                grad = (1 - a_t) * (e_real - noise)          # SDS: frozen judge only, no online fake model
            else:
                with amp:
                    e_fake = fake(x_t, cond, t)
                lf = F.mse_loss(e_fake.float(), noise) * (b / n_items)
                lf.backward()
                loss_fake += float(lf.detach())
                grad = (1 - a_t) * (e_real - e_fake.detach().float())
            x0.backward(grad / n_sec, retain_graph=True)
        h = frac.detach() - target
        lam = (lam + args.frac_lambda_rate * h).clamp(-args.frac_lambda_max, args.frac_lambda_max)
        frac_loss = args.w_frac * (frac - target).abs().sum() + (lam * 2.0 * vol[0].numel() * frac).sum()
        frac_loss.backward()
        opt_u.step()
        if args.grad_mode != "sds":
            opt_fake.step()
        with torch.no_grad():
            u.clamp_(-args.u_clamp, args.u_clamp)
            u_ema.mul_(args.ema).add_(u, alpha=1 - args.ema)
        stop_now = False
        if args.stop_every > 0 and it % args.stop_every == 0:
            with torch.no_grad():
                sec_lab = periodic_sections_mp(hard_onehot(u_ema[None])[0], 48, dev).argmax(1)
                # compare windows of the training-crop size, so that edge truncation of chords / correlations matches the reference
                Wc = data.shape[-1]
                oi, oj = torch.randint(0, S - Wc + 1, (2, 48))
                sec_lab = torch.stack([sec_lab[i, a:a + Wc, b:b + Wc] for i, (a, b) in enumerate(zip(oi.tolist(), oj.tolist()))])
                d, da, dc, di = section_distance(section_statistics(sec_lab, K), stop_ref, K)
            improved = stop_best is None or d < stop_best * (1 - args.stop_tol)
            if improved:
                stop_best, stop_best_it, u_best = d, it, u_ema.clone()
            stop_row = {"d": round(d, 4), "acov": round(da, 4), "chord": round(dc, 4), "iface": round(di, 4), "best_it": stop_best_it}
            if args.stop_patience > 0 and it >= args.stop_min_steps and (it - stop_best_it) >= args.stop_patience * args.stop_every:
                stop_now = True
        if it % args.log_every == 0 or it == 1 or stop_now:
            row = {"iter": it, "fake_loss": round(loss_fake, 4), "frac": [round(float(x), 4) for x in frac.detach()], "lambda": [float(f"{x:.1e}") for x in lam],
                   "changed_vs_init": round(float((u.detach().argmax(0) != u0_labels).float().mean()), 4), "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "time": round(time.time() - start)}
            if args.stop_every > 0 and it % args.stop_every == 0:
                row["stop"] = stop_row
            log.append(row)
            print(json.dumps(row), flush=True)
        if stop_now:
            print(f"auto-stop at iter {it}: no improvement of the section distance since iter {stop_best_it} (best {stop_best:.4f}); output = u_ema of iter {stop_best_it}", flush=True)
            with torch.no_grad():
                u_ema.copy_(u_best)
            save_ckpt(it)
            break
        if it % args.save_every == 0 or it == args.iters:
            lab = u_ema.argmax(0).cpu().numpy().astype(np.uint8)
            save_labels_png([lab[S // 2], lab[:, S // 2], lab[:, :, S // 2]], K, os.path.join(args.output, f"iter{it:05d}_xyz.png"))
        if it in snap_steps:
            lo_s = (S - args.crop_out) // 2
            lab_s = u_ema.argmax(0)[lo_s:lo_s + args.crop_out, lo_s:lo_s + args.crop_out, lo_s:lo_s + args.crop_out].cpu().numpy().astype(np.uint8)
            os.makedirs(os.path.join(args.output, f"snap_{it:05d}"), exist_ok=True)
            np.save(os.path.join(args.output, f"snap_{it:05d}", "labels.npy"), lab_s)
        if args.ckpt_every and (it % args.ckpt_every == 0 or it == args.iters):
            save_ckpt(it)
        if args.duty < 1.0:
            torch.cuda.synchronize()
            time.sleep((time.time() - t_step0) * (1.0 - args.duty) / args.duty)
    lab = u_ema.argmax(0).cpu().numpy().astype(np.uint8)
    np.save(os.path.join(args.output, "labels_full.npy"), lab)
    lo = (S - args.crop_out) // 2
    crop = lab[lo:lo + args.crop_out, lo:lo + args.crop_out, lo:lo + args.crop_out]
    np.save(os.path.join(args.output, "labels.npy"), crop)
    save_labels_png([crop[k] for k in (16, 48, 80, 112)] + [crop[:, k] for k in (32, 96)] + [crop[:, :, k] for k in (32, 96)], K, os.path.join(args.output, "final_slices.png"))
    json.dump(log, open(os.path.join(args.output, "log.json"), "w"), indent=1)
    print("crop phase fractions", np.round(np.bincount(crop.reshape(-1), minlength=K) / crop.size, 4).tolist(), "| cell", np.round(np.bincount(lab.reshape(-1), minlength=K) / lab.size, 4).tolist())
    print(f"saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
