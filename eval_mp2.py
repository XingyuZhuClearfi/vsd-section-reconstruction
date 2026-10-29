"""Unified evaluation of a K-phase label volume against a real labelled volume at the same scale (replaces eval_mp.py).

Fixes of eval_mp.py: every count is normalised (specks per 128^2 slice area, cluster counts per 128^3), the two-point
function is the normalised autocovariance over r >= 1 (eval_mp's "S2 max error" was dominated by the r = 0 term, i.e.
the phase-fraction difference), the real reference is the WHOLE ground-truth volume (size-independent quantities) plus
the spread over random 128^3 real blocks (size-dependent ones), and a reconstruction larger than 128^3 is also evaluated
on its non-overlapping 128^3 blocks so that connectivity and counts are comparable with the real blocks.

usage: python eval_mp2.py --gt GT.npy --recon labels.npy [--pore-phase 0] [--n-blocks 8] [--sheet out.png] [--json out.json]
"""
import argparse
import json

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont

RMAX = 40
BLOCK = 128


def chord(ph, rng, n_lines=400):
    lens = []
    for ax in range(3):
        for _ in range(n_lines):
            idx = [rng.integers(0, s) for s in ph.shape]
            idx[ax] = slice(None)
            d = np.diff(np.concatenate([[0], ph[tuple(idx)].astype(int), [0]]))
            lens += list(np.where(d == -1)[0] - np.where(d == 1)[0])
    return float(np.mean(lens)) if lens else 0.0


def autocov(ph):
    """normalised autocovariance for r = 1..RMAX, averaged over the 3 axes, non-periodic:
    cov(a, b) / sqrt(var(a) var(b)) with a, b the two lag-shifted sub-arrays and their own means, so that the estimator is
    exactly symmetric under phase complement (1 - ph) and unbiased by edge truncation in small volumes."""
    out = []
    for r in range(1, RMAX + 1):
        acc = 0.0
        for ax in range(3):
            a = np.take(ph, range(ph.shape[ax] - r), axis=ax).astype(np.float32); b = np.take(ph, range(r, ph.shape[ax]), axis=ax).astype(np.float32)
            ma, mb = a.mean(), b.mean()
            acc += ((a * b).mean() - ma * mb) / max(np.sqrt((ma - ma ** 2) * (mb - mb ** 2)), 1e-9)
        out.append(acc / 3)
    return np.array(out)


def specks_per_area(L, k, rng, n_slices=10):
    out = []
    for ax in range(3):
        for i in rng.choice(L.shape[ax], min(n_slices, L.shape[ax]), replace=False):
            s = np.take(L, i, axis=ax) == k
            lab, _ = ndimage.label(s)
            sz = np.bincount(lab.ravel())[1:]
            out.append(np.sum(sz <= 4) * BLOCK ** 2 / s.size)
    return float(np.mean(out))


def interface(L, K):
    out = {}
    for i in range(K):
        for j in range(i + 1, K):
            c = 0
            for ax in range(3):
                a = np.take(L, range(L.shape[ax] - 1), axis=ax)
                b = np.take(L, range(1, L.shape[ax]), axis=ax)
                c += np.sum(((a == i) & (b == j)) | ((a == j) & (b == i)))
            out[f"{i}-{j}"] = float(c / L.size)
    return out


def size_free(L, K, rng):
    """quantities that do not depend on the volume size: fractions, chords, interface densities, autocovariance."""
    return {"frac": [float((L == k).mean()) for k in range(K)], "chord": [chord(L == k, rng) for k in range(K)],
            "iface": interface(L, K), "acov": [autocov(L == k) for k in range(K)]}


def size_dep(L, K, pore, rng):
    """quantities that depend on the volume size (evaluated on 128^3 blocks only): specks per 128^2, pore connectivity,
    pore bodies and enclosed pore share per 128^3."""
    m = L == pore
    lab, n = ndimage.label(m, structure=np.ones((3, 3, 3)))
    sz = np.bincount(lab.ravel())[1:]
    faces = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    touching = np.zeros(n + 1, bool)
    touching[faces] = True
    scale = BLOCK ** 3 / L.size
    return {"specks": [specks_per_area(L, k, rng) for k in range(K)],
            "conn": float(sz.max() / sz.sum()) if sz.size else 0.0,
            "pore_bodies": float(np.sum(sz >= 27) * scale),
            "enclosed": float(sz[~touching[1:]].sum() / max(sz.sum(), 1))}


def blocks_of(V, rng=None, n=None):
    """non-overlapping 128^3 blocks of V (all of them), or n random ones if n is given."""
    S = np.array(V.shape)
    if n is None:
        return [V[a:a + BLOCK, b:b + BLOCK, c:c + BLOCK] for a in range(0, S[0] - BLOCK + 1, BLOCK) for b in range(0, S[1] - BLOCK + 1, BLOCK) for c in range(0, S[2] - BLOCK + 1, BLOCK)]
    return [V[tuple(slice(o, o + BLOCK) for o in rng.integers(0, S - BLOCK + 1))] for _ in range(n)]


def rng_(seed=0):
    return np.random.default_rng(seed)


def fmt_range(vals, f):
    return f"{f(min(vals))}..{f(max(vals))}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True)
    p.add_argument("--recon", required=True)
    p.add_argument("--pore-phase", type=int, default=0)
    p.add_argument("--n-blocks", type=int, default=8)
    p.add_argument("--sheet", default="")
    p.add_argument("--json", default="")
    p.add_argument("--name", default="reconstruction")
    a = p.parse_args()
    gt = np.load(a.gt)
    rec = np.load(a.recon)
    K = int(max(gt.max(), rec.max())) + 1
    pore = a.pore_phase

    # real: whole volume for size-free quantities; random blocks for spread and size-dependent quantities
    real_whole = size_free(gt, K, rng_(0))
    real_blocks = blocks_of(gt, rng_(1), a.n_blocks)
    real_bf = [size_free(b, K, rng_(2 + i)) for i, b in enumerate(real_blocks)]
    real_bd = [size_dep(b, K, pore, rng_(20 + i)) for i, b in enumerate(real_blocks)]
    # reconstruction: whole volume for size-free; its own 128^3 blocks for size-dependent
    rec_whole = size_free(rec, K, rng_(0))
    rec_blocks = blocks_of(rec) if min(rec.shape) >= BLOCK else [rec]
    rec_bd = [size_dep(b, K, pore, rng_(40 + i)) for i, b in enumerate(rec_blocks)]

    W = 26
    print(f"{'metric':30s}{'real whole':>{W}s}{'real 128^3 blocks':>{W}s}{a.name + ' (' + 'x'.join(map(str, rec.shape)) + ')':>{W}s}{a.name + ' 128^3 blocks':>{W}s}")
    print("-- size-independent (whole volumes; blocks show natural spread)")
    for k in range(K):
        print(f"{'phase %d fraction' % k:30s}{real_whole['frac'][k]:>{W}.3f}{fmt_range([r['frac'][k] for r in real_bf], lambda v: f'{v:.3f}'):>{W}s}{rec_whole['frac'][k]:>{W}.3f}{'':>{W}s}")
    for k in range(K):
        print(f"{'phase %d mean chord' % k:30s}{real_whole['chord'][k]:>{W}.2f}{fmt_range([r['chord'][k] for r in real_bf], lambda v: f'{v:.1f}'):>{W}s}{rec_whole['chord'][k]:>{W}.2f}{'':>{W}s}")
    for key in real_whole["iface"]:
        print(f"{'interface %s per voxel' % key:30s}{real_whole['iface'][key]:>{W}.4f}{fmt_range([r['iface'][key] for r in real_bf], lambda v: f'{v:.4f}'):>{W}s}{rec_whole['iface'][key]:>{W}.4f}{'':>{W}s}")
    acov_err = []
    for k in range(K):
        err = float(np.abs(rec_whole["acov"][k] - real_whole["acov"][k]).max())
        spread = [float(np.abs(r["acov"][k] - real_whole["acov"][k]).max()) for r in real_bf]
        acov_err.append(err)
        print(f"{'phase %d autocov max err r>=1' % k:30s}{'0':>{W}s}{fmt_range(spread, lambda v: f'{v:.3f}'):>{W}s}{err:>{W}.3f}{'':>{W}s}")
    print("-- size-dependent (128^3 blocks only; per 128^2 slice / per 128^3)")
    for k in range(K):
        rv = [r["specks"][k] for r in real_bd]
        cv = [r["specks"][k] for r in rec_bd]
        print(f"{'phase %d specks per 128^2' % k:30s}{'':>{W}s}{(f'{np.mean(rv):.1f}  ' + fmt_range(rv, lambda v: f'{v:.0f}')):>{W}s}{'':>{W}s}{(f'{np.mean(cv):.1f}  ' + fmt_range(cv, lambda v: f'{v:.0f}')):>{W}s}")
    for key, name, f in [("conn", f"phase {pore} largest cluster share", lambda v: f"{v:.2f}"), ("pore_bodies", f"phase {pore} bodies >=27 vox", lambda v: f"{v:.0f}"), ("enclosed", f"phase {pore} enclosed share", lambda v: f"{v:.3f}")]:
        rv = [r[key] for r in real_bd]
        cv = [r[key] for r in rec_bd]
        print(f"{name:30s}{'':>{W}s}{(f(np.mean(rv)) + '  ' + fmt_range(rv, f)):>{W}s}{'':>{W}s}{(f(np.mean(cv)) + '  ' + fmt_range(cv, f)):>{W}s}")

    if a.json:
        out = {"real_whole": {k: v for k, v in real_whole.items() if k != "acov"}, "real_whole_acov": [x.tolist() for x in real_whole["acov"]],
               "rec_whole": {k: v for k, v in rec_whole.items() if k != "acov"}, "rec_whole_acov": [x.tolist() for x in rec_whole["acov"]],
               "acov_err": acov_err, "real_blocks": real_bd, "rec_blocks": rec_bd, "shape": list(rec.shape)}
        json.dump(out, open(a.json, "w"), indent=1)
    if a.sheet:
        rng = rng_(5)
        grey = np.linspace(255, 0, K).astype(np.uint8)
        T, pad, lab_w = BLOCK, 3, 190
        win = lambda V, ax: (lambda s, o: s[o[0]:o[0] + T, o[1]:o[1] + T])(np.take(V, rng.integers(0, V.shape[ax]), axis=ax), rng.integers(0, np.array(V.shape)[[i for i in range(3) if i != ax]] - T + 1))
        rows = [("REAL 3D slices", [win(gt, i % 3) for i in range(8)]), ("REAL 3D slices", [win(gt, i % 3) for i in range(8)]), (a.name, [win(rec, i % 3) for i in range(8)])]
        font = ImageFont.truetype("arial.ttf", 13)
        sheet = Image.new("RGB", (lab_w + 8 * (T + pad), len(rows) * (T + pad)), (60, 60, 60))
        d = ImageDraw.Draw(sheet)
        for r, (name, tiles) in enumerate(rows):
            d.text((6, r * (T + pad) + 50), name, fill=(255, 255, 0), font=font)
            for c, t in enumerate(tiles):
                sheet.paste(Image.fromarray(grey[t]).convert("RGB"), (lab_w + c * (T + pad), r * (T + pad)))
        sheet.save(a.sheet)
        print("saved", a.sheet)


if __name__ == "__main__":
    main()
