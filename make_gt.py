"""Ground-truth label volume at the judge's scale, for eval_mp2.py.

Downscales the whole segmented volume with the same rule as build_rock_dataset_mp.py uses for the training slices:
each phase is resampled linearly as an indicator image, then the label is the phase with the largest fraction.

usage: python make_gt.py --raw SAVII2.dat --shape 1000 1000 1000 --phases 0,1,9 --scale 2.5 --out data/rocks/sav3_gt_labels.npy
"""
import argparse, os, numpy as np
from scipy.ndimage import zoom

p = argparse.ArgumentParser()
p.add_argument("--raw", required=True); p.add_argument("--shape", type=int, nargs=3, required=True)
p.add_argument("--phases", required=True, help="raw values of the phases in label order, e.g. 0,1,9 -> labels 0,1,2")
p.add_argument("--scale", type=float, default=1.0); p.add_argument("--out", required=True)
a = p.parse_args()
vals = [int(x) for x in a.phases.split(",")]; K = len(vals)
v = np.fromfile(a.raw, dtype=np.uint8).reshape(a.shape); lut = np.zeros(256, np.uint8)
for i, val in enumerate(vals): lut[val] = i
lab = lut[v]; del v
if a.scale > 1:
    gt = np.stack([zoom(((lab == k) * 255).astype(np.uint8), 1 / a.scale, order=1) for k in range(K)]).argmax(0).astype(np.uint8)
else:
    gt = lab
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); np.save(a.out, gt)
print(f"{a.out}: shape {gt.shape}, phase fractions {np.round(np.bincount(gt.ravel(), minlength=K) / gt.size, 3).tolist()}")
