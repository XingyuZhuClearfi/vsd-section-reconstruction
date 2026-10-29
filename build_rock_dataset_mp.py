"""K-phase version of build_rock_dataset_rot.py: same slices / rotations / valid-region crops, but the image is
a label map 0..K-1.  Downscaling uses per-phase area fractions (each one-hot channel resampled linearly, then
argmax), the label analogue of 'area-average then threshold 0.5'.  Output: uint8 tensor (N,1,C,C), C = --crop (128)."""
import argparse, os, numpy as np, torch
from PIL import Image
import torchvision.transforms.functional as TF
from scipy.ndimage import minimum_filter, zoom

p = argparse.ArgumentParser()
p.add_argument("--raw", required=True); p.add_argument("--shape", type=int, nargs=3, required=True)
p.add_argument("--phases", required=True, help="raw values of the phases in label order, e.g. 0,1,9 -> labels 0,1,2")
p.add_argument("--name", required=True); p.add_argument("--slices", required=True, help="e.g. z:200,y:300,x:500")
p.add_argument("--scale", type=float, default=1.0); p.add_argument("--per-angle", type=int, default=10)
p.add_argument("--max-crops", type=int, default=24000); p.add_argument("--out", default="data/rocks")
p.add_argument("--crop", type=int, default=128, help="crop size (judge window)")
p.add_argument("--axis-stride", type=int, default=4, help="grid stride of the crops at 0/90/180/270 degrees (axis-mode grid)")
p.add_argument("--axis-mode", choices=["grid", "random"], default="grid",
               help="grid: dense grid crops at 0/90/180/270 (default, ~70-90%% of the set); random: treat those angles like every other angle")
a = p.parse_args()
C = a.crop
vals = [int(x) for x in a.phases.split(",")]; K = len(vals)
v = np.fromfile(a.raw, dtype=np.uint8).reshape(a.shape)
lut = np.full(256, 255, np.uint8)
for i, val in enumerate(vals): lut[val] = i
lab = lut[v]
assert (lab != 255).all(), "unexpected raw values"


def chord(phase):
    lens = []
    for ax in range(2):
        for k in range(0, phase.shape[ax], 4):
            line = np.take(phase, k, axis=ax); d = np.diff(np.concatenate([[0], line.astype(int), [0]])); lens += list(np.where(d == -1)[0] - np.where(d == 1)[0])
    return float(np.mean(lens)) if lens else 0.0


def downscale(sl, s):
    if s <= 1.0:
        return sl
    ch = np.stack([zoom((sl == k).astype(np.float32), 1.0 / s, order=1) for k in range(K)])
    return ch.argmax(0).astype(np.uint8)


specs = [(t.split(":")[0], int(t.split(":")[1])) for t in a.slices.split(",")]
rng = np.random.default_rng(0); crops = []; previews = []; fr_slices = []
for axname, k in specs:
    sl = np.take(lab, k, axis={"z": 0, "y": 1, "x": 2}[axname])
    fr_slices.append(np.bincount(sl.reshape(-1), minlength=K) / sl.size)
    print(f"slice {axname}={k}: fractions {np.round(fr_slices[-1], 3).tolist()}, chords per phase " + " ".join(f"{chord(sl == q):.1f}" for q in range(K)))
    sl = downscale(sl, a.scale); previews.append(sl)
    img = Image.fromarray(sl); W, H = img.size
    n_positions = max((W - C + 1) * (H - C + 1), 1)  # valid crop positions of the unrotated image (any aspect ratio)
    ones = Image.fromarray(np.full(img.size[::-1], 255, np.uint8))
    for ang in range(360):
        rot = np.array(TF.rotate(img, ang, interpolation=TF.InterpolationMode.NEAREST, expand=True))
        valid = np.array(TF.rotate(ones, ang, interpolation=TF.InterpolationMode.NEAREST, expand=True)) > 127
        ok = minimum_filter(valid.astype(np.uint8), size=C, mode="constant", cval=0)
        H2, W2 = valid.shape; cand = np.argwhere(ok[C // 2:H2 - C // 2 + 1, C // 2:W2 - C // 2 + 1] > 0)
        if len(cand) == 0:
            continue
        if ang % 90 == 0 and a.axis_mode == "grid":
            for ci in range(0, H2 - C + 1, a.axis_stride):
                for cj in range(0, W2 - C + 1, a.axis_stride):
                    crops.append(rot[ci:ci + C, cj:cj + C].copy())
            continue
        n = int(a.per_angle * len(cand) / n_positions)
        for idx in rng.choice(len(cand), min(n, len(cand)), replace=False):
            ci, cj = cand[idx]; crops.append(rot[ci:ci + C, cj:cj + C].copy())
grey = np.linspace(255, 0, K).astype(np.uint8)
h = max(q.shape[0] for q in previews); strip = np.full((h, sum(q.shape[1] + 4 for q in previews)), 60, np.uint8); x0 = 0
for q in previews:
    strip[:q.shape[0], x0:x0 + q.shape[1]] = grey[q]; x0 += q.shape[1] + 4
os.makedirs(a.out, exist_ok=True); Image.fromarray(strip).save(f"{a.out}/{a.name}_training_slices.png")
if len(crops) > a.max_crops: crops = [crops[i] for i in rng.choice(len(crops), a.max_crops, replace=False)]
t = torch.from_numpy(np.array(crops, dtype=np.uint8))[:, None]
torch.save(t, f"{a.out}/{a.name}_tensor.pt")
fr = np.bincount(t.numpy().reshape(-1), minlength=K) / t.numel()
print(f"{a.name}: {len(specs)} slices, K={K}, crops {tuple(t.shape)}, crop phase fractions {np.round(fr, 3).tolist()}, slice mean {np.round(np.mean(fr_slices, 0), 3).tolist()}")
