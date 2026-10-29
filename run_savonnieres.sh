#!/bin/bash
# The Savonnières example of the README, end to end: ground truth, training crops, judge, reconstruction, evaluation.
# RAW is the segmented volume of Digital Rocks Portal project 72 (1000^3, uint8; raw values 0 macropore, 1 microporous, 9 calcite).
set -e
RAW=${RAW:-SAVII2_mid_1000x1000x1000x8b-3phase-cleaner.dat}
SEED=${SEED:-2001}
D=data/rocks

python make_gt.py --raw $RAW --shape 1000 1000 1000 --phases 0,1,9 --scale 2.5 --out $D/sav3_gt_labels.npy
python build_rock_dataset_mp.py --raw $RAW --shape 1000 1000 1000 --phases 0,1,9 --name sav3 \
    --slices z:200,z:500,z:800,y:300,y:700,x:300,x:700 --scale 2.5 --crop 128 --per-angle 13 --max-crops 24000 --axis-mode random
python train_diffusion_mp.py --data $D/sav3_tensor.pt --n-phases 3 --steps 20000 --save-every 0 --all-unknown-prob 0.5 \
    --output $D/outputs_diffusion_sav3 --checkpoint $D/checkpoints_diffusion_sav3
python recon_vsd_voxel_mp.py --data $D/sav3_tensor.pt --model $D/checkpoints_diffusion_sav3/ema.pth --n-phases 3 \
    --size 192 --crop-out 128 --sections 24 --micro-batch 24 --micro-sleep-ms 0 --ramp-steps 0 --duty 1.0 \
    --ema 0.9975 --frac-lambda-rate 1e-6 --fake-train decoder --iters 8000 --ckpt-every 1000 --resume \
    --seed $SEED --output $D/recon_sav3_s$SEED
python eval_mp2.py --gt $D/sav3_gt_labels.npy --recon $D/recon_sav3_s$SEED/labels.npy --pore-phase 0 \
    --name sav3_s$SEED --json $D/eval_sav3_s$SEED.json --sheet $D/eval_sav3_s$SEED.png
