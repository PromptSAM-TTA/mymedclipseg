"""Aggregate 5-fold binary metrics: mean +- std of mDice/mIoU."""
import json
import sys
import numpy as np

outdir = sys.argv[1]
dices, ious = [], []
for f in range(5):
    p = f"{outdir}/KAGGLE_binary_fold{f}/seg_results/seed1/metrics_binary.json"
    m = json.load(open(p))
    dices.append(m["mDice"])
    ious.append(m["mIoU"])
    print(f"fold{f}: mDice={m['mDice']:.4f}  mIoU={m['mIoU']:.4f}")

print("----")
print(f"mDice = {np.mean(dices):.4f} +- {np.std(dices):.4f}")
print(f"mIoU  = {np.mean(ious):.4f} +- {np.std(ious):.4f}")
