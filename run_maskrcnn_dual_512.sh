#!/usr/bin/env bash
# Mask R-CNN on DUAL @512x512: train (with validation) -> test metrics.
set -euo pipefail

cd /workspace/code/MedCLIPSeg-main
export PYTHONUNBUFFERED=1
PY=/opt/miniconda3/envs/medclipseg/bin/python

echo "[$(date '+%F %T')] === TRAIN Mask R-CNN DUAL @512 START ==="
PYTHONPATH=. $PY train_maskrcnn.py --epochs 30 --size 512 --batch-size 2 --device cuda:1 --outdir ./maskrcnn_dual_512 \
  2>&1 | tee logs/train_maskrcnn_dual_512.log
echo "[$(date '+%F %T')] === TRAIN done ==="

echo "[$(date '+%F %T')] === EVAL (test metrics @512) START ==="
PYTHONPATH=. $PY eval_maskrcnn.py --size 512 --ckpt ./maskrcnn_dual_512/maskrcnn_dual_best.pth --device cuda:1 \
  2>&1 | tee logs/eval_maskrcnn_dual_512.log
echo "[$(date '+%F %T')] === EVAL done ==="
