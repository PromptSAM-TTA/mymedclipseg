#!/usr/bin/env bash
# Mask R-CNN on DUAL: train (with validation) -> test metrics.
set -euo pipefail

cd /workspace/code/MedCLIPSeg-main
export PYTHONUNBUFFERED=1
PY=/opt/miniconda3/envs/medclipseg/bin/python

echo "[$(date '+%F %T')] === TRAIN Mask R-CNN DUAL START ==="
PYTHONPATH=. $PY train_maskrcnn.py --epochs 30 --device cuda:1 --outdir ./maskrcnn_dual \
  2>&1 | tee logs/train_maskrcnn_dual.log
echo "[$(date '+%F %T')] === TRAIN done ==="

echo "[$(date '+%F %T')] === EVAL (test metrics) START ==="
PYTHONPATH=. $PY eval_maskrcnn.py --ckpt ./maskrcnn_dual/maskrcnn_dual_best.pth --device cuda:1 \
  2>&1 | tee logs/eval_maskrcnn_dual.log
echo "[$(date '+%F %T')] === EVAL done ==="
