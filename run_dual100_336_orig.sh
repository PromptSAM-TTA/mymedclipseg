#!/usr/bin/env bash
# DUAL multi-class FDI @336 (from ORIGINAL high-res data), 100 epochs, lr 2e-4.
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate medclipseg

cd /workspace/code/MedCLIPSeg-main

CONFIG=configs/DUAL100_336_orig.yaml
OUTDIR=./dual100_336_orig_lr2e4
SEED=1
mkdir -p logs

echo "[$(date '+%F %T')] === TRAIN DUAL100_336_orig START ==="
python train.py --config-file "$CONFIG" --output-dir "$OUTDIR" --seed "$SEED" \
  2>&1 | tee "logs/train_DUAL100_336_orig_seed1.log"
echo "[$(date '+%F %T')] === TRAIN done ==="

echo "[$(date '+%F %T')] === EVAL DUAL100_336_orig START ==="
python eval_inredd.py --config-file "$CONFIG" --output-dir "$OUTDIR" --seed "$SEED" \
  2>&1 | tee "logs/eval_DUAL100_336_orig_seed1.log"
echo "[$(date '+%F %T')] === EVAL done ==="
