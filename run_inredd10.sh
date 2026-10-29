#!/usr/bin/env bash
# INREDD 10% subset (train/val each 10% of original), test fixed. 100 epochs, lr 2e-4.
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate medclipseg

cd /workspace/code/MedCLIPSeg-main

CONFIG=configs/INREDD10.yaml
OUTDIR=./inredd10_lr2e4
SEED=1
mkdir -p logs

echo "[$(date '+%F %T')] === TRAIN INREDD10 START ==="
python train.py --config-file "$CONFIG" --output-dir "$OUTDIR" --seed "$SEED" \
  2>&1 | tee "logs/train_INREDD10_seed1.log"
echo "[$(date '+%F %T')] === TRAIN done ==="

echo "[$(date '+%F %T')] === EVAL INREDD10 START ==="
python eval_inredd.py --config-file "$CONFIG" --output-dir "$OUTDIR" --seed "$SEED" \
  2>&1 | tee "logs/eval_INREDD10_seed1.log"
echo "[$(date '+%F %T')] === EVAL done ==="
