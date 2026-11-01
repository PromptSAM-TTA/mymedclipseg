#!/usr/bin/env bash
# KAGGLE full data (100%), test fixed. 100 epochs, lr 2e-4.
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate medclipseg

cd /workspace/code/MedCLIPSeg-main

CONFIG=configs/KAGGLE100.yaml
OUTDIR=./kaggle100_lr2e4
SEED=1
mkdir -p logs

echo "[$(date '+%F %T')] === TRAIN KAGGLE100 START ==="
python train.py --config-file "$CONFIG" --output-dir "$OUTDIR" --seed "$SEED" \
  2>&1 | tee "logs/train_KAGGLE100_seed1.log"
echo "[$(date '+%F %T')] === TRAIN done ==="

echo "[$(date '+%F %T')] === EVAL KAGGLE100 START ==="
python eval_inredd.py --config-file "$CONFIG" --output-dir "$OUTDIR" --seed "$SEED" \
  2>&1 | tee "logs/eval_KAGGLE100_seed1.log"
echo "[$(date '+%F %T')] === EVAL done ==="
