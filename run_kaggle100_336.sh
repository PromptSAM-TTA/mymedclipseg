#!/usr/bin/env bash
# KAGGLE full data @336x336, 100 epochs, lr 2e-4, prompt "tooth <FDI>".
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate medclipseg

cd /workspace/code/MedCLIPSeg-main

CONFIG=configs/KAGGLE100_336.yaml
OUTDIR=./kaggle100_336_lr2e4
SEED=1
mkdir -p logs

echo "[$(date '+%F %T')] === TRAIN KAGGLE100_336 START ==="
python train.py --config-file "$CONFIG" --output-dir "$OUTDIR" --seed "$SEED" \
  2>&1 | tee "logs/train_KAGGLE100_336_seed1.log"
echo "[$(date '+%F %T')] === TRAIN done ==="

echo "[$(date '+%F %T')] === EVAL KAGGLE100_336 START ==="
python eval_inredd.py --config-file "$CONFIG" --output-dir "$OUTDIR" --seed "$SEED" \
  2>&1 | tee "logs/eval_KAGGLE100_336_seed1.log"
echo "[$(date '+%F %T')] === EVAL done ==="
