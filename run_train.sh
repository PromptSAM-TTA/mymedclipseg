#!/usr/bin/env bash
# Offline train + eval + visualize (generic across datasets).
#
# Usage:
#   CONFIG=configs/INREDD.yaml bash run_train.sh
#   CONFIG=configs/DUAL.yaml   bash run_train.sh
#   CONFIG=configs/DUAL.yaml OUTDIR=./run75 EPOCHS=75 SEED=1 bash run_train.sh
#
# 变量:
#   CONFIG  要用的配置文件（默认 configs/INREDD.yaml）
#   SEED    随机种子（默认 1）
#   OUTDIR  输出根目录（默认 .）；换一个值即可不覆盖旧结果
#   EPOCHS  可选，命令行覆盖 NUM_EPOCHS
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate medclipseg

cd /workspace/code/MedCLIPSeg-main

CONFIG="${CONFIG:-configs/INREDD.yaml}"
SEED="${SEED:-1}"
OUTDIR="${OUTDIR:-.}"
EPOCHS="${EPOCHS:-}"
LR="${LR:-}"

# 数据集短名（取自 config 文件名），用于日志前缀，避免不同数据集日志互踩
DATASET_NAME="$(basename "${CONFIG}" .yaml)"

OPTS=()
if [ -n "$EPOCHS" ]; then
  OPTS+=(TRAIN.NUM_EPOCHS "$EPOCHS")
fi
if [ -n "$LR" ]; then
  OPTS+=(TRAIN.LEARNING_RATE "$LR")
fi

TAG="${DATASET_NAME}_seed${SEED}"
if [ -n "$EPOCHS" ]; then
  TAG="${TAG}_ep${EPOCHS}"
fi
if [ -n "$LR" ]; then
  TAG="${TAG}_lr${LR}"
fi

mkdir -p logs

echo "[$(date '+%F %T')] === TRAIN ${TAG} (outdir=${OUTDIR}) START ==="
python train.py \
  --config-file "${CONFIG}" \
  --output-dir "${OUTDIR}" \
  --seed "${SEED}" \
  "${OPTS[@]}" \
  2>&1 | tee "logs/train_${TAG}.log"
echo "[$(date '+%F %T')] === TRAIN done ==="

echo "[$(date '+%F %T')] === EVAL ${TAG} START ==="
python eval_inredd.py \
  --config-file "${CONFIG}" \
  --output-dir "${OUTDIR}" \
  --seed "${SEED}" \
  2>&1 | tee "logs/eval_${TAG}.log"
echo "[$(date '+%F %T')] === EVAL done ==="

echo "[$(date '+%F %T')] === VISUALIZE ${TAG} START ==="
python visualize_test.py \
  --config-file "${CONFIG}" \
  --output-dir "${OUTDIR}" \
  --seed "${SEED}" \
  2>&1 | tee "logs/visualize_${TAG}.log"
echo "[$(date '+%F %T')] === VISUALIZE done ==="

echo "[$(date '+%F %T')] === ALL DONE ==="
