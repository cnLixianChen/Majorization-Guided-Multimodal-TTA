#!/usr/bin/env bash
set -euo pipefail

# run_mg_mtta_multi_seed.sh
# Run MG-MTTA on ImageNet-C for one or multiple seeds.
# Usage example:
#   SEEDS="1" GPU=0 MODE=quick WEIGHTS=/path/to/weights bash scripts/run_mg_mtta_multi_seed.sh

usage() {
  cat <<EOF
Usage: SEEDS="1,2,3" MODE=quick|full GPU=0 BATCH_SIZE=64 NUM_WORKERS=4 WEIGHTS=/path/to/weights bash scripts/run_mg_mtta_multi_seed.sh
  MODE=quick -> CORRUPTION.NUM_EX=500 (fast smoke)
  MODE=full  -> CORRUPTION.NUM_EX=-1  (full dataset)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-1}"
MODE="${MODE:-full}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WEIGHTS="${WEIGHTS:-./checkpoints/open_clip/open_clip_model.safetensors}"
TS_GLOBAL="$(date +%y%m%d_%H%M%S)"

LOG_PARENT="${LOG_PARENT:-run_logs/multi_seed}"
LOG_DIR="${LOG_DIR:-${LOG_PARENT}/${TS_GLOBAL}}"
mkdir -p "$LOG_DIR"

if [[ "$MODE" == "quick" ]]; then
  CORRUPTION_NUM_EX="${CORRUPTION_NUM_EX:-500}"
else
  CORRUPTION_NUM_EX="${CORRUPTION_NUM_EX:--1}"
fi

IFS=',' read -r -a SEED_ARR <<< "$SEEDS"

echo "[INFO] MODE=$MODE SEEDS=${SEEDS} GPU=$GPU"
echo "[INFO] BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS WEIGHTS=$WEIGHTS"
echo "[INFO] CORRUPTION.NUM_EX=$CORRUPTION_NUM_EX LOG_DIR=$LOG_DIR"

for SEED in "${SEED_ARR[@]}"; do
  echo "[INFO] Starting seed=$SEED"
  TS="${TS_GLOBAL}_${SEED}"

  (
    CUDA_VISIBLE_DEVICES="$GPU" BATCLIP_CUDA_INDEX=0 \
    RNG_SEED="$SEED" python test_time.py --cfg cfgs/imagenet_c/mg_mtta_imagenet_c.yaml \
      CORRUPTION.DATASET imagenet_c MODEL.ARCH ViT-B-16-quickgelu \
      MODEL.WEIGHTS "$WEIGHTS" \
      TEST.BATCH_SIZE "$BATCH_SIZE" TEST.NUM_WORKERS "$NUM_WORKERS" \
      CORRUPTION.NUM_EX "$CORRUPTION_NUM_EX"
  ) 2>&1 | tee "$LOG_DIR/mg_mtta_imagenetc_seed_${SEED}_${TS}.log"

  STATUS=$?
  if [[ "$STATUS" -ne 0 ]]; then
    echo "[ERROR] Run failed for seed=$SEED (exit=$STATUS). See log: $LOG_DIR/mg_mtta_imagenetc_seed_${SEED}_${TS}.log"
    exit 1
  fi

  echo "[INFO] seed=$SEED DONE"
  echo "[INFO] log: $LOG_DIR/mg_mtta_imagenetc_seed_${SEED}_${TS}.log"
done

echo "[DONE] All seeds completed. Logs in $LOG_DIR"
