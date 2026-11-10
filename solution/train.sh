#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-${ROOT}/.venv/bin/python}
DATA=${1:-${ROOT}/dataset/RECIST_train_npz}
OUTPUT=${2:-${ROOT}/artifacts/finetune}
GPUS=${GPUS:-2}
EPOCHS=${EPOCHS:-6}
MODEL_SIZE=${MODEL_SIZE:-tiny}
case "${MODEL_SIZE}" in
  small)
    CONFIG=configs/efficientmedsam_s_512_FLARE_RECIST.yaml
    DEFAULT_INIT=${ROOT}/checkpoints/eff_medsam2_small_FLARE25_RECIST_baseline.pt
    DEFAULT_RESOLUTION=512
    DEFAULT_NUM_FRAMES=8
    ;;
  tiny)
    CONFIG=configs/efficientmedsam_ti_512_FLARE_RECIST.yaml
    DEFAULT_INIT=${ROOT}/checkpoints/eff_medsam2_tiny_FLARE25_RECIST_baseline.pt
    DEFAULT_RESOLUTION=384
    DEFAULT_NUM_FRAMES=12
    ;;
  *) echo "MODEL_SIZE must be small or tiny" >&2; exit 2 ;;
esac
INIT=${INIT:-${DEFAULT_INIT}}
RESOLUTION=${RESOLUTION:-${DEFAULT_RESOLUTION}}
NUM_FRAMES=${NUM_FRAMES:-${DEFAULT_NUM_FRAMES}}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-6}
BASE_LR=${BASE_LR:-5e-6}
VISION_LR=${VISION_LR:-1e-6}
TRAIN_LIST=${TRAIN_LIST:-${ROOT}/splits/train.txt}

# The trainer changes into baseline/ so canonicalize all user-supplied paths
# first. This also makes the persisted Hydra config directly reproducible.
DATA=$(realpath -m "${DATA}")
OUTPUT=$(realpath -m "${OUTPUT}")
INIT=$(realpath -m "${INIT}")
TRAIN_LIST=$(realpath -m "${TRAIN_LIST}")

test -d "${DATA}" || { echo "missing dataset: ${DATA}" >&2; exit 2; }
test -f "${INIT}" || { echo "missing checkpoint: ${INIT}" >&2; exit 2; }
test -f "${TRAIN_LIST}" || { echo "missing split: ${TRAIN_LIST}" >&2; exit 2; }

cd "${ROOT}/baseline"
exec "${PYTHON}" -m training.train \
  --config "${CONFIG}" \
  --use-cluster 0 \
  --num-gpus "${GPUS}" \
  --dataset-path "${DATA}" \
  --file-list "${TRAIN_LIST}" \
  --checkpoint-path "${INIT}" \
  --epochs "${EPOCHS}" \
  --resolution "${RESOLUTION}" \
  --num-frames "${NUM_FRAMES}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --num-workers "${TRAIN_NUM_WORKERS}" \
  --base-lr "${BASE_LR}" \
  --vision-lr "${VISION_LR}" \
  --output-path "${OUTPUT}"
