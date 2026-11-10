#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${1:-${ROOT}/dataset/RECIST_train_npz}
OUTPUT=${2:-${ROOT}/artifacts/finetune-tiny-384}
GPU_IDS=${GPU_IDS:-0,1}
MAX_USED_MB=${MAX_USED_MB:-4096}
POLL_SECONDS=${POLL_SECONDS:-60}

if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to overwrite existing training output: ${OUTPUT}" >&2
  exit 2
fi

while true; do
  mapfile -t GPU_USED < <(
    nvidia-smi --id="${GPU_IDS}" --query-gpu=memory.used \
      --format=csv,noheader,nounits
  )
  if [[ ${#GPU_USED[@]} -eq 2 ]] \
      && (( GPU_USED[0] <= MAX_USED_MB )) \
      && (( GPU_USED[1] <= MAX_USED_MB )); then
    break
  fi
  printf '[wait] %(%FT%T%z)T gpu%s memory used: %s MiB; %s MiB\n' \
    -1 "${GPU_IDS}" "${GPU_USED[0]:-unknown}" "${GPU_USED[1]:-unknown}"
  sleep "${POLL_SECONDS}"
done

printf '[start] %(%FT%T%z)T gpu%s are available\n' -1 "${GPU_IDS}"
exec env \
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  GPUS=2 \
  EPOCHS="${EPOCHS:-6}" \
  MODEL_SIZE=tiny \
  RESOLUTION=384 \
  NUM_FRAMES=12 \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}" \
  TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-6}" \
  bash "${ROOT}/solution/train.sh" "${DATA}" "${OUTPUT}"
