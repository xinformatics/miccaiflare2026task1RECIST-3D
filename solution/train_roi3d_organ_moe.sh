#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/roi3d-organ-moe-inductor-b14}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

OUTPUT=${OUTPUT:-"${ROOT}/artifacts/roi3d-organ-moe-e195"}
INIT=${INIT:-"${ROOT}/artifacts/roi3d-a2-long300/epoch_195.pt"}
EPOCHS=${EPOCHS:-80}

exec "${ROOT}/.venv/bin/python" -m roi3d.train \
  --cache-root "${ROOT}/cache_geo_v2" \
  --hidden "${ROOT}/dataset/validation_npz/validation_hidden_npz" \
  --output "${OUTPUT}" \
  --init "${INIT}" \
  --base 32 \
  --batch-size 14 \
  --lr 1e-4 \
  --shared-lr 1e-5 \
  --epochs "${EPOCHS}" \
  --steps-per-epoch 321 \
  --workers 4 \
  --size-alpha 0.75 \
  --rot-z 0.35 \
  --rot-xy 0.10 \
  --scale-aug 0.15 \
  --shift-aug 6 \
  --band-w 3.0 \
  --band-k 2 \
  --organ-experts \
  --expert-rank 8 \
  --router-loss-w 0.10 \
  --boundary-loss-w 0.15 \
  --freeze-shared-epochs 10 \
  --save-every 5 \
  --compile
