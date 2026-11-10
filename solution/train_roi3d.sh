#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/roi3d-inductor}

exec "${ROOT}/.venv/bin/python" -m roi3d.train \
  --cache-root "${ROOT}/cache_geo_v2" \
  --hidden "${ROOT}/dataset/validation_npz/validation_hidden_npz" \
  --output "${ROOT}/artifacts/roi3d-a2-b20" \
  --base 32 \
  --batch-size 20 \
  --lr 1e-3 \
  --epochs 20 \
  --steps-per-epoch 250 \
  --workers 4 \
  --size-alpha 0.75 \
  --rot-z 0.35 \
  --rot-xy 0.10 \
  --scale-aug 0.15 \
  --shift-aug 6 \
  --band-w 3.0 \
  --band-k 2 \
  --save-every 1 \
  --compile
