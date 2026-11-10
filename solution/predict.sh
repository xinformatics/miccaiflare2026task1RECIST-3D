#!/usr/bin/env bash
set -euo pipefail

INPUTS=${1:-/workspace/inputs}
OUTPUTS=${2:-/workspace/outputs}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WEIGHTS=${WEIGHTS:-${ROOT}/checkpoints/eff_medsam2_tiny_FLARE25_RECIST_baseline.pt}
IMAGE_SIZE=${IMAGE_SIZE:-512}

mkdir -p "${OUTPUTS}"
exec python3 "${ROOT}/baseline/Efficient_MedSAM2_infer_CT_lesion_npz_recist.py" \
  --weights "${WEIGHTS}" \
  --model-size tiny \
  --device cpu \
  --no-compile \
  --image-size "${IMAGE_SIZE}" \
  --prompt box \
  --z-slab-scale 1.2 \
  --max-planned-steps 120 \
  --fallback-z-slab-scale 1.0 \
  --state-mode shared \
  --empty-stop-patience 3 \
  --imgs-path "${INPUTS}" \
  --pred-save-dir "${OUTPUTS}"
