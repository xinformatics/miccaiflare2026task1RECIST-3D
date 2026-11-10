#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${IMAGE:-pancancer-recist3d:latest}
INPUTS=${1:-${ROOT}/baseline/data/validation_public_npz}
OUTPUTS=${2:-${ROOT}/artifacts/docker-smoke}

mkdir -p "${OUTPUTS}"
docker run --rm -m 8g --cpus 12 \
  -v "${INPUTS}:/workspace/inputs:ro" \
  -v "${OUTPUTS}:/workspace/outputs" \
  "${IMAGE}" /bin/bash -c "sh predict.sh"
python "${ROOT}/solution/validate_submission.py" "${INPUTS}" "${OUTPUTS}"
