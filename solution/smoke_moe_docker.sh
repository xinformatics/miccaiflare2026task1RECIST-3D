#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${IMAGE:-aidaraomayoclinic:latest}
INPUTS=${1:-"${ROOT}/artifacts/docker-moe-smoke-input"}
OUTPUTS=${2:-"${ROOT}/artifacts/docker-moe-smoke-output"}

if [[ ! -d "${INPUTS}" ]] || ! compgen -G "${INPUTS}/*.npz" >/dev/null; then
  echo "smoke input must be a directory containing at least one NPZ: ${INPUTS}" >&2
  exit 2
fi
mkdir -p "${OUTPUTS}"
/usr/bin/time -v docker run --rm -m 8g --cpus 12 \
  -v "${INPUTS}:/workspace/inputs:ro" \
  -v "${OUTPUTS}:/workspace/outputs" \
  "${IMAGE}" bash /workspace/docker_moe/predict.sh
"${ROOT}/.venv/bin/python" "${ROOT}/solution/validate_submission.py" \
  "${INPUTS}" "${OUTPUTS}"
