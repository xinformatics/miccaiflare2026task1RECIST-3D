#!/bin/sh
set -eu

INPUTS=${1:-/workspace/inputs}
OUTPUTS=${2:-/workspace/outputs}
ROOT=${ROOT:-/workspace}
DEVICE=${DEVICE:-cpu}
PYTHON=${PYTHON:-python3}
MODELS=${MODELS:-${ROOT}/models}
LEGACY_SCRIPT=${LEGACY_SCRIPT:-${ROOT}/baseline/legacy_effmedsam2_infer.py}
if [ ! -d "${MODELS}" ]; then MODELS="${ROOT}/docker_moe/models"; fi
WORK=$(mktemp -d /tmp/pancancer-moe.XXXXXX)
if [ "${KEEP_WORK:-0}" = 1 ]; then
  echo "keeping intermediate predictions in ${WORK}"
else
  trap 'rm -rf "${WORK}"' EXIT
fi
mkdir -p "${OUTPUTS}"

CHEAP_COUNT=$("${PYTHON}" "${ROOT}/docker_moe/route_cpu_cases.py" \
  --input "${INPUTS}" --output "${WORK}/cheap_inputs" \
  --slab-scale 1.6 --max-total-slices 40 | tail -n 1)
MEDIUM_COUNT=$("${PYTHON}" "${ROOT}/docker_moe/route_cpu_cases.py" \
  --input "${INPUTS}" --output "${WORK}/medium_inputs" \
  --slab-scale 1.6 --min-total-slices 40 --max-total-slices 120 | tail -n 1)
FALLBACK_COUNT=$("${PYTHON}" "${ROOT}/docker_moe/route_cpu_cases.py" \
  --input "${INPUTS}" --output "${WORK}/fallback_inputs" \
  --slab-scale 1.6 --min-total-slices 120 --max-total-slices 1000000 | tail -n 1)

if [ "${CHEAP_COUNT}" -gt 0 ]; then
  "${PYTHON}" "${ROOT}/docker_moe/run_legacy_sam.py" \
    --script "${LEGACY_SCRIPT}" \
    --weights "${MODELS}/effmedsam2_small.pt" --size small --device "${DEVICE}" \
    --prompt box --box-scale 0.9 --slab-scale 1.6 \
    --input "${WORK}/cheap_inputs" --output "${WORK}/small_box09"

  "${PYTHON}" "${ROOT}/docker_moe/run_legacy_sam.py" \
    --script "${LEGACY_SCRIPT}" \
    --weights "${MODELS}/effmedsam2_small.pt" --size small --device "${DEVICE}" \
    --prompt 3points --box-scale 0.9 --slab-scale 1.4 --bounded-3points \
    --input "${WORK}/cheap_inputs" --output "${WORK}/small_3points"
fi

if [ $((CHEAP_COUNT + MEDIUM_COUNT)) -gt 0 ]; then
  mkdir -p "${WORK}/tiny_inputs"
  cp -a "${WORK}/cheap_inputs/." "${WORK}/tiny_inputs/"
  cp -a "${WORK}/medium_inputs/." "${WORK}/tiny_inputs/"
  "${PYTHON}" "${ROOT}/docker_moe/run_legacy_sam.py" \
    --script "${LEGACY_SCRIPT}" \
    --weights "${MODELS}/effmedsam2_tiny.pt" --size tiny --device "${DEVICE}" \
    --prompt box --box-scale 0.9 --slab-scale 1.6 \
    --input "${WORK}/tiny_inputs" --output "${WORK}/tiny_box09"
fi

if [ "${CHEAP_COUNT}" -gt 0 ]; then
  "${PYTHON}" -m roi3d.predict \
    -i "${WORK}/cheap_inputs" -o "${WORK}/cheap_outputs" \
    --checkpoint "${MODELS}/organ_moe_epoch65.pt" \
    --member-dirs "${WORK}/small_box09" "${WORK}/tiny_box09" "${WORK}/small_3points" \
    --ensemble-weights 0.46,0.18,0.18,0.18 \
    --domain-routing filename --batch-size 4 --device "${DEVICE}"
  cp "${WORK}/cheap_outputs/"*.npz "${OUTPUTS}/"
fi

if [ "${MEDIUM_COUNT}" -gt 0 ]; then
  "${PYTHON}" -m roi3d.predict \
    -i "${WORK}/medium_inputs" -o "${WORK}/medium_outputs" \
    --checkpoint "${MODELS}/organ_moe_epoch65.pt" \
    --baseline-dir "${WORK}/tiny_box09" --weight 0.57 \
    --domain-routing filename --batch-size 4 --device "${DEVICE}"
  cp "${WORK}/medium_outputs/"*.npz "${OUTPUTS}/"
fi

if [ "${FALLBACK_COUNT}" -gt 0 ]; then
  "${PYTHON}" -m roi3d.predict \
    -i "${WORK}/fallback_inputs" -o "${WORK}/fallback_outputs" \
    --checkpoint "${MODELS}/organ_moe_epoch65.pt" \
    --domain-routing filename --batch-size 4 --device "${DEVICE}"
  cp "${WORK}/fallback_outputs/"*.npz "${OUTPUTS}/"
fi

rm -f "${OUTPUTS}/inference_times.json"
