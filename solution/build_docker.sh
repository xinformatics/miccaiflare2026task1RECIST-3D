#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${IMAGE:-pancancer-recist3d:latest}
ARCHIVE=${1:-${ROOT}/artifacts/pancancer-recist3d.tar.gz}

docker build --platform linux/amd64 -t "${IMAGE}" -f "${ROOT}/solution/Dockerfile" "${ROOT}"
mkdir -p "$(dirname "${ARCHIVE}")"
docker save "${IMAGE}" | gzip -1 > "${ARCHIVE}"
echo "wrote ${ARCHIVE}"

