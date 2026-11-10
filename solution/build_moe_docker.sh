#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${IMAGE:-aidaraomayoclinic:latest}
ARCHIVE=${1:-"${ROOT}/artifacts/aidaraomayoclinic.tar.gz"}

docker build --platform linux/amd64 -t "${IMAGE}" \
  -f "${ROOT}/docker_moe/Dockerfile" "${ROOT}"
mkdir -p "$(dirname "${ARCHIVE}")"
docker save "${IMAGE}" | gzip -1 > "${ARCHIVE}"
echo "wrote ${ARCHIVE}"
