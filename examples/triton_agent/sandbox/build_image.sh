#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE=${BASE_IMAGE:-triton-claude-code-env:latest}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-triton-claude-code-env:kernel-bench}
SANDBOX_USER=${SANDBOX_USER:-claude}

docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "SANDBOX_USER=${SANDBOX_USER}" \
  --tag "${OUTPUT_IMAGE}" \
  "${SCRIPT_DIR}"
