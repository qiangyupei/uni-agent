#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE=${BASE_IMAGE:-triton-claude-code-env:latest}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-triton-claude-code-env:kernel-bench}
SANDBOX_USER=${SANDBOX_USER:-claude}
SANDBOX_UID=${SANDBOX_UID:-1000}
SANDBOX_GID=${SANDBOX_GID:-1000}

docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "SANDBOX_USER=${SANDBOX_USER}" \
  --build-arg "SANDBOX_UID=${SANDBOX_UID}" \
  --build-arg "SANDBOX_GID=${SANDBOX_GID}" \
  --tag "${OUTPUT_IMAGE}" \
  "${SCRIPT_DIR}"
