#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR="examples/blackbox_recipes/triton_agent"
OUTPUT_DIR="${1:-outputs/triton_agent/synthetic}"

python "${RECIPE_DIR}/prepare_data.py" \
  --train-source "${RECIPE_DIR}/fixtures/synthetic/train" \
  --validation-source "${RECIPE_DIR}/fixtures/synthetic/validation" \
  --dataset-name uni-agent-triton-synthetic-smoke \
  --dataset-revision recipe-v1 \
  --npukernelbench-levels= \
  --source-manifest "${RECIPE_DIR}/config/synthetic_source_manifest.json" \
  --output-dir "${OUTPUT_DIR}" \
  --format parquet
