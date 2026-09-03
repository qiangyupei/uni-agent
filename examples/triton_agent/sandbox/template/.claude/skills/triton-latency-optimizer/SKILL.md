---
name: triton-latency-optimizer
description: >
  Optimize a fully-correct Triton implementation for latency. Use after the
  prepared verifier first reports passed_cases == total_cases.
argument-hint: "op-name"
---

# Triton Latency Optimizer

Correctness comes first:
- Invoke this skill only after `bash tools/verify_once.sh <op_name>` reports
  full correctness.
- Do not benchmark before `passed_cases == total_cases`.
- Do not tune `num_warps`, `num_ctas`, `num_stages`, or related Ascend-rejected
  launch kwargs.

Optimization loop:
- Make one small performance change at a time.
- Prefer fewer kernel launches, fused elementwise work, coalesced accesses,
  less redundant loading, and launch grids that match the output shape.
- Preserve all supported shapes and dtypes.
- Re-run `bash tools/verify_once.sh <op_name>` after every change. A full-pass
  run automatically benchmarks the candidate and updates the best snapshot only
  when speedup is strictly higher.
- Stop immediately when output contains `[best-snapshot] early-stop`.

Do not invoke `benchmark.py` directly or add custom timing code.
