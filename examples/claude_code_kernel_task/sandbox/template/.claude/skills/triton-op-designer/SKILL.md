---
name: triton-op-designer
description: >
  Optional algorithm sketching skill. Do not use during prepared KernelBench
  rollout unless explicitly requested; write ModelNew directly instead.
argument-hint: "op-name"
---

# Triton Op Designer

Prepared rollout default: do not invoke this skill.

Use it only when explicitly asked to produce an algorithm sketch before coding.
For normal rollout, follow `INSTRUCTIONS.md`, read `src/<op_name>.py`, write
`ModelNew`, and verify with `tools/verify_once.sh`.

If used, keep the sketch short:
- Tensor shapes and output contract.
- One-kernel mapping strategy.
- Key masks and dtype handling.
- No benchmark or latency work.
