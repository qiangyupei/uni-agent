---
name: triton-op-coding
description: >
  Prepared KernelBench rollout coding path. Write ModelNew directly from
  INSTRUCTIONS.md and src/<op_name>.py; do not load broad references or invoke
  designer/extractor first.
argument-hint: "op-name"
---

# Triton Op Coding

Use this skill only to implement or repair the prepared operator.

Fast path:
- Read only `INSTRUCTIONS.md` and `src/<op_name>.py` before the first write.
- Do not read the target stub before the first `Write`.
- Write a complete `ModelNew` using at least one `@triton.jit` kernel.
- Run `bash tools/verify_once.sh <op_name>` after each meaningful change.
- Repair from `output/verify/verify_result_summary.json`.
- After `passed_cases == total_cases`, switch to the
  `triton-latency-optimizer` Skill and continue one performance change per
  verifier run until `[best-snapshot] early-stop`.

Keep forward simple:
- Allocate output tensors.
- Compute shapes, strides, and launch grid.
- Launch as many Triton kernels as the implementation requires; fuse stages
  only when it improves correctness or performance.
- Do not perform host-side tensor math in `forward()`.

Avoid known bad patterns:
- No `.data_ptr()` kernel arguments.
- No `tl.to(...)`, `tl.constant(...)`, or `tl.constexpr(...)` calls.
- Use `ARG: tl.constexpr` only as a kernel parameter annotation.
- No `@triton.autotune` or `@triton.heuristics`.
- No chained comparisons inside `@triton.jit`; split masks into named steps.
- No Python `assert` or `raise NotImplementedError` for verifier cases.
- No markdown summaries or status files.

Repair policy:
- Preserve partial best implementations.
- If compile passes and some cases pass, make small targeted edits.
- If the same failure repeats twice, simplify the kernel instead of adding
  another special case.
