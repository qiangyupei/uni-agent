---
name: triton-op-verifier
description: >
  Prepared KernelBench rollout verifier. Use only the workspace validation
  wrapper from INSTRUCTIONS.md. It benchmarks automatically after a full pass.
argument-hint: "op-name"
---

# Triton Op Verifier

Use this skill only as the prepared rollout verification path.

Required flow:
- Run `bash tools/verify_once.sh <op_name>` in the foreground.
- Read `output/verify/verify_result_summary.json` after failure.
- Treat AST success as only a precheck.
- Treat correctness as success only when `passed_cases == total_cases`.
- After full correctness, read the speedup printed by the wrapper and continue
  latency optimization until the verifier hook stops the run.

Do not:
- Read verifier source, full raw logs, or full skill/reference docs by default.
- Create a separate verification project.
- Run custom `python3 -c` torch/triton probes.
- Run benchmark commands or ad hoc latency probes; the wrapper handles the
  benchmark for each fully-correct implementation.
- Edit or change permissions on verifier scripts or validation wrappers.
- Poll background tasks, use `sleep`, `ps`, `pgrep`, `tail -f`, `nohup`, or `&`.

Repair policy:
- Use the compact verifier summary as the only failure signal.
- If an implementation has partial pass, preserve it and make the smallest edit.
- If the same compile/AST issue repeats twice, simplify the kernel instead of
  adding more special cases.
- Do not write markdown summaries or status files.
