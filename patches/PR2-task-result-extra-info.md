# PR 2: bounded TaskResult metadata forwarding

Suggested title: `[framework, docs] feat: forward bounded TaskResult metadata`

If PR 3 is opened early as a stacked Draft, use
`[1/2][framework, docs] feat: forward bounded TaskResult metadata` instead.

Target branch: `verl-project/uni-agent:main` (independent of PR 1; base of PR 3)

## Summary

Forward compact, use-case-specific `TaskResult.extra_info` fields through the
generic Task Runner into a Gateway session's `reward_info`. This lets verifier
metrics reach finalized trajectory metadata without adding task knowledge to
Gateway or the Agent Framework.

`TaskResult.extra_info` already exists. This PR only defines how the generic
Runner forwards it when the existing `report_reward=true` option is enabled.
No tracking issue is required because the change is a focused extension of an
existing Task Runner contract.

## Changes

- Merge JSON-serializable `extra_info` into session `reward_info`.
- Keep `reward`, `acc`, and `finished` reserved for canonical `TaskResult`
  fields; ignore colliding metadata keys.
- Omit all non-reserved metadata, with a warning, when it is not JSON
  serializable or exceeds 64 KiB after UTF-8 serialization. Canonical fields
  remain in the reward payload.
- Document the contract in Task and Reward, with a short link from the Gateway
  documentation.
- Add CPU tests for forwarding, reserved keys, invalid JSON, recursion, and the
  serialized-size boundary.

## API

```python
return TaskResult(
    reward=score,
    accuracy=pass_rate,
    finished=agent_result.finished,
    extra_info={
        "compile_ok": compile_ok,
        "passed_cases": passed_cases,
        "latency_ms": latency_ms,
    },
)
```

The 64 KiB limit is a Task Runner policy, not a Gateway, HTTP, Ray, or trajectory
schema requirement. Session metadata is copied into finalized trajectories and
training records, so large logs, source code, and evaluator artifacts should use
artifact storage instead.

## Compatibility

- `report_reward=false` preserves the original Runner behavior and does not
  forward `extra_info`.
- With `report_reward=true` and no `extra_info`, the canonical payload is
  unchanged.
- Invalid, colliding, or oversized metadata cannot replace or remove canonical
  reward fields.
- This PR does not change reward POST timeout or delivery-failure handling.

## Validation

Validated independently against
`28174fdab3787d307ae3a96d32d3737b600575a0`:

- independent `git apply --check`: passed;
- metadata branch focused Task Runner/routing tests: `21 passed`;
- focused Ruff check and format check: passed;
- combined three-PR `git am`: passed.

Commands:

```bash
python -m pytest -q tests/uni_agent/framework/test_task_runner.py \
  tests/uni_agent/tasks/test_inference_task_routing.py
python -m ruff check uni_agent/framework/task_runner.py \
  tests/uni_agent/framework/test_task_runner.py
python -m ruff format --check uni_agent/framework/task_runner.py \
  tests/uni_agent/framework/test_task_runner.py
pre-commit run --all-files --show-diff-on-failure
```

## Checklist

- [x] The PR is focused and explains why no issue is needed.
- [x] The title follows the repository format and names the owning layers.
- [x] CPU tests cover forwarding, invalid values, and the size boundary.
- [x] API behavior, compatibility, and size-limit rationale are documented.
- [x] Logs, fixtures, and examples contain no credentials or private data.
- [ ] `pre-commit run --all-files --show-diff-on-failure` passes in the final PR branch.
