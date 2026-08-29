# PR 1: optional trajectory postprocessor

Suggested title: `[framework] feat: add trajectory postprocessor hook`

Target branch: `verl-project/uni-agent:main` (independent of PR 2 and PR 3)

## Summary

Add a narrow, opt-in Framework extension point for recipe-specific trajectory
filtering, cropping, replacement, and ordering. The hook receives finalized
Gateway trajectories after the Runner's built-in `trajectory_selection`, and
runs before reward scoring, artifact logging, unfinished masking, and
TransferQueue materialization.

No tracking issue is required: this is a focused, backward-compatible extension
boundary for policies that need finalized trajectories but do not belong in
Gateway or the generic Task layer.

## Changes

- Add optional `trajectory_postprocessor_fqn` and
  `trajectory_postprocessor_kwargs` Framework configuration.
- Resolve the callable when the Agent Framework starts. Invalid explicit
  configuration, import failures, and non-callable targets fail before rollout
  work begins.
- Call the extension as
  `postprocessor(tuple(trajectories), **trajectory_postprocessor_kwargs)`.
- Support synchronous and asynchronous processors.
- Require `list[Trajectory]` as the result. An empty list filters that session,
  and returned trajectories must preserve the finalized session `reward_info`.
- Document the callable contract, execution order, compatibility, and
  TransferQueue boundary.
- Add focused CPU coverage for sync and async execution, keyword forwarding,
  ordering before scoring and TransferQueue output, empty filtering, strict
  validation, and `reward_info` preservation.

## Configuration and API

```yaml
actor_rollout_ref:
  rollout:
    custom:
      agent_framework:
        trajectory_postprocessor_fqn: my_recipe.trajectory.process_trajectories
        trajectory_postprocessor_kwargs:
          max_total_tokens: 262144  # 256K prompt + response tokens
```

```python
from uni_agent.gateway.session import Trajectory


def process_trajectories(
    trajectories: tuple[Trajectory, ...],
    *,
    max_total_tokens: int = 262_144,
) -> list[Trajectory]:
    return [
        trajectory
        for trajectory in trajectories
        if len(trajectory.prompt_ids) + len(trajectory.response_ids) <= max_total_tokens
    ]
```

`max_total_tokens` belongs to this example processor; it is not a built-in
Framework option. A processor that crops instead of filters must preserve token
array alignment, valid turn boundaries, and the finalized `reward_info`. Use
`dataclasses.replace` when constructing transformed trajectories so unrelated
fields remain intact. Synchronous processors execute in the
`AgentFrameworkWorker` event loop and must not perform blocking I/O.

## Compatibility

Omitting `trajectory_postprocessor_fqn`, or setting it to `null`, preserves the
original path: no extension is imported or called.

A configured FQN must be a non-empty string, and its kwargs must be a mapping.
Non-empty kwargs without an FQN, import failures, and non-callable targets are
Framework initialization errors. Processor execution errors, invalid return
values, and changes to finalized `reward_info` also fail explicitly.

## Validation

Patch and style validation completed against
`28174fdab3787d307ae3a96d32d3737b600575a0`:

- single combined PR patch `git am`: passed, with an applied tree identical to
  the PR branch;
- focused Ruff check and format check: passed;
- focused `compileall`: passed.

Rerun the CPU suite and repository hooks in the PR environment before updating
the public validation result:

```bash
python -m pytest -q tests/uni_agent/framework/test_generate_sequences_on_cpu.py
python -m ruff check uni_agent/framework/framework.py \
  tests/uni_agent/framework/test_generate_sequences_on_cpu.py
python -m ruff format --check uni_agent/framework/framework.py \
  tests/uni_agent/framework/test_generate_sequences_on_cpu.py
pre-commit run --all-files --show-diff-on-failure
```

The latest CPU suite was not rerun in the Windows audit environment because its
Python installation lacks `fastapi` during test collection.

## Checklist

- [x] The PR is focused and explains why no issue is needed.
- [x] The title follows the repository format and names the owning layer.
- [x] Tests cover sync and async execution, keyword forwarding, ordering, empty
      filtering, strict validation, and finalized `reward_info` preservation.
- [x] The configuration and callable contract are documented with an example.
- [x] Existing configurations remain unaffected unless
      `trajectory_postprocessor_fqn` is explicitly configured.
- [x] Logs, fixtures, and examples contain no credentials or private data.
- [ ] The focused CPU test passes in the final PR environment.
- [ ] `pre-commit run --all-files --show-diff-on-failure` passes.
