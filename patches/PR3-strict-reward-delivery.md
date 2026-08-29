# PR 3: strict Task reward delivery

Suggested title after PR 2 merges:
`[framework, docs] feat: add strict task reward delivery`

For an early stacked Draft, use
`[2/2][framework, docs] feat: add strict task reward delivery`.

Recommended target branch: `verl-project/uni-agent:main`, after PR 2 merges

## Summary

Add an opt-in `reward_post_strict` policy to the generic Task Runner. Training
jobs that depend on runner-produced reward can fail the session when reward
delivery is unavailable instead of finalizing with missing or stale reward
metadata.

The existing `report_reward` option remains the switch that enables reward
POSTs. `reward_post_strict` only controls delivery-failure handling. No tracking
issue is required because this is a focused, backward-compatible delivery
policy for the existing endpoint.

## Dependency

This commit is stacked on PR 2. For a contribution from a fork, the cleanest
workflow is to wait for PR 2 to merge, rebase this branch onto the updated
upstream `main`, and then open the PR so reviewers see only the strict-delivery
commit.

For early review, open a Draft PR against upstream `main` and state that it
depends on PR 2. The Draft will temporarily contain both commits. After PR 2
merges, rebase onto upstream `main` and force-push with lease; the diff will then
shrink to the strict-delivery commit. The `[2/2]` title prefix can be removed at
that point.

## Changes

- Add opt-in `reward_post_strict=false` to the generic Task Runner.
- In strict mode, propagate a missing session reward endpoint, transport error,
  non-success HTTP response, or request timeout.
- Bound every reward POST attempt to 30 seconds so a stalled endpoint cannot
  keep a Task Runner alive indefinitely.
- Preserve best-effort behavior by default: failures are logged and swallowed.
- Keep metadata validation independent from delivery policy. Invalid or
  oversized `extra_info` is omitted in both modes while canonical fields can
  still be delivered strictly.

## Configuration

```yaml
runner_kwargs:
  report_reward: true
  reward_post_strict: true
```

`reward_post_strict=true` has no effect when `report_reward=false`, because no
delivery is requested.

## Compatibility

- Both options default to `false`; callers that do not report reward are
  unchanged.
- With `report_reward=true` and strict mode disabled, delivery remains
  best-effort.
- Reward POSTs now have a 30-second upper bound instead of an unbounded client
  timeout, including in best-effort mode.
- Strict mode affects endpoint delivery only, not `TaskResult.extra_info`
  validation.

## Security boundary

Strict delivery is fail-closed transport handling. It does not authenticate or
authorize the Gateway reward endpoint and does not replace capability tokens,
network ACLs, or server-side rate limits.

## Validation

Validated on top of PR 2:

- PR 3 forward-check on the PR 2 tree: passed;
- PR 3 reverse-check on the strict branch: passed;
- strict stack focused Task Runner/routing tests: `25 passed`;
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

- [x] The PR is limited to reward-delivery semantics and documentation.
- [x] The title follows the repository format; an early Draft identifies the stacked series.
- [x] Default best-effort and opt-in strict behavior are both tested.
- [x] Tests cover missing endpoint, timeout configuration, transport failure,
  non-success response, and strict-policy forwarding.
- [x] Metadata omission remains independent from strict delivery.
- [x] The transport/security boundary is documented.
- [x] Logs, fixtures, and examples contain no credentials or private data.
- [ ] `pre-commit run --all-files --show-diff-on-failure` passes in the final PR branch.
