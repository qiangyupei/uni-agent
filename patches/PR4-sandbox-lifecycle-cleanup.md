# PR 4: bounded sandbox lifecycle cleanup

Suggested title: `[sandbox, docs] fix: bound lifecycle cleanup`

Recommended target branch: `verl-project/uni-agent:main`

## Summary

Make lifecycle-owned sandbox cleanup survive caller cancellation and finish
within a configured bound. This keeps per-session sandboxes recoverable when an
agent is cancelled or force-stopped while its Python worker remains alive.

No tracking issue is required because this is a focused fix to the existing
`Sandbox` context-manager contract.

## Changes

- Shield lifecycle-owned `stop()` calls from caller cancellation.
- Bound cleanup with `SANDBOX_STOP_TIMEOUT`, defaulting to 60 seconds.
- Clean up a partial start before retrying, and stop retrying if cleanup fails.
- Let OpenYuanRong kill failures propagate while retaining the provider handle
  for a later retry.
- Document the provider TTL/reaper boundary for process-level hard kills.

The implementation continues to use the existing `Sandbox.start()`, `stop()`,
context manager, and `entered()` interfaces. Providers do not need a new API.

## Configuration

```bash
export SANDBOX_STOP_TIMEOUT=60
```

The value must be a positive finite number of seconds. Invalid values fall back
to 60 seconds.

## Compatibility and boundary

- Normal context-manager behavior is unchanged: startup happens on entry and
  cleanup happens on exit.
- Cleanup errors now propagate instead of being hidden during failed startup.
- OpenYuanRong `stop()` now propagates kill errors so lifecycle code can report
  them and callers can retry with the retained handle.
- Cancellation and agent termination can be handled only while the Python
  worker is running. `SIGKILL`, worker-node loss, or equivalent process death
  cannot execute Python `finally` blocks and still requires provider-side
  expiration or reaping.

## Validation

Validated on Windows with Python 3.13.6 against
`28174fdab3787d307ae3a96d32d3737b600575a0`:

```bash
python -m pytest -q tests/uni_agent/sandbox/test_lifecycle_cleanup.py \
  tests/uni_agent/sandbox/test_docker_sandbox.py
# 13 passed in 0.26s

python -m pytest -q tests/uni_agent/sandbox -k "not seed"
# 43 passed, 6 deselected in 0.23s

python -m ruff check uni_agent/sandbox/base.py \
  uni_agent/sandbox/openyuanrong.py \
  tests/uni_agent/sandbox/test_lifecycle_cleanup.py

python -m ruff format --check uni_agent/sandbox/base.py \
  uni_agent/sandbox/openyuanrong.py \
  tests/uni_agent/sandbox/test_lifecycle_cleanup.py
```

The independent `git am`, focused tests, Ruff check, and format check passed;
the applied tree matches the PR branch. The six deselected seed-provider cases
reference a module and registry entry absent from this baseline and are
unrelated to lifecycle cleanup.

## Checklist

- [x] The PR is limited to lifecycle cleanup and its documented behavior.
- [x] Existing provider interfaces are reused; no new provider API is added.
- [x] Tests cover cancellation, cleanup timeout, partial-start failure, retry
  suppression, and retryable OpenYuanRong termination.
- [x] The process-level hard-kill boundary is documented.
- [x] Logs, fixtures, and examples contain no credentials or private data.
- [x] Focused tests and Ruff checks pass.
- [ ] Repository-wide hooks pass in the final PR environment.
