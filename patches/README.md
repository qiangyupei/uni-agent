# Uni-Agent prerequisite patches

These patches stay outside the recipe branch so each change can be reviewed as
a focused Uni-Agent pull request.

PR 1, PR 2, and PR 4 use Uni-Agent commit
`28174fdab3787d307ae3a96d32d3737b600575a0` as their reviewed baseline. PR 3 is
stacked on PR 2. Each mail patch records its exact commit in the opening `From`
line.

## Recipe dependencies

The recipe's configured feature set requires:

- `PR1-trajectory-postprocessor.patch`: adds the opt-in
  `trajectory_postprocessor_fqn` hook used for best-prefix selection and
  trajectory filtering. Omitting the FQN leaves the Framework path unchanged.
- `PR2-task-result-extra-info.patch`: forwards bounded, JSON-serializable
  `TaskResult.extra_info` alongside canonical reward fields.

The following patches are optional production hardening:

- `PR3-strict-reward-delivery.patch`: lets a Task Runner fail the session when
  an enabled reward POST cannot be delivered. It requires PR 2 and does not add
  Gateway authentication or authorization.
- `PR4-sandbox-lifecycle-cleanup.patch`: shields and bounds lifecycle-owned
  cleanup, prevents retry after failed partial-start cleanup, and keeps a failed
  OpenYuanRong kill handle retryable. It protects cleanup while the Python
  worker is still running; a process-level hard kill still requires a
  provider-side TTL or reaper.

Ready-to-paste PR descriptions are in the matching `.md` files.

## Applying the patches

Apply patches from the Uni-Agent repository root. For the recipe's required
features:

```bash
git am --keep-non-patch patches/PR1-trajectory-postprocessor.patch
git am --keep-non-patch patches/PR2-task-result-extra-info.patch
```

Apply PR 3 after PR 2 if strict reward delivery is wanted. PR 4 is independent:

```bash
git am --keep-non-patch patches/PR3-strict-reward-delivery.patch
git am --keep-non-patch patches/PR4-sandbox-lifecycle-cleanup.patch
```

PR 1 is a single combined mail patch representing its complete PR diff. The
`.../` in a patch diffstat is Git's abbreviation for a shared path prefix, not
a real directory; full paths appear in the `diff --git` headers.

## Validation

Run the focused checks for each applied patch:

```bash
python -m pytest -q tests/uni_agent/framework/test_generate_sequences_on_cpu.py
python -m pytest -q tests/uni_agent/framework/test_task_runner.py \
  tests/uni_agent/tasks/test_inference_task_routing.py
python -m pytest -q tests/uni_agent/sandbox/test_lifecycle_cleanup.py \
  tests/uni_agent/sandbox/test_docker_sandbox.py
python -m ruff check uni_agent tests/uni_agent
python -m ruff format --check uni_agent tests/uni_agent
```

Recorded focused results are:

- PR 1: `38 passed, 4 warnings`;
- PR 2: `21 passed, 3 warnings`;
- PR 3 stack: `25 passed`;
- PR 4 lifecycle and Docker suites: `13 passed`;
- PR 4 sandbox suite excluding unavailable seed-provider cases: `43 passed,
  6 deselected`.

The six seed-provider cases are unrelated to PR 4 and reference a module and
registry entry absent from its baseline. Focused Ruff check and format check
pass for PR 4. All four mail patches also apply cleanly in the displayed order
against the shared baseline.
