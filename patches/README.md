# Uni-Agent prerequisite patches

These three patches are deliberately kept out of the recipe branch so each can
be reviewed and submitted to Uni-Agent as a small, independent pull request.
All are based on Uni-Agent commit
`26a49e2646dfe2cb1caa668df2b112ed0afc3ad1`.

The corresponding standalone review commits are
`8a86d845dbf8aa465873f95de7f375c08a18b534` (`pr/trajectory-postprocessor`)
and `2b746fe91a6678dfee01a0778b0125a38ae5aa3e`
(`pr/task-extra-info`), plus
`45955dc982ea4577528d77d1f55cc3feb73a9b31`
(`pr/sandbox-cleanup`). None is merged into the recipe branch.

The Triton example needs all three patches:

- `0001-framework-feat-add-trajectory-postprocessor-hook.patch` adds the
  framework-level `trajectory_postprocessor_fqn` hook and a narrow immutable
  context. The callable runs after built-in trajectory selection and before
  reward scoring, logging, and TransferQueue delivery.
- `0001-framework-tasks-docs-feat-forward-task-result-metada.patch` adds bounded
  `TaskResult.extra_info` forwarding. Metadata must be strict JSON, may contain
  at most eight nested containers, and may serialize to at most 64 KiB;
  `reward`, `acc`, and `finished` remain framework-owned. It also adds the
  opt-in `reward_post_strict` delivery policy used by this recipe; this is
  fail-closed transport handling, not Gateway endpoint authentication.
- `0001-sandbox-docs-fix-bound-lifecycle-cleanup.patch` shields and bounds
  lifecycle-owned `stop()`, preserves cleanup exception chains, and keeps a
  failed OpenYuanRong kill handle retryable. It does not replace provider-side
  `idle_timeout` or an external TTL/reaper for hard-killed Ray workers.

Apply them in either order from the repository root:

```bash
git am --keep-non-patch patches/0001-framework-feat-add-trajectory-postprocessor-hook.patch
git am --keep-non-patch patches/0001-framework-tasks-docs-feat-forward-task-result-metada.patch
git am --keep-non-patch patches/0001-sandbox-docs-fix-bound-lifecycle-cleanup.patch
```

For a non-committing inspection, use `git apply --check <patch>`. The patches
contain their own documentation and CPU tests. Before opening each PR, run:

```bash
python -m pytest -q tests/uni_agent/framework/test_generate_sequences_on_cpu.py
python -m pytest -q tests/uni_agent/framework/test_task_runner.py
python -m pytest -q tests/uni_agent/sandbox/test_lifecycle_cleanup.py
python -m ruff check uni_agent tests/uni_agent/framework tests/uni_agent/sandbox/test_lifecycle_cleanup.py
python -m ruff format --check uni_agent tests/uni_agent/framework tests/uni_agent/sandbox/test_lifecycle_cleanup.py
```

At the pinned baseline, six unrelated seed-provider sandbox tests reference a
module/registry that is absent from that same tree. The cleanup PR validation
therefore records the focused lifecycle and Docker suites separately instead
of treating those pre-existing collection/expectation failures as regressions.
