# Migration scope and verification ledger

## Pinned baselines

- Uni-Agent: `28174fdab3787d307ae3a96d32d3737b600575a0` (recipe and patch baseline).
- verl submodule: `483b8a009ba3a97563edee3a19887e4862b8094a`, tag `v0.9.0`.
- Legacy source fork: Uni-Agent
  `691f85d9c968c50fed3587467feb87db3e894d16` with verl
  `e128d532ffa3a0b5d4b60e4a478d286ca8eac3e5`.

Re-run the CPU and NPU checks below whenever any baseline, image digest, CANN,
torch-npu, Triton Ascend, vLLM, vLLM-Ascend, or Claude Code version changes.

## Included and excluded behavior

| Area | Decision | New implementation |
| --- | --- | --- |
| NPU operator data/task/reward | Included | `uni_agent.tasks.kernel_bench` preprocessing, Task, legacy agent-time metric selection, and partial-credit reward |
| NPUKernelBench and DrKernel | Included | JSON/JSONL sidecars, official parquet adapter, multi-case generation, known-invalid validation exclusions |
| Trajectory crop/filter/best selection | Included | Pure `process_trajectories(trajectories, **policy)` with explicit policy arguments |
| Per-session sandbox lifecycle | Included | Official `Task.build_sandbox()` async context plus recipe-local remote `DockerSandbox`; a retry creates a fresh container |
| Claude Code | Included through stock API | Official `ClaudeCodeAgent`; native transcript hooks provide only a best-turn hint |
| Observability | Included through stock API | Framework/task logs, trajectory dumps, and bounded `TaskResult.extra_info` provenance |
| Gateway changes and protocol shim | Excluded | Stock Gateway/session API over the directly reachable LAN address |
| KV-cache router | Excluded | No recipe code or patch |
| Legacy Megatron/MindSpeed and custom LM-head work | Excluded | NPU launcher uses official VeOmni; optional GPU launcher uses stock verl 0.9 Megatron only |
| Checkpoint changes | Excluded | Only the official verl v0.9 recipe settings remain |
| Custom debug services/UI | Excluded | Existing Uni-Agent logging and dumps only |
| NPU-memory custom commits | Not migrated by request | See the audit note below |

The unused `task-extractor` skill and NPUKernelBench's roughly 39 MiB
`_all_case.json` files are omitted. The skills used by Claude, numerical
verifier, latency backend, and `verify_once.sh` flow are staged under
`examples/triton_agent/sandbox` for local image construction. Their existing
headers and notices must be retained. This development migration records the
licence boundary but does not treat it as an implementation blocker.

## Core patches

The framework, Gateway, Sandbox, and Agent implementations remain unchanged by
the recipe itself. It adds `uni_agent.tasks.kernel_bench` and its lazy Task
registry entry. The first two patches in the repository-level `patches/`
directory are required for the intended recipe semantics:

1. framework-level `trajectory_postprocessor_fqn` plus directly forwarded
   `trajectory_postprocessor_kwargs`; and
2. bounded JSON-serializable forwarding of `TaskResult.extra_info`, with
   `reward`, `acc`, and `finished` reserved.

Production runs should also stack opt-in fail-closed reward POST delivery and
cancellation-safe bounded Sandbox `stop()`. Without those two patches, normal
execution still works, but reward delivery is best-effort and cancellation
cleanup retains the stock boundary.

These are narrow reusable APIs. They are kept as separate PR candidates. The
KernelBench implementation follows the main repository layout under
`uni_agent/tasks/kernel_bench`; recipe-only configuration and orchestration stay
under `examples/triton_agent`. The current `Task` and `Sandbox` APIs already
provide context-managed per-attempt creation/destruction and provider-specific
kwargs; the lifecycle patch closes
the generic cancellation/timeout/error-propagation gap without adding a
recipe-only pool. Remote-Docker selection and evaluator-device contracts remain
example-local. The provider uses the configured runtime timeout as the
`--rm` container's PID-1 sleep bound, so a hard-killed Ray worker does not leave
an unbounded container.

## Deliberate stock-Claude differences

The old runner tee'd Claude's stream protocol, polled verifier state, and could
hard-kill the agent after a verifier-patience threshold. Stock
`ClaudeCodeAgent` does not expose that stream or process handle. This migration
uses native Claude `PreToolUse`, `PostToolUse`, and `PostToolUseFailure` hooks
instead. They bind a changed best implementation to its assistant turn and
cooperatively stop Claude after 7 stale correctness verifies (best reward at
least 0.15) or 3 stale latency verifies after full correctness. The thresholds
are Task configuration and zero disables either phase.

After Claude exits, the Task reads the agent-time best pair, falling back first
to the last verifier artifacts and then to `metrics.json`; it recomputes reward
but does not rerun verification. The launcher selects `best`, so an unambiguous
single trajectory is cropped to the bound assistant turn. Multiple Gateway
chains retain legal final prefixes because the scalar hook index has no chain
identity. There is no runner-side verifier polling, protocol shim, OS hard kill,
or repair-round loop. The retained fresh-sandbox no-implementation retry is
also disabled by default pending a real attempt/chain mapping test.

The previous training scripts used a 9000-second Claude time budget (10800 for
their separate validation entrypoint). The recipe Task config now uses
`agent.run_timeout=9000`; this is the timeout of the single stock Claude CLI
command, not the sandbox lifetime. Correctness and benchmarking occur only when
Claude invokes the agent-time verifier; final artifact selection is bounded
file I/O rather than another NPU evaluation.

## NPU memory audit

`git merge-base --is-ancestor <commit> v0.9.0` confirms that verl v0.9.0
already contains all of the audited upstream correctness fixes below; they need
no recipe backport:

| Commit | Behavior already present in v0.9.0 |
| --- | --- |
| `64ea4542` | BF16 precision-aware optimizer state and DDP grad-bucket alignment |
| `587b1204` | precision-aware optimizer offload fix |
| `0a1c5b64` | ref-model offload storage leak fix |
| `03bc8411` | avoid 2x peak host memory during Megatron model offload |
| `e52747a4` | release Transformer Engine FP8 workspaces during CPU offload |

The following legacy-fork commits are absent from both v0.9.0 and the reviewed
newer verl gitlink, but are custom Megatron/MindSpeed memory/performance
optimizations rather than confirmed missing upstream correctness fixes:

- CP/rank parameter-offload de-duplication: `47e7de63`, `48dde05c`,
  `6afc1006`, and `763a0350`;
- HDO weight/state de-duplication and compression: `e377ca2a`;
- response-only LM-head computation: `3d1d6319`;
- longest-first micro-batch ordering: `e128d532`; and
- the legacy `moe_zero_memory` series: `773423c9`, `93496da9`, `d33a966a`,
  `62279f9a`, and `5617876d`.

They target the explicitly excluded custom Megatron/MindSpeed path. The NPU
launcher follows official VeOmni, and the GPU launcher uses stock verl 0.9
Megatron without these patches. Per the migration decision, none is backported.
Consequently, this example cannot claim the old single-node
35B/CP8/roughly-160K resource envelope without separate qualification.

The external vLLM-Ascend stack still needs an explicit version gate: 0.18 has
known residual memory behavior, while the relevant follow-up fixes
`92eeab2f`/`3a025575` are available in the 0.19.1rc1-or-newer line. verl v0.9.0
does not by itself pin a complete validated CANN/torch-npu/Triton
Ascend/vLLM-Ascend deployment for this workload. Qualify that exact stack and
peak memory before scaling; do not infer the old 35B/160K envelope from CPU
tests or from the official script skeleton.

## External blockers and trust boundary

- Benchmark archives are not vendored. Their immutable checksum, source URL,
  revision, and licence approval are required before publishing generated data.
- The staged legacy CANN verifier declares CANN Open Software License 2.0
  obligations, including repository-level licence/notice handling, while
  Uni-Agent's root is Apache-2.0. NPUKernelBench and DrKernel redistribution
  terms also need owner confirmation. Do not publish the staged sandbox assets
  until maintainers/legal approve the resulting LICENSE/NOTICE layout.
- The sandbox image and evaluator scripts are deployment artifacts. Publish a
  digest and their licences separately. The local `latest` default is for
  bring-up only and should be replaced with that immutable reviewed digest.
- By explicit migration decision, after Claude exits the Task does not
  independently rerun correctness or latency. It first reads a substantive best
  implementation paired with `metrics_best.json`; if absent it recovers a pair from the last
  `output/verify` JSON and staged implementation; it then falls back to the
  substantive current implementation plus `metrics.json`.
- Those JSON files and implementations are agent-writable. Bounded regular-file
  reads, non-finite-value rejection, implementation-shape checks, digest
  comparison, and timestamps reduce accidental mismatch and filesystem abuse,
  but cannot authenticate numerical results because the agent can rewrite both
  members of a pair. This deliberately restores the legacy reward trust model;
  it is not a runner-owned trust boundary.
- Workspace-provided reward fields are not reused. The recipe recalculates
  partial-credit reward from the selected correctness/performance fields using
  the current weights, including the requested removal of the all-correct
  bonus.
- The image must install root-owned verifier/cleanup/NPU-lease assets and mount
  a host-local lock namespace genuinely shared by every sandbox on one physical
  NPU host. Different hosts must not share locks even when their local device
  IDs are equal. `--check` proves file shape/ownership, not cross-container
  sharedness.
- Agent-time `tools/verify_once.sh` must enter
  `/opt/triton-agent-tools/with_npu_lease.py`; there is no post-agent NPU
  invocation. Direct, unleased accelerator execution is outside the supported
  image contract.
- The Gateway base URL must be reachable from both the Ray runner and each
  remote Docker host. The default deployment uses direct LAN routing and no
  reverse tunnel; Docker endpoint authentication and Gateway routing must be
  integration-tested in the deployment network.
- The launcher supplies the checkout with `ray job submit --working-dir`, so no
  runtime-env file is required by default. Optional runner-side overrides such
  as `SANDBOX_STOP_TIMEOUT` belong in Ray `runtime_env.env_vars`; entrypoint-only
  shell variables are not a worker propagation mechanism.
- The stock `/sessions/{id}/reward_info` route has no runner-only capability.
  Strict final POST makes delivery failure fatal but cannot authenticate that
  route. Training is blocked until the sandbox can reach only its
  `/v1/messages` path (with direct Gateway egress denied), or an independently
  reviewed Gateway capability mechanism is available. No Gateway patch is
  included here, per the agreed migration scope.
- The recipe-local provider reuses stock `DockerSandbox` through
  `docker --host`, and maps `runtime_timeout` to a finite PID-1 `sleep`; normal
  Task exit still removes the container immediately with `docker rm -f`.
- Each session is assigned deterministically across the configured remote
  Docker endpoints. Every daemon resolves its own device and lock-directory
  bind mounts. A global NFS lock would incorrectly serialize equal device IDs
  on different hosts.

## Verification matrix

| Check | CPU/CI | Real NPU/deployment |
| --- | --- | --- |
| Python compile, Ruff, Bash syntax | Required, automated | Same image revision |
| Stable UID, leakage, JSONL, DrKernel duplicates/levels/invalid refs | Required, automated | Re-run on full pinned source |
| Legacy best/artifact/current metric selection and reward normalization | Required, automated | Compare with real agent-time verifier outputs |
| Sandbox destroy/retry and pre-read process cleanup | Fake provider test | Kill/cancel/TTL fault injection |
| NPU lease parsing and runner env forwarding | Required, automated | Concurrent verifier test on every physical device/host |
| Trajectory alignment/crop/best/no-impl/empty policy | Required, automated | Real multi-chain Gateway rollout |
| Hydra configuration and NPU/GPU topology | Syntax/compose smoke | One optimizer step per backend, then scale/memory qualification |
| Gateway routing and reward posting | Direct-route config + strict-failure unit test | Real sandbox-network ACL/capability |

An all-empty postprocessed GRPO prompt is a hard pre-training validation failure,
not a sample to pad with an invalid trajectory. Validate surviving group sizes
on a real rollout before starting a long run.

## Local verification record (updated 2026-08-28)

- Current Windows recipe suite excluding `test_runner.py` (the review Python
  lacks the complete Uni-Agent web/runtime dependencies): `56 passed, 1
  skipped`; the skip is the Linux/root
  process-group lease integration. This includes the resolved Docker run-args
  and host-lock mount regression.
- Earlier combined-patch example CPU suite: `53 passed, 1 skipped`; the skip is
  the Linux/root process-group lease integration on the Windows review host.
- After splitting metadata from strict delivery, the Task Runner/routing suite
  passed `21` tests at the metadata commit and `25` at the stacked strict head.
- Earlier combined patched framework suite: `70 passed`; Sandbox lifecycle +
  Docker focus: `18 passed`.
- Ruff check/format, `compileall`, Bash syntax, patch whitespace/apply checks,
  and `git diff --check`: passed.
- verl v0.9 Hydra composition accepted both launchers' overrides, including
  trajectory postprocessing, strict reward posting, async vLLM, VeOmni NPU, and
  stock Megatron GPU configuration. Real one-step NPU and NVIDIA runs remain
  required in their target images.
- Full local legacy inputs were read without vendoring them: NPUKernelBench
  `101` total / `61` selected level-1+2 rows; DrKernel `1999` train / `196`
  validation rows; all emitted UIDs were unique.
- Synthetic JSONL and Parquet generation each produced one disjoint train and
  validation row with verified source, manifest, output, and UID digests.

These are CPU/config/data checks. No result above substitutes for the real NPU,
Gateway ACL/capability, provider TTL, shared-device lock, verifier-image, GRPO
cardinality, one-step VeOmni, or peak-memory qualification gates in the table.
