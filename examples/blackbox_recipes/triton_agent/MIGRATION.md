# Migration scope and verification ledger

## Pinned baselines

- Uni-Agent: `26a49e2646dfe2cb1caa668df2b112ed0afc3ad1` (latest reviewed main on 2026-08-20).
- verl submodule: `483b8a009ba3a97563edee3a19887e4862b8094a`, tag `v0.9.0`.
- Legacy source fork: Uni-Agent
  `691f85d9c968c50fed3587467feb87db3e894d16` with verl
  `e128d532ffa3a0b5d4b60e4a478d286ca8eac3e5`.

Re-run the CPU and NPU checks below whenever any baseline, image digest, CANN,
torch-npu, Triton Ascend, vLLM, vLLM-Ascend, or Claude Code version changes.

## Included and excluded behavior

| Area | Decision | New implementation |
| --- | --- | --- |
| NPU operator data/task/reward | Included | Example-local deterministic preparer, Task, trusted verifier contract, partial-credit reward |
| NPUKernelBench and DrKernel | Included | JSON/JSONL sidecars, official parquet adapter, multi-case generation, known-invalid validation exclusions |
| Trajectory crop/filter/best selection | Included | Pure `process_trajectories(trajectories, *, context)` using only `context.options` |
| Per-session sandbox lifecycle | Included | Official `Task.build_sandbox()` async context; a retry creates a fresh sandbox |
| Claude Code | Included through stock API | Official `ClaudeCodeAgent`; native transcript hooks provide only a best-turn hint |
| Observability | Included through stock API | Framework/task logs, trajectory dumps, and bounded `TaskResult.extra_info` provenance |
| Gateway changes and protocol shim | Excluded | Stock Gateway/session API; only provider-local URL/tunnel binding remains in the example |
| KV-cache router | Excluded | No recipe code or patch |
| Megatron/MindSpeed and custom LM-head work | Excluded | Official VeOmni NPU recipe is the training base |
| Checkpoint changes | Excluded | Only the official verl v0.9 recipe settings remain |
| Custom debug services/UI | Excluded | Existing Uni-Agent logging and dumps only |
| NPU-memory custom commits | Not migrated by request | See the audit note below |

The legacy workspace's `task-extractor`, `designer`, `npu-arch`, and
`op-coding` CANNBot skills were not invoked by the retained execution path and
are not copied. NPUKernelBench's roughly 39 MiB `_all_case.json` was likewise
unused by its loader and is intentionally omitted. The verifier and latency
tools that are actually executed remain pinned sandbox-image assets until their
CANN and benchmark redistribution obligations are resolved.

## Required core patches

The core tree is intentionally unchanged. Apply the three independent patches in
the repository-level `patches/` directory before training:

1. framework-level `trajectory_postprocessor_fqn` plus deeply read-only
   `trajectory_postprocessor_kwargs`; and
2. bounded JSON-serializable forwarding of `TaskResult.extra_info`, with
   `reward`, `acc`, and `finished` reserved, plus opt-in fail-closed POST; and
3. cancellation-safe bounded Sandbox `stop()` plus retryable OpenYuanRong kill
   failure.

These are narrow reusable APIs. They are kept as separate PR candidates. The
current `Task` and `Sandbox` APIs already provide context-managed per-attempt
creation/destruction and provider-specific kwargs; the third patch closes the
generic cancellation/timeout/error-propagation gap without adding a recipe-only
pool. OpenYuanRong tunnel and evaluator-device contracts are deployment-specific,
so they remain example-local. A hard-killed Ray worker still requires provider
`idle_timeout` and an external TTL/reaper.

## Deliberate stock-Claude differences

The old runner tee'd Claude's stream protocol, polled verifier state, and could
hard-stop the agent after a verifier-patience threshold. Stock
`ClaudeCodeAgent` does not expose that stream or a per-turn cancellation hook.
This migration therefore uses configured `max_turns`/`run_timeout` and native
Claude `PreToolUse`/`PostToolUse` transcript hooks. The hooks annotate a best
snapshot only after `tools/verify_once.sh` changes both the best implementation
and metrics; numerical reward is always recomputed by the final trusted
verifier. Because the stock hook exposes no Gateway chain identity, the scalar
best-turn hint is both agent-controlled and missing a chain identity, so the
shipped training policy ignores it and keeps legal final prefixes. `best` is
retained only as an experimental processor option. There is no runner-side
verifier polling, shim, hard early-stop, or repair-round loop. The old default
repair-round count was zero; the retained fresh-sandbox no-implementation retry
is also disabled by default pending a real attempt/chain mapping test.

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

They target the explicitly excluded Megatron/MindSpeed path, while this recipe
follows the official VeOmni NPU path. Per the migration decision, none is
backported in this iteration. Consequently, this example cannot claim the old
single-node 35B/CP8/roughly-160K resource envelope without a separate NPU peak
memory qualification.

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
- The legacy CANN verifier declares CANN Open Software License 2.0 obligations,
  including repository-level licence/notice handling, while Uni-Agent's root is
  Apache-2.0. NPUKernelBench and DrKernel redistribution terms also need owner
  confirmation. No such payload is copied until maintainers/legal approve the
  resulting LICENSE/NOTICE layout.
- The sandbox image and evaluator scripts are deployment artifacts. Publish a
  digest and their licences separately; the shipped placeholder intentionally
  fails until it is replaced with that immutable reviewed digest.
- The legacy verifier cannot be copied or invoked unchanged: its current
  wrapper exits nonzero when only some cases pass, whereas this recipe reserves
  nonzero for evaluation-infrastructure failure. The reviewed final orchestrator
  must emit positive, consistent, implementation-attested partial counts and
  return zero; it must not hide genuine infrastructure failure with `|| true`.
- Agent-controlled verifier JSON, implementation files, and artifacts are
  accepted only as bounded regular non-symlink files. Performance data must be
  finite and attest the exact verified implementation digest.
- Digest matching alone cannot authenticate case counts written by candidate
  code. The image's trusted parent must execute candidates in a child
  process/namespace without write access to attestation outputs or harness
  dependencies, wait for descendants, and only then atomically write final
  JSON. This is a hard image qualification gate because those assets are not
  redistributed by this recipe.
- The image must install root-owned verifier/cleanup/NPU-lease assets and mount
  a lock namespace genuinely shared by every sandbox that can reach the same
  physical devices. `--check` proves file shape/ownership, not cross-container
  sharedness.
- Both agent-time `tools/verify_once.sh` and final verification must enter
  `/opt/triton-agent-tools/with_npu_lease.py`. Direct, unleased accelerator
  execution is outside the supported image contract.
- The Gateway base URL must be reachable from the Ray runner. For OpenYuanRong,
  only Claude's session URL is rewritten to the provider tunnel; reward posting
  remains runner-side. Provider authentication and tunnel routing must be
  integration-tested in the deployment network.
- The stock `/sessions/{id}/reward_info` route has no runner-only capability.
  Strict final POST makes delivery failure fatal but cannot authenticate that
  route. Training is blocked until the sandbox can reach only its
  `/v1/messages` path (with direct Gateway egress denied), or an independently
  reviewed Gateway capability mechanism is available. No Gateway patch is
  included here, per the agreed migration scope.
- OpenYuanRong's current provider adapter retains `runtime_timeout` locally but
  does not forward it to the SDK constructor. Independent operation timeouts,
  context cleanup, SDK `idle_timeout`, and an external TTL/reaper are therefore
  all required. The configured 4200-second budget covers setup + 1800-second
  agent + cleanup + 1200-second verify + artifact collection with margin.

## Verification matrix

| Check | CPU/CI | Real NPU/deployment |
| --- | --- | --- |
| Python compile, Ruff, Bash syntax | Required, automated | Same image revision |
| Stable UID, leakage, JSONL, DrKernel duplicates/levels/invalid refs | Required, automated | Re-run on full pinned source |
| Reward and verifier attestation (including zero-case rejection) | Required, automated | Trusted verifier must overwrite outputs |
| Sandbox destroy/retry and pre-verify process cleanup | Fake provider test | Kill/cancel/TTL fault injection |
| NPU lease parsing and runner env forwarding | Required, automated | Concurrent verifier test on every physical device/host |
| Trajectory alignment/crop/best/no-impl/empty policy | Required, automated | Real multi-chain Gateway rollout |
| Hydra configuration and VeOmni topology | Syntax/compose smoke | One optimizer step, then scale/memory qualification |
| Gateway tunnel and reward posting | Copy-on-write + strict-failure unit test | Real OpenYuanRong path ACL/capability |

An all-empty postprocessed GRPO prompt is a hard pre-training validation failure,
not a sample to pad with an invalid trajectory. Validate surviving group sizes
on a real rollout before starting a long run.

## Local verification record (2026-08-20)

- Example CPU suite against all three applied core patches: `53 passed, 1
  skipped`; the skip is the Linux/root process-group lease integration on the
  Windows review host.
- Combined patched framework suite: `70 passed`; Sandbox lifecycle + Docker
  focus: `18 passed`. Both patch application orders produced the same Git tree.
- Ruff check/format, `compileall`, Bash syntax, patch whitespace/apply checks,
  and `git diff --check`: passed.
- verl v0.9 Hydra composition accepted all 128 training overrides, including
  `selection=all_final`, strict reward posting, vLLM async rollout, and the
  VeOmni per-GPU token limits.
- Full local legacy inputs were read without vendoring them: NPUKernelBench
  `101` total / `61` selected level-1+2 rows; DrKernel `1999` train / `196`
  validation rows; all emitted UIDs were unique.
- Synthetic JSONL and Parquet generation each produced one disjoint train and
  validation row with verified source, manifest, output, and UID digests.

These are CPU/config/data checks. No result above substitutes for the real NPU,
Gateway ACL/capability, provider TTL, shared-device lock, verifier-image, GRPO
cardinality, one-step VeOmni, or peak-memory qualification gates in the table.
