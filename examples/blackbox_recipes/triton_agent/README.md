# Triton Ascend operator-generation recipe

This example migrates the NPU operator task and training-trajectory behavior to
the stock Uni-Agent Task, Sandbox, Gateway, and Claude Code APIs. It targets
Uni-Agent `26a49e2646dfe2cb1caa668df2b112ed0afc3ad1` plus the `verl` v0.9.0
submodule (`483b8a009ba3a97563edee3a19887e4862b8094a`). It intentionally contains no
custom Gateway, KV-cache router, Megatron, checkpoint, debug, Claude protocol
shim, or NPU-memory patches.

See `MIGRATION.md` for the exact scope matrix, old/new behavior differences,
NPU-memory audit, external blockers, and verification ledger.

## Required core patches

Apply the three independent patches shipped under the repository-level
`patches/` directory before running training:

1. the framework-level `trajectory_postprocessor_fqn` hook; and
2. bounded, JSON-serializable `TaskResult.extra_info` reward forwarding plus
   opt-in fail-closed reward delivery; and
3. cancellation-safe, bounded Sandbox lifecycle cleanup with retryable
   OpenYuanRong kill failure.

The hook is configured once in `run_train.sh` at
`actor_rollout_ref.rollout.custom.agent_framework`. Built-in
`trajectory_selection` remains `all`, so the pure recipe processor receives
every materialized chain before scoring and TransferQueue writes.
The lifecycle patch uses `SANDBOX_STOP_TIMEOUT`; `run_train.sh` defaults it to
120 seconds. This is a local cleanup bound, not a provider TTL.

This repository intentionally does not redistribute the benchmark payload or
the CANN-licensed verifier/image assets. The included synthetic fixture tests
the data and task schema, not an NPU rollout. A reviewed, digest-pinned image
that satisfies the contract below is therefore a hard prerequisite, not an
optional production refinement.

## Runtime architecture

Each rollout session invokes `TritonOperatorTask.run`. An attempt enters one
`Sandbox` async context, stages its task, launches the existing
`ClaudeCodeAgent`, performs one bounded final verification, optionally downloads
small artifacts, and exits the context. When explicitly enabled, a missing
implementation gets at most one retry in a newly created sandbox while
retaining the same Gateway session; the first sandbox is fully destroyed before
retry staging begins. The shipped config keeps this retry disabled until chain
identity is qualified. Sandbox
destruction is therefore the normal process/resource cleanup path on success,
exceptions, cancellation, and retry. The remote provider must also configure a
TTL/reaper for hard-killed Ray workers.

When OpenYuanRong cannot directly reach the per-session Gateway, the example's
thin runner copies the task config, injects that session's `upstream` and
`proxy_port`, and rewrites only Claude Code's URL to the in-sandbox tunnel.
Reward reporting retains the original runner-side URL. Disable
`SANDBOX_GATEWAY_TUNNEL` for providers with direct network reachability. These
two tunnel kwargs are an OpenYuanRong provider contract, not a portable Sandbox
API; another remote provider needs an equivalent example-local binding.

The stock Gateway reward endpoint is not authenticated. The task runner uses
the core patch's `reward_post_strict=True`, so a failed trusted final POST aborts
the session instead of consuming an earlier value, but fail-closed delivery is
not endpoint authentication. Before any real training, give the runner a
runner-only reward capability or enforce a network proxy/ACL that exposes only
this session's `/v1/messages` route to the sandbox and denies direct Gateway
reachability. The raw OpenYuanRong host/port tunnel does not prove that property
by itself. Reward-endpoint isolation is a hard deployment gate; this recipe
does not reintroduce the explicitly excluded Gateway changes.

One sandbox is not automatically one NPU. This recipe makes exclusivity
executable: the runner injects a reviewed device list and lock location, and
both agent-time and final verifier commands must enter the root-owned
`with_npu_lease.py` wrapper. The wrapper takes an advisory lock, sets Ascend
device-visibility variables, runs the verifier in its own process group, and
keeps the lock FD inherited so an uncatchably killed wrapper cannot release a
device still used by its child. Keep evaluator NPUs separate from
training/rollout NPUs and enforce
`MAX_CONCURRENT_SESSIONS <= EVALUATOR_NPU_COUNT`.

The lock directory must be a genuinely shared mount for every sandbox that can
reach the same physical-device namespace. Pre-create root-owned mode-0666
`device-<id>.lock` files in a root-owned sticky directory; the wrapper refuses
to create or trust agent-owned locks. A host-local device-ID namespace needs a
host-local shared mount, while a remote multi-host scheduler needs an equivalent
provider binding. The wrapper's `--check` cannot prove that two containers see
the same inode, so concurrent integration testing remains mandatory.

## Sandbox image contract

`config/task_config.yaml` expects the image to declare `WORKDIR /workspace`, the
provider `cwd` to remain `/workspace`, and `/opt/triton-agent-template` to
contain reviewed, redistributable assets. Replace its deliberately invalid
`REPLACE_WITH_PINNED_TRITON_AGENT_NPU_IMAGE_DIGEST` value with an immutable
reviewed digest. Every attempt runs `pwd` without a
workdir override and fails before Claude starts unless it equals
`workspace_dir`; this is required because the stock `ClaudeCodeAgent` correctly
uses the sandbox process cwd rather than a recipe-specific launch path.
Claude must run as a non-root user. Before rollout, the task rejects root and
checks that every trusted tool is a non-symlink regular executable owned by
root, not group/other writable, beneath a root-owned non-writable directory
chain. A dedicated worker is not a reason to run the agent as root: root could
replace the verifier and invalidate reward trust.

The template contains:

- `INSTRUCTIONS.md`;
- `tools/verify_once.sh`, a symlink to the immutable image command below;
- protected verifier code and image-owned test inputs; and
- the pinned Claude Code CLI (the stock agent can install it, but a pinned image
  is reproducible and avoids runtime internet access).

The image must separately provide immutable, root-owned executables outside the
agent workspace:

- `/opt/triton-agent-tools/cleanup_task_processes.sh`, scoped to the session
  cgroup/PIDs and required both immediately before final verification and in the
  attempt's `finally` cleanup;
- `/opt/triton-agent-tools/final_verify.sh`, which uses protected inputs,
  independently recompiles/runs the selected implementation, and replaces the
  final verifier JSON;
- `/opt/triton-agent-tools/verify_once.sh`, accepting one safe operator name
  and invoking `with_npu_lease.py` before any NPU work; and
- `/opt/triton-agent-tools/with_npu_lease.py`, built from the reviewed recipe
  asset and installed root-owned/non-writable.

Before Claude starts, the task resolves the workspace `tools/verify_once.sh`
symlink and requires that it point exactly to the trusted image command. This
preflight plus the native transcript hook verifies the supported agent-time
path; the deployment should still deny direct unleased device access where its
container/runtime supports that policy.

Task-specific reference code and sidecars necessarily enter the agent-visible
workspace; chmod alone is not a trust boundary. After the agent exits and
required process cleanup succeeds, the runner atomically overwrites every
reference/sidecar plus `TASK_METADATA.json` from original sample metadata and
records an expected SHA-256 manifest. It removes the complete agent-controlled
`output` subtree without following a symlink argument, recreates
`output/verify`, runs the trusted verifier, requires process cleanup again, then
re-hashes every input. Any post-reset change forces reward to zero. Runner-side
reads accept only bounded regular non-symlink files; verifier JSON is limited to
1 MiB and implementation files to 2 MiB, with a bounded remote read.

The verifier should produce:

- `output/verify/verify_result.json` with case totals, compile status,
  `verified_impl_path`, and `verified_impl_sha256`;
- `output/verify/perf_result.json` with a finite nonnegative speedup plus the
  same `verified_impl_path` and `verified_impl_sha256` when benchmarking
  succeeds;
- optionally `verify_result_summary.json`.

The trusted orchestrator must not import or execute candidate code in the same
interpreter/security context that writes those attestations. Run the candidate
in a child process or stronger namespace that cannot write verifier outputs or
modify the harness, wait for all candidate descendants to exit, and have the
root-owned trusted parent atomically create the final JSON. Every transitive
harness dependency and protected case input must likewise be image-owned and
non-writable. Candidate-written attestation JSON is untrusted even when its
claimed implementation digest happens to match.

Exit code zero means the evaluation and attestation infrastructure completed,
not that every correctness case passed. Partial case failure must be represented
by consistent counts while still returning zero so the preserved partial-credit
reward can be computed. A nonzero exit is treated as an infrastructure failure
and forces reward to zero. Boolean fields must be JSON booleans, not strings;
non-finite JSON constants and performance values are rejected.

The attested path must be exactly the current or best implementation filename,
and its SHA-256 must match bytes read back by the task. Otherwise reward is
forced to zero. The verifier must also report `total_cases > 0` and internally
consistent passed/failed totals. Agent-writable
`metrics.json`/`metrics_best.json` never supplies reward numbers; only trusted
final verifier outputs do. The paired
`metrics_best.json` and best implementation can supply only the experimental
assistant-turn hint described below; the training default does not consume it.

The task installs project-local Claude Code `PreToolUse` and `PostToolUse` hooks
using the native Bash hook API. Only a command containing
`tools/verify_once.sh` is observed. The pre-hook counts unique assistant messages
from the provided `transcript_path` and fingerprints the existing best pair;
the post-hook annotates `metrics_best.json` only if both best files exist and
both contents changed. The task rechecks the recorded best-implementation
SHA-256 before forwarding `train_best`, preventing a stale assistant index
after final verification. This does not make the scalar index trusted: the
hook state is agent-writable and has no Gateway chain identity. Consequently,
`run_train.sh` defaults to `selection=all_final`. `selection=best` is an
explicit experiment only after a runner/Gateway-owned snapshot identity is
available; even then, multiple chains fall back to each legal final prefix.

Unlike the old protocol shim, stock `ClaudeCodeAgent` does not expose stream
events for runner-side verifier polling or a hard verifier-patience early stop.
The migration uses native hook input only for snapshot alignment and relies on
`max_turns`/`run_timeout`; there is no repair-round loop. Details are recorded in
`MIGRATION.md`.

If `retry_on_missing_impl` is opted in, the second fresh-sandbox attempt creates
a new Gateway chain without private abort/create/reset APIs. Reward metadata
tells the processor to discard the previous chain. Gateway trajectories
currently expose no attempt/chain identity, so the recipe would have to treat
the last finalized chain as the retry result. The shipped value is therefore
`false`; a real retry plus Claude background/compaction rollout must prove the
mapping before enabling it.

## Data preparation

Large benchmark payloads and files with unresolved redistribution terms are not
vendored. Create an immutable source manifest from
`config/data_source.example.json`, download and review the archive, and obtain
maintainer/licence approval. Its `sha256` must be the preparer's deterministic
digest of the exact extracted train/validation inputs (the manifest file itself,
`__pycache__`, and `.pyc` files are excluded); a placeholder or mismatch is a
hard error. Record and verify any original archive digest separately. Then
prepare genuinely disjoint directories. NPUKernelBench JSON and multi-line
JSONL sidecars are preserved;
the verifier-visible filename is canonicalized. Levels 1 and 2 match the old
default. The latest old training recipe selected all operation families except
three known resource-heavy operators, so those exclusions are explicit:

```bash
python examples/blackbox_recipes/triton_agent/prepare_data.py \
  --train-source /data/reviewed/train \
  --validation-source /data/reviewed/validation \
  --dataset-name reviewed-npu-benchmark \
  --dataset-revision <immutable-revision> \
  --dataset-kind npukernelbench \
  --npukernelbench-levels 1,2 \
  --filter-profile none \
  --exclude-op MoeInitRouting \
  --exclude-op SwiGLUQuant \
  --exclude-op KVRMSNormRopeCache \
  --arch ascend910b1 \
  --source-manifest /data/reviewed/source_manifest.json \
  --output-dir /data/triton-agent
```

For the official DrKernel parquet layout (`training_*.parquet` plus
`validation_level*.parquet`), use:

```bash
python examples/blackbox_recipes/triton_agent/prepare_data.py \
  --train-source /data/drkernel \
  --validation-source /data/drkernel \
  --dataset-name drkernel \
  --dataset-revision <immutable-revision> \
  --dataset-kind drkernel \
  --drkernel-num-cases 10 \
  --drkernel-validation-levels 1,2 \
  --filter-profile none \
  --exclude-op MoeInitRouting \
  --exclude-op SwiGLUQuant \
  --exclude-op KVRMSNormRopeCache \
  --arch ascend910b1 \
  --source-manifest /data/drkernel/source_manifest.json \
  --output-dir /data/triton-agent
```

The DrKernel adapter generates deterministic `get_input_groups`, varying a safe
dynamic batch dimension when static analysis permits and otherwise repeating
the official case. It stably de-duplicates exact training rows, namespaces
reused problem IDs by validation level, and excludes four known-invalid
validation references: `66_Matmul_Dropout_Softmax`,
`80_Gemm_Max_Subtract_GELU`, `83_Conv3d_GroupNorm_Min_Clamp_Dropout`, and
`92_cumsum_exclusive`.

`--filter-profile legacy-warmup` remains an opt-in reproduction of the older
conv/attention exclusion policy and applies symmetrically to train and
validation. Explicit include/exclude keywords, operation names, code length,
and best-effort static input/output element limits are available. Seeded
`--max-train-rows`/`--max-validation-rows` selection occurs only after the full
split-leakage check; UIDs are content-derived before ordering and never contain
row indices.

The preparer derives UIDs from dataset identity, immutable revision, source ID,
recipe schema, architecture, and the final rewritten task/prompt semantics
rather than shuffle order. It rejects overlapping IDs and renamed duplicate
content. `dataset_summary.json` records the verified source/manifest digest and
both generated output digests. To calculate a manifest value before a run:

```bash
python -c "from pathlib import Path; from examples.blackbox_recipes.triton_agent.prepare_data import source_tree_sha256; print(source_tree_sha256([Path('/data/reviewed/train'), Path('/data/reviewed/validation')]))"
```

A two-row synthetic fixture validates the schema only:

```bash
bash examples/blackbox_recipes/triton_agent/prepare_synthetic.sh
```

## Training

Start from the official NPU VeOmni/vLLM-Ascend environment, set the model and
exclusive evaluator capacity, then append cluster/model parallelism overrides:

`RUNTIME_ENV` must make this repository importable on every Ray worker, either
with a reviewed `working_dir`/package upload or by installing the exact recipe
commit in the worker image. It must also make `TASK_CONFIG`, train/validation
Parquet files, model paths, and runner-local artifact destinations resolve
consistently across nodes; a driver-only checkout is insufficient for
`dispatch_mode=ray_task`.

```bash
MODEL_PATH=/models/Qwen3-Coder-30B-A3B-Instruct \
TRAIN_FILE=/data/triton-agent/train.parquet \
VAL_FILE=/data/triton-agent/validation.parquet \
EVALUATOR_NPU_COUNT=8 \
EVALUATOR_NPU_DEVICE_IDS=0,1,2,3,4,5,6,7 \
EVALUATOR_NPU_LOCK_DIR=/var/lock/triton-agent-npu \
MAX_CONCURRENT_SESSIONS=8 \
bash examples/blackbox_recipes/triton_agent/run_train.sh \
  actor_rollout_ref.actor.optim.lr=5e-7
```

Pin a verl-0.9-supported model, CANN, torch/torch-npu, Triton Ascend, vLLM,
vLLM-Ascend, Claude Code, and sandbox image digest. This recipe makes no claim
that the previous 35B/160K/single-node memory envelope survives removal of the
old Megatron-only optimizations; validate peak memory and reduce scale first.

## Trajectory and observability behavior

> **Run-blocking GRPO check:** `empty_policy=drop` can change the surviving
> response count per prompt. Before training, run real multi-chain rollouts and
> assert that every prompt retains the group size expected by the selected
> advantage estimator. An all-empty prompt must be resampled or fail the run;
> never pad it with an invalid/no-op trajectory.

The default processor policy is `all_final`; agent-writable best-turn hints do
not alter credit assignment. The processor validates exact
`response_ids`/`response_mask`/logprob alignment,
crops only on assistant boundaries, clears stale generation-step metadata and
routing captures, and recomputes the assistant-ending turn count. It never maps
a scalar best-turn hint across multiple Gateway chains because chain ordering
is not a global assistant-message timeline. Its empty policy is explicit: the
default `drop` makes the session fail rather than silently training a no-op
sample.

Existing `framework.log`, `task.log`, trajectory JSON/NPZ dumps, failure summary,
and compact reward metadata are used directly. Set `artifact_dir` only when
small generated implementations/verifier JSON must be copied before sandbox
destruction; file size and download time are bounded.

## Checks

```bash
python -m ruff check examples/blackbox_recipes/triton_agent
python -m pytest -q examples/blackbox_recipes/triton_agent/tests
bash -n examples/blackbox_recipes/triton_agent/run_train.sh
```

The unit suite covers stable IDs and split leakage, reward normalization,
sandbox lifecycle, trusted-input reset/attestation, tunnel and NPU-lease config,
token-array alignment, assistant-boundary crops, best-prefix selection, and
explicit empty policies. Before upstreaming, also run one real NPU
verify/benchmark, concurrent device isolation tests, a real Gateway multi-chain
rollout, and a one-step VeOmni train.
