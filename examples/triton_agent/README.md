# Triton Ascend operator-generation recipe

This example migrates the NPU operator task and training-trajectory behavior to
the stock Uni-Agent Task, Sandbox, Gateway, and Claude Code APIs. It targets
Uni-Agent `28174fdab3787d307ae3a96d32d3737b600575a0` plus the `verl` v0.9.0
submodule (`483b8a009ba3a97563edee3a19887e4862b8094a`). It intentionally contains no
custom Gateway, KV-cache router, Megatron, checkpoint, debug, Claude protocol
shim, or NPU-memory patches.

See `MIGRATION.md` for the exact scope matrix, old/new behavior differences,
NPU-memory audit, external blockers, and verification ledger.

The reusable Task, reward, preprocessing, and transcript-hook code lives in
`uni_agent/tasks/kernel_bench`. The `examples/triton_agent` directory contains
only recipe configuration, launchers, remote-sandbox bindings, and deployment
assets. The Task builds Uni-Agent's existing `claude_code` agent; no
recipe-local Agent or protocol shim is used.

## Layout

```text
uni_agent/tasks/kernel_bench/
  task.py                         # Task and TaskConfig
  reward.py                       # metric normalization and reward
  preprocess.py                   # dataset preparation
  assets/track_verify_snapshot.py # Claude Code task hook

examples/triton_agent/
  task_config_kernel_bench.yaml   # agent, task, and sandbox defaults
  runner.py                       # remote-host/NPU binding entry point
  trajectory_processor.py         # best-prefix trajectory policy
  run_train.sh                    # NPU training launcher
  run_train_gpu.sh                # GPU training launcher
  remote_docker.py, network.py    # recipe-specific sandbox glue
  sandbox/                        # evaluator image layer and immutable tools
```

The package name is `kernel_bench`, while the registered Task and prepared-data
route remain `triton_operator` for compatibility with existing datasets.

## Core patches

The intended best-prefix and reward-metadata behavior requires the first two
patches under the repository-level `patches/` directory:

1. the framework-level `trajectory_postprocessor_fqn` hook; and
2. bounded, JSON-serializable `TaskResult.extra_info` reward forwarding.

Those two patches are the minimum functional stack for the healthy training
path. The remaining patches strengthen failure handling but are not required
for initial bring-up.

For a production run, also stack:

3. opt-in fail-closed reward delivery; and
4. cancellation-safe, bounded Sandbox lifecycle cleanup.

The hook is configured once in `run_train.sh` at
`actor_rollout_ref.rollout.custom.agent_framework`. Built-in
`trajectory_selection` remains `all`, so the pure recipe processor receives
every materialized chain before scoring and TransferQueue writes.
The lifecycle patch reads `SANDBOX_STOP_TIMEOUT` in the Ray worker. Set it in
`RUNTIME_ENV.env_vars` when overriding the patch default. The recipe's remote
Docker provider also turns `sandbox.runtime_timeout` into a container TTL.
Without patches 3 and 4, the healthy path still creates, reports, and destroys
the sandbox, but reward POST failures remain best-effort and cancellation
cleanup has the stock lifecycle boundary.

Benchmark datasets are not vendored. The local image layer under `sandbox/`
contains the legacy numerical verifier and selected skill material needed by
the recipe; retain their headers and complete the licence review recorded in
`sandbox/NOTICE.md` before redistribution. The included tests do not replace a
real NPU rollout. Use a digest-pinned image for production.

## Runtime architecture

Each rollout session invokes
`uni_agent.tasks.kernel_bench.task.TritonOperatorTask.run`. An attempt enters one
`Sandbox` async context, stages its task, launches the existing
`ClaudeCodeAgent`, stops remaining task processes, selects the metrics produced
by agent-time verifier calls, optionally downloads small artifacts, and exits
the context. It does **not** run correctness or latency verification again after
Claude exits. When explicitly enabled, a missing
implementation gets at most one retry in a newly created sandbox while
retaining the same Gateway session; the first sandbox is fully destroyed before
retry staging begins. The shipped config keeps this retry disabled until chain
identity is qualified. Sandbox
destruction is therefore the normal process/resource cleanup path on success,
exceptions, cancellation, and retry. The remote container runs
`sleep <runtime_timeout>` as PID 1 with Docker `--rm`, providing a bounded
fallback if a Ray worker is hard-killed.

The example-local `triton_remote_docker` provider subclasses Uni-Agent's stock
`DockerSandbox`. The runner assigns each session to one configured Docker
endpoint, then the inherited lifecycle performs `docker run`, `exec`, `cp`, and
`rm -f` through `docker --host`. The default deployment uses Docker over SSH and
direct LAN access to the per-session Gateway; it does not use an OpenYuanRong
reverse tunnel. The Gateway must advertise an address reachable from every
sandbox host. Multi-host assignment is a stable hash, not a capacity-aware
scheduler or failover layer; remove unhealthy endpoints before a run or point
the recipe at an independently operated Docker-compatible control plane.
This task configuration must use `examples.triton_agent.runner.run_triton_task`,
which imports and registers the example-local provider before delegating to the
generic Task runner.

The stock Gateway reward endpoint is not authenticated. With the strict
reward-delivery patch applied, the example's `reward_post_strict=True` makes a
failed runner-side reward POST abort the session instead of consuming an earlier
value. Without that patch, stock `run_task` ignores this optional runner argument
and reward delivery remains best-effort. Fail-closed delivery is not endpoint
authentication. Before any real training, give the runner a
runner-only reward capability or enforce a network proxy/ACL that exposes only
this session's `/v1/messages` route to the sandbox and denies direct Gateway
reachability. Direct Docker-host networking does not prove that property.
Reward-endpoint isolation is a hard deployment gate; this recipe
does not reintroduce the explicitly excluded Gateway changes.

One sandbox is not automatically one NPU. This recipe makes exclusivity
executable: the runner injects a reviewed device list and lock location, and
every supported agent-time verifier command must enter the root-owned
`with_npu_lease.py` wrapper. The wrapper takes an advisory lock, sets Ascend
device-visibility variables, runs the verifier in its own process group, and
keeps the lock FD inherited so an uncatchably killed wrapper cannot release a
device still used by its child. Keep evaluator NPUs separate from
training/rollout devices. Session/container concurrency is intentionally
independent of the NPU count: several live Claude sessions may share a host,
while verifier calls wait for a device lock. Waiting for a lock consumes the
session's agent time budget, so tune `MAX_CONCURRENT_SESSIONS` to both Docker
capacity and expected verifier contention.

All remote Docker hosts are expected to expose the same configured device IDs.
The launcher's default session cap is the number of device IDs times the number
of configured hosts.
The runner mounts `EVALUATOR_NPU_LOCK_DIR` from the selected daemon host at the
same path in each container. Containers on one physical host therefore share
that host's device pool; different hosts use independent local files even when
their device IDs match. Do not place this directory on global NFS. Pre-create
root-owned mode-0666 `device-<id>.lock` files in a root-owned sticky directory;
the wrapper refuses to create or trust agent-owned locks. Its `--check` cannot
prove that two containers see the same backing inode, so concurrent integration
testing remains mandatory.

## Sandbox image contract

`task_config_kernel_bench.yaml` uses `triton-claude-code-env:latest` for local
bring-up. Build the supplied `sandbox/` layer on every remote Docker daemon;
replace the mutable tag with an immutable reviewed digest for production. The
image declares `WORKDIR /workspace`, the provider `cwd` remains `/workspace`,
and `/opt/triton-agent-template` contains the fresh-workspace template. Every
attempt runs `pwd` without a
workdir override and fails before Claude starts unless it equals
`workspace_dir`; this is required because the stock `ClaudeCodeAgent` correctly
uses the sandbox process cwd rather than a recipe-specific launch path.
Claude must run as a non-root user. Before rollout, the task rejects root and
checks that every trusted tool is a non-symlink regular executable owned by
root, not group/other writable, beneath a root-owned non-writable directory
chain. A dedicated worker is not a reason to run the agent as root: root could
replace the verifier and invalidate reward trust.

The template contains:

- `CLAUDE.md`, `INSTRUCTIONS.md`, and selected Triton/NPU skills;
- `tools/verify_once.sh`, a symlink to the immutable image command below;
- a verifier `scripts` symlink to immutable image-owned code.

The base `triton-claude-code-env` image must provide a pinned Claude Code CLI
release supporting `PostToolUseFailure` and universal `continue: false` hook
output. The stock agent can install Claude, but a pinned image is reproducible
and avoids runtime internet access.

The image must separately provide immutable, root-owned executables outside the
agent workspace:

- `/opt/triton-agent-tools/cleanup_task_processes.sh`, which terminates
  registered verifier process groups before final artifact selection and in the
  attempt's `finally` cleanup;
- `/opt/triton-agent-tools/verify_once.sh`, accepting one safe operator name
  and implementing the retained agent-time stage, correctness, benchmark, and
  best-snapshot flow; and
- `/opt/triton-agent-tools/with_npu_lease.py`, built from the reviewed recipe
  asset, installed root-owned/non-writable, and invoked by `verify_once.sh`
  before any NPU work.

The shipped Docker `run_args` mirror the old Ascend deployment's privileged
device and driver mounts. Adjust host paths to the qualified cluster image and
verify every bind source exists on every daemon host before rollout.
Because privileged containers can bypass an advisory file lock and access a
device directly, this is a cooperative resource-sharing contract, not a
security boundary against malicious task code.

The image-owned `verify_once.sh` must preserve the previous execution contract:

1. stage the current implementation in `output/verify`;
2. run correctness under the NPU lease;
3. run the latency benchmark only after full correctness; and
4. when the rank improves, update `metrics_best.json` and
   `src/<op>_triton_ascend_impl_best.py` as one logical snapshot. The best
   metrics must include the numeric `reward` used by correctness early-stop.

Before Claude starts, the Task resolves the workspace `tools/verify_once.sh`
symlink and requires it to point exactly to the immutable image command. The
deployment should still deny direct unleased device access where its runtime
supports that policy.

After Claude exits, required process cleanup makes the workspace quiescent and
the Task selects results in the same order as the previous training code:

1. a substantive best implementation paired with `metrics_best.json`;
2. otherwise, `output/verify/verify_result.json`, optional
   `verify_result_summary.json`/`perf_result.json`, and the staged implementation
   are used to recover a best pair;
3. otherwise, a substantive current implementation paired with `metrics.json`;
4. otherwise, the attempt reports missing implementation or missing metrics.

The Task rejects oversized, malformed, non-finite, or symlink artifact files and
checks implementation shape and available snapshot digests. Serialized reward
fields in the workspace are ignored: `uni_agent.tasks.kernel_bench.reward`
recomputes
AST/compile/correctness/speedup components from the selected raw metrics, and a
full pass is represented by pass rate one rather than a separate bonus.

This is deliberately the legacy trust model. The implementation, verifier JSON,
`metrics.json`, and `metrics_best.json` all remain writable by the agent. A
matching digest, timestamp, or best pair detects accidental mismatch but cannot
authenticate case counts or latency because the agent can rewrite both sides.
There is no runner-owned trust proof, and the Task does not invoke the evaluator
after Claude exits. Use this recipe only where that tradeoff is accepted;
restoring trusted reward would require a runner/provider-owned verifier boundary
rather than treating these checks as one.

The task installs project-local Claude Code `PreToolUse`, `PostToolUse`, and
`PostToolUseFailure` hooks for Bash. Only commands containing
`tools/verify_once.sh` are observed. The pre-hook counts unique assistant
messages and fingerprints the best pair. After either a successful or failed
verifier call, the post-hook binds a changed implementation to the current
assistant turn; a better remeasurement of unchanged implementation bytes keeps
the previous binding. The Task rechecks that binding against the selected best
implementation before forwarding `train_best`.

The same post-hook implements the two old patience phases. Before full
correctness, it stops after 7 consecutive verifier calls without a best update,
provided the best reward is at least 0.15. After full correctness, it stops
after 3 consecutive calls without a latency improvement. Configure these with
`verify_early_stop_patience`, `verify_early_stop_min_reward`, and
`latency_optimize_patience`; a patience of zero disables that phase. At the
threshold the hook returns Claude Code's native `continue: false`, so this is a
cooperative agent stop rather than an OS process kill. A valid stop marker plus
a digest-checked `train_best` is treated as a finished episode; `run_timeout=9000`
remains the outer wall-clock bound.

`run_train.sh` now requests `selection=best`. A single finalized trajectory is
cropped to the recorded assistant boundary; multiple Gateway chains still use
`best_fallback=all_final` because the scalar hook index has no chain identity.
The hook state and numerical metrics remain agent-writable under this recipe's
documented legacy trust model, and there is still no runner-side protocol shim,
hard kill, or repair-round loop. Details are recorded in `MIGRATION.md`.

If `retry_on_missing_impl` is opted in, the second fresh-sandbox attempt creates
a new Gateway chain without private abort/create/reset APIs. Reward metadata
tells the processor to discard the previous chain. Gateway trajectories
currently expose no attempt/chain identity, so the recipe would have to treat
the last finalized chain as the retry result. The shipped value is therefore
`false`; a real retry plus Claude background/compaction rollout must prove the
mapping before enabling it.

## Data preparation

Benchmark payloads are not vendored. For an NPUKernelBench-style source tree,
prepare disjoint train and validation directories and run:

```bash
python -m uni_agent.tasks.kernel_bench.preprocess \
  --train-source /data/reviewed/train \
  --validation-source /data/reviewed/validation \
  --output-dir /data/triton-agent
```

For the official DrKernel parquet layout (`training_*.parquet` plus
`validation_level*.parquet`), use:

```bash
python -m uni_agent.tasks.kernel_bench.preprocess \
  --train-source /data/drkernel \
  --validation-source /data/drkernel \
  --dataset-kind drkernel \
  --drkernel-validation-levels 1,2 \
  --output-dir /data/triton-agent
```

Both commands write `train.parquet`, `validation.parquet`, and
`dataset_summary.json`. The defaults are architecture `ascend910b1`, dataset
revision `local`, ten DrKernel input groups, and NPUKernelBench levels 1 and 2.

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

The preparer derives stable UIDs, rejects train/validation overlap, and records
the generated output digests in `dataset_summary.json`. For an audited run,
`--dataset-name`, `--dataset-revision`, and `--source-manifest` may be supplied;
the optional manifest format is shown in `config/data_source.example.json`.
Calculate its source digest with:

```bash
python -c "from pathlib import Path; from uni_agent.tasks.kernel_bench.preprocess import source_tree_sha256; print(source_tree_sha256([Path('/data/reviewed/train'), Path('/data/reviewed/validation')]))"
```

## Training

First prepare each homogeneous remote NPU host. The bind-mount source below is
resolved by that host's Docker daemon, so repeat it independently on every
host:

```bash
sudo install -d -o root -g root -m 1777 /var/lock/triton-agent-npu
for device in 0 1 2 3 4 5 6 7; do
  sudo touch "/var/lock/triton-agent-npu/device-${device}.lock"
  sudo chown root:root "/var/lock/triton-agent-npu/device-${device}.lock"
  sudo chmod 0666 "/var/lock/triton-agent-npu/device-${device}.lock"
done

cd /path/to/uni-agent/examples/triton_agent/sandbox
OUTPUT_IMAGE=triton-claude-code-env:new bash build_image.sh
```

For a remote daemon, set `DOCKER_HOST=ssh://sandbox-user@npu-host-01` for the
three Docker commands. Repeat for every entry in `REMOTE_DOCKER_HOSTS`, or push
the derived image to a registry and pre-pull it on every daemon. See
`sandbox/README.md` for details.

Configure Docker-over-SSH access from every Ray node that may execute an Agent
Runner, and verify it before training:

```bash
docker --host ssh://sandbox-user@npu-host-01 info
```

`REMOTE_DOCKER_HOSTS` is one comma-separated string of Docker `--host`
endpoints, not a YAML or shell array. The recommended form is:

```text
ssh://sandbox-user@npu-host-01,ssh://sandbox-user@npu-host-02:2222
```

Use `unix:///var/run/docker.sock` for a daemon local to every runner process.
An authenticated `tcp://host:2376` endpoint is also accepted by Docker, but its
TLS environment and certificate files must be available to the Ray workers.
Do not include empty entries; each endpoint must already contain the configured
sandbox image and expose the same `EVALUATOR_NPU_DEVICE_IDS` list.

Do not expose an unauthenticated Docker TCP daemon. SSH keys, host keys, and any
TLS credentials should be provisioned outside Hydra arguments and logs.

Then start from the official NPU VeOmni/vLLM-Ascend environment, set the model
and remote evaluator pool, and append cluster/model parallelism overrides:

Both launchers locate the repository root from their own path and pass it to
`ray job submit --working-dir`. This makes the example modules and relative
`TASK_CONFIG` available to the Ray job without a separate runtime-env file, so
the scripts may be launched directly from `examples/triton_agent`. The verl
v0.9.0 environment should already be installed on the Ray image.

`RUNTIME_ENV` is optional and is only needed for deployment-specific packages
or environment variables that must be propagated to Ray workers. Do not put a
second `working_dir` in that file. The stock Sandbox startup-concurrency default
normally needs no override; set `SANDBOX_STOP_TIMEOUT` there only when changing
the lifecycle patch default.

```bash
cd examples/triton_agent

MODEL_PATH=/models/Qwen3-Coder-30B-A3B-Instruct \
TRAIN_FILE=/data/triton-agent/train.parquet \
VAL_FILE=/data/triton-agent/validation.parquet \
REMOTE_DOCKER_HOSTS=ssh://sandbox-user@npu-host-01,ssh://sandbox-user@npu-host-02 \
EVALUATOR_NPU_DEVICE_IDS=0,1,2,3,4,5,6,7 \
EVALUATOR_NPU_LOCK_DIR=/var/lock/triton-agent-npu \
MAX_CONCURRENT_SESSIONS=8 \
bash run_train.sh \
  actor_rollout_ref.actor.optim.lr=5e-7
```

For NVIDIA training/rollout with the same remote Ascend verifier pool, use the
stock verl 0.9 Megatron/V1 synchronous launcher:

```bash
cd examples/triton_agent

REMOTE_DOCKER_HOSTS=ssh://sandbox-user@npu-host-01,ssh://sandbox-user@npu-host-02 \
EVALUATOR_NPU_DEVICE_IDS=0,1,2,3,4,5,6,7 \
MODEL_PATH=/models/Qwen3-Coder-30B-A3B-Instruct \
bash run_train_gpu.sh
```

`run_train_gpu.sh` changes only the training/rollout backend to NVIDIA
Megatron/vLLM. It deliberately omits the legacy fork's custom Megatron memory,
Gateway, KV-routing, checkpoint, debug, and Claude shim changes. Put cluster
specific NCCL, NIC, CUDA allocator, and vLLM environment variables in an
optional `RUNTIME_ENV` file when Ray must propagate them to every worker.

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

The processor function itself defaults to `all_final`, while the training
launcher explicitly selects `best`. The processor validates exact
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
python -m ruff check uni_agent/tasks/kernel_bench examples/triton_agent \
  tests/uni_agent/tasks/kernel_bench
python -m pytest -q tests/uni_agent/tasks/kernel_bench examples/triton_agent/tests
bash -n examples/triton_agent/run_train.sh
bash -n examples/triton_agent/run_train_gpu.sh
```

The unit suite covers stable IDs and split leakage, reward normalization,
sandbox lifecycle, legacy metric selection/fallback, transcript binding,
two-phase early-stop, remote Docker and NPU-lease config, token-array alignment,
assistant-boundary crops, best-prefix selection, and explicit empty policies.
Before upstreaming, also run one real NPU
verify/benchmark, concurrent device isolation tests, a real Gateway multi-chain
rollout, and a one-step VeOmni train.
