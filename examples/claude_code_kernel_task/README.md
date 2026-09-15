# Triton Ascend operator-generation recipe

Train an LLM through UniAgent's Claude Code agent, with operator verification on remote Ascend hosts. Use this branch with its pinned verl revision and a matching installed environment; release v0.9.0 alone does not provide the CT APIs required by this baseline. Do not reapply the historical patches under `patches/`.

## Layout

```text
uni_agent/tasks/kernel_bench/
  task.py                    # workspace, agent execution, result collection
  reward.py                  # verifier metrics and reward
  preprocess.py              # DrKernel / NPUKernelBench preparation
  track_verify_snapshot.py   # best-prefix and early-stop hook

examples/triton_agent/
  task_config_kernel_bench.yaml
  runner.py                  # per-session bindings and Claude output budget
  remote_docker.py, network.py
  trajectory_processor.py
  run_train.sh               # NPU training
  run_train_gpu.sh           # GPU training, remote NPU verification
  sandbox/                   # image template, skills and verifier tools
```

The Task is registered as `triton_operator`. The runner delegates to the stock `run_task`; the Task uses `uni_agent.agents.claude_code`.

## 1. Prepare the sandbox hosts

See [sandbox/README.md](sandbox/README.md) for image construction. The existing base image must contain Claude Code, Python, Ascend/CANN, torch-npu, Triton Ascend, sudo and the `claude` user. The image layer copies files and configures permissions; it does not download packages.

Build the image on each remote Docker daemon:

```bash
cd examples/triton_agent/sandbox
DOCKER_HOST=ssh://root@npu-host-01 OUTPUT_IMAGE=triton-claude-code-env:new bash build_image.sh
```

Use the same image name in `task_config_kernel_bench.yaml`. Check all device/driver bind-mount sources in that file against each host. Retain the notices in [sandbox/NOTICE.md](sandbox/NOTICE.md); review third-party licences before redistribution.

On each NPU host, create the shared lock files:

```bash
sudo install -d -o root -g root -m 1777 /var/lock/triton-agent-npu
for device in 0 1 2 3 4 5 6 7; do
  sudo touch "/var/lock/triton-agent-npu/device-${device}.lock"
  sudo chown root:root "/var/lock/triton-agent-npu/device-${device}.lock"
  sudo chmod 0666 "/var/lock/triton-agent-npu/device-${device}.lock"
done
```

Configure SSH access and trusted host keys from every Ray node that may run tasks, then check:

```bash
docker --host ssh://root@npu-host-01 info
```

`REMOTE_DOCKER_HOSTS` is a comma-separated string, for example `ssh://root@npu-host-01,ssh://root@npu-host-02:2222`. All hosts must provide the configured image and device IDs. The sandbox must reach the Gateway's advertised LAN address; no reverse tunnel is used. Do not expose an unauthenticated Docker TCP endpoint.

Each session creates a fresh container and destroys it after execution. Containers on one host share that host's lock directory; different hosts have independent device pools. Do not use a global NFS lock directory. Verifier calls acquire a device only while verifying/benchmarking, not for the whole session.

Claude runs as a non-root user; the fixed verifier/cleanup commands use sudo for NPU access. Privileged containers and advisory locks are a cooperative deployment model, not an isolation boundary against malicious code. Container TTL is the fallback for hard-killed workers; it cannot guarantee immediate cleanup after host/daemon failures.

## 2. Prepare data

For DrKernel parquet files (`training_*.parquet` and `validation_level*.parquet`):

```bash
python -m uni_agent.tasks.kernel_bench.preprocess \
  --dataset-kind drkernel \
  --train-source /data/drkernel \
  --validation-source /data/drkernel \
  --drkernel-validation-levels 1,2 \
  --output-dir /data/triton-agent
```

For NPUKernelBench source trees with Python operators and JSON case files:

```bash
python -m uni_agent.tasks.kernel_bench.preprocess \
  --train-source /data/bench/train \
  --validation-source /data/bench/validation \
  --output-dir /data/triton-agent
```

Both produce `train.parquet`, `validation.parquet`, and `dataset_summary.json`. No manifest is required or accepted. DrKernel defaults to ten input groups; use `--drkernel-num-cases` to change it. See `--help` for level selection, filtering and sample limits. Benchmark datasets are not included.

## 3. Start training

Run from `examples/triton_agent` in the configured training environment:

```bash
MODEL_PATH=/models/your-model \
TRAIN_FILE=/data/triton-agent/train.parquet \
VAL_FILE=/data/triton-agent/validation.parquet \
REMOTE_DOCKER_HOSTS=ssh://root@npu-host-01,ssh://root@npu-host-02 \
EVALUATOR_NPU_DEVICE_IDS=0,1,2,3,4,5,6,7 \
MAX_RESPONSE_LENGTH=8192 \
TRAIN_TOTAL_LENGTH_LIMIT=32768 \
bash run_train_gpu.sh
```

Use `run_train.sh` for NPU training. Adjust model parallelism and concurrency to the actual cluster. A 32768-token training limit is a smaller bring-up setting, not a guarantee of sufficient memory.

Both launchers submit the repository with Ray `--working-dir`; `RUNTIME_ENV` is optional for deployment-specific dependencies/environment. The installed verl and accelerator packages must match the selected source revision.

- `MAX_RESPONSE_LENGTH` sets rollout `response_length` and Claude's `CLAUDE_CODE_MAX_OUTPUT_TOKENS`. An independent Gateway per-call cap requires the response-length PR.
- `MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH` defines Gateway chain capacity; `MAX_MODEL_LEN` configures the inference engine limit.
- `TRAIN_TOTAL_LENGTH_LIMIT` limits the trajectory prefix retained for training.
- `MAX_CONCURRENT_SESSIONS` controls runner concurrency per framework worker, not the number of NPUs.
- `agent.run_timeout` in the task YAML bounds Claude execution; `sandbox.runtime_timeout` bounds container lifetime.

## Task results and trajectories

The workspace template supplies `CLAUDE.md`, `INSTRUCTIONS.md` and skills. The stock agent passes the single user message to `claude -p`; the dataset's system message is not automatically passed as a CLI system prompt.

During execution, `tools/verify_once.sh` performs AST checking, NPU correctness testing and, after full correctness, latency benchmarking. It maintains the best implementation/metrics pair. After Claude exits, the Task stops remaining verifier processes and reads that pair, falling back to staged verifier artifacts or current metrics. It does not run a final verification again.

Rewards retain AST, compilation, correctness and speedup components. `finished=True` describes agent completion, not correctness; use accuracy/pass rate to judge verification.

The Claude hook records best-snapshot assistant indices and cooperatively stops searches after configured non-improvement counts. Defaults: seven verify calls before correctness (best reward at least 0.15), three latency calls after correctness. Set a patience to zero to disable that phase. Do not remove the hook if best-prefix or early-stop is needed.

The postprocessor selects best-prefix trajectories at assistant boundaries. Multiple chains fall back to `all_final`, since a scalar assistant index cannot identify a Gateway chain. Missing implementations are filtered; retry is disabled by default. Token/mask/logprob alignment and framework reward fields are preserved.

## Logs

Gateway timeout diagnostics use the `gateway_request` prefix in Gateway Ray worker logs and the collected main log. Each HTTP request has a unique `request` ID plus its `session` ID. Events cover preparation/lock wait, backend start/end, decoding, response headers/body completion, observed disconnects and errors. Every 60 seconds, `waiting` reports the active stage; `elapsed_s` is measured from receipt. Bodies and headers are not logged. Backend elapsed time includes routing, queueing and generation; the separately logged backend request ID is unchanged. Gateway currently waits for the full backend result before sending SSE. Logging does not add keepalives or change timeout/retry behavior. Disconnects are recorded only when ASGI observes them; response completion does not prove client receipt.

The launchers select the recipe's `KernelAgentFramework` through `framework_class_fqn`. It adds a `kernel_bench` object to each trajectory's JSON summary: best/selected assistant indices (zero-based), requested/applied selection, crop reason, verify count, correctness/latency non-improvement counts, and early-stop reason. Missing hook state is recorded as `null`, not zero. Counts describe the final attempt, not just the selected prefix. This uses a small override of the Framework's private summary method; no generic UniAgent code is changed. Filtered sessions still have no trajectory dump.

Each completed Task prints one `[triton-result]` line to the Ray job's main log and session logger, including sample/session identity, operator name, passed/total cases, reward, metric source and completion status. This reports the selected verification snapshot, not whether the framework later retains the trajectory. Tasks that raise before returning a result are reported through the existing exception logs.

The GPU launcher saves the main log under `LOG_DIR/<experiment>.log`. Both launchers save per-session logs under `LOG_DIR/<experiment>/step_<n>/<session_id>/`: `task.log`, `framework.log`, and, when trajectory dumping is reached, `trajectory.json` / `trajectory.npz`. `AGENT_LOG_DIR` is derived from `LOG_DIR` and is not a separate environment override. Ray workers must have access to the same filesystem for these files to appear together.

Claude stdout/stderr logging contains only tails, not a full transcript. `artifact_dir` is disabled by default; enable it in the task YAML to export selected metrics JSON and implementation files before sandbox destruction. It does not export full Claude transcripts or raw verifier logs. In multi-node runs, use shared storage or collect artifacts from each runner host.
