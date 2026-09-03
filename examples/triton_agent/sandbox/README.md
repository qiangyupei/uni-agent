# KernelBench sandbox layer

This directory contains every recipe-owned file installed in the evaluator
image:

- `template/` becomes `/opt/triton-agent-template` and is copied into each
  fresh session workspace;
- `tools/` becomes the root-owned, non-writable `/opt/triton-agent-tools`;
- the Dockerfile links the workspace verifier entry point and verifier scripts
  back to that immutable tools directory.

Build a derived image without overwriting the base tag:

```bash
cd examples/triton_agent/sandbox
bash build_image.sh
```

This produces `triton-claude-code-env:kernel-bench` for inspection. Either use
that tag in `task_config_kernel_bench.yaml`, or build the `latest` tag expected
by the supplied task config as follows.

To replace `triton-claude-code-env:latest`, first retain a base tag so a later
build cannot accidentally inherit from its own output:

```bash
docker tag triton-claude-code-env:latest triton-claude-code-env:base
BASE_IMAGE=triton-claude-code-env:base \
OUTPUT_IMAGE=triton-claude-code-env:latest \
bash build_image.sh
```

Check the resulting image before rollout:

```bash
docker run --rm --entrypoint bash triton-claude-code-env:latest -lc \
  'test "$(id -u)" != 0 && test -x /opt/triton-agent-tools/verify_once.sh && test -d /opt/triton-agent-template'
```

The base image must already contain Claude Code, Python, the Ascend runtime,
torch/torch-npu, Triton Ascend, and the verifier's Python dependencies. The
layer creates and selects a non-root `claude` user with UID/GID 1000 when it is
missing. Override `SANDBOX_USER`, `SANDBOX_UID`, and `SANDBOX_GID` if those IDs
are already assigned differently in the base image.

Because the task uses `pull_policy: never`, build the same image on every
Docker endpoint, or push it to a registry and pre-pull it on every endpoint.
For Docker over SSH, prefix the commands with `DOCKER_HOST=ssh://user@host`.
