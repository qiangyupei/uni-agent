# KernelBench sandbox layer

This directory contains every recipe-owned file installed in the evaluator
image:

- `template/` becomes `/opt/triton-agent-template` and is copied into each
  fresh session workspace;
- `tools/` becomes the root-owned, non-writable `/opt/triton-agent-tools`;
- the Dockerfile links the workspace verifier entry point and verifier scripts
  back to that immutable tools directory.

Build the image referenced by the example task configuration:

```bash
cd examples/triton_agent/sandbox
OUTPUT_IMAGE=triton-claude-code-env:new bash build_image.sh
```

This copies the recipe assets from the local build context into the existing
`triton-claude-code-env:latest` base image without downloading packages.

Check the resulting image before rollout:

```bash
docker run --rm --entrypoint bash triton-claude-code-env:new -lc \
  'test "$(id -u)" != 0 && test -x /opt/triton-agent-tools/verify_once.sh && test -d /opt/triton-agent-template'
```

The base image must already contain Claude Code, Python, `timeout`, the Ascend
runtime, torch/torch-npu, Triton Ascend, the verifier's Python dependencies, and
the non-root `claude` user. The derived layer only copies recipe files and sets
their links and permissions. Set `SANDBOX_USER` when the existing user has a
different name; the layer does not install packages or create users.

Because the task uses `pull_policy: never`, build the same image on every
Docker endpoint, or push it to a registry and pre-pull it on every endpoint.
For Docker over SSH, prefix the commands with `DOCKER_HOST=ssh://user@host`.
