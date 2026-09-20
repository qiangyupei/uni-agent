# Ascend Triton Agent sandbox layer

Build files for the evaluator image used by [the recipe](../README.md):

- `template/` becomes `/opt/triton-agent-template`, copied into each session workspace.
- `tools/` becomes `/opt/triton-agent-tools`; the workspace verifier links to these image-owned tools.
- Task execution, reward calculation and runtime-installed hooks remain in `uni_agent/tasks/kernel_bench/`.

## Skills source

The skills and reference material come from [CANNBot Skills PR #205](https://gitcode.com/cann/cannbot-skills/pull/205), with local RL adaptations: prepared-task instructions, compact verifier feedback, a unified verification/benchmark entry point, and integration with best-snapshot tracking and early stopping. They are not an unmodified upstream copy.

## Build

The base image must already contain Claude Code, Python, Bash, `timeout`, sudo/visudo, Ascend/CANN, torch-npu, Triton Ascend, verifier dependencies and the non-root `claude` user. The verifier expects `/usr/local/Ascend/cann/set_env.sh`. This layer copies files and configures permissions; it does not download packages or create users.

```bash
cd examples/claude_code_kernel_task/sandbox
DOCKER_HOST=ssh://root@npu-host-01 \
BASE_IMAGE='<your-prepared-ascend-claude-image>:<tag>' \
bash build_image.sh
```

Replace the placeholder with your prepared base image; `BASE_IMAGE` is required and has no default. The output defaults to `triton-claude-code-env:latest`, matching the task YAML. If you set a different `OUTPUT_IMAGE`, update the YAML too. Use a distinct base-image tag to avoid overwriting it. Set `SANDBOX_USER` if the existing user has a different name.

## Verify

Check the image layout and sudo policy:

```bash
docker --host ssh://root@npu-host-01 run --rm --entrypoint bash \
  triton-claude-code-env:latest -lc \
  'test "$(id -u)" != 0 && sudo -n -l /opt/triton-agent-tools/verify_once.sh smoke >/dev/null && test -d /opt/triton-agent-template'
```

For actual NPU verification, use the complete device/driver mounts and host-local locks described in the parent README. In a prepared session container, with no concurrent verifier running:

```bash
docker --host ssh://root@npu-host-01 exec -w /workspace CONTAINER \
  bash tools/verify_once.sh OP_NAME
```

The Task prepares the operator, implementation and case files. Check `output/verify/verify_result_summary.json` for the verification summary and `verify_result.raw.log` for detailed errors. The wrapper benchmarks fully correct candidates and saves the best snapshot; final Task collection reads those artifacts.
