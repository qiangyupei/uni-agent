from __future__ import annotations

import asyncio
from pathlib import Path

from uni_agent.sandbox.base import ExecResult, SandboxConfig
from uni_agent.sandbox.docker import DockerSandbox
from uni_agent.tasks import TaskConfigResolver

from ..network import bind_remote_sandbox
from ..remote_docker import RemoteDockerSandbox


def test_remote_docker_reuses_stock_lifecycle(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    async def fake_run_docker(self, *args: str, timeout: float | None = None) -> ExecResult:
        calls.append(args)
        return ExecResult(exit_code=0, stdout="container-id\n", stderr="")

    monkeypatch.setattr(DockerSandbox, "_run_docker", fake_run_docker)
    sandbox = RemoteDockerSandbox(
        docker_host="ssh://sandbox-a",
        image="registry.example/triton@sha256:deadbeef",
        npu_lock_dir="/var/lock/triton-agent-npu",
        runtime_timeout=30,
        env={"TRITON_EVAL_DEVICE_IDS": "0,1"},
        cwd="/workspace",
        run_args=["--privileged"],
    )

    async def run() -> None:
        async with sandbox:
            pass

    asyncio.run(run())

    assert calls[0][:3] == ("--host", "ssh://sandbox-a", "image")
    run_call = calls[1]
    assert run_call[:4] == ("--host", "ssh://sandbox-a", "run", "--rm")
    assert "--privileged" in run_call
    assert "/var/lock/triton-agent-npu:/var/lock/triton-agent-npu" in run_call
    assert ("--workdir", "/workspace") == run_call[run_call.index("--workdir") :][:2]
    assert ("--env", "TRITON_EVAL_DEVICE_IDS=0,1") == run_call[run_call.index("--env") :][:2]
    assert run_call[-1] == "30"
    assert calls[2][:4] == ("--host", "ssh://sandbox-a", "rm", "-f")


def test_sample_binding_preserves_image_run_args() -> None:
    tools = {"task": {"name": "triton_operator"}}
    tools = bind_remote_sandbox(
        tools,
        hosts="ssh://sandbox-a",
        session_id="session-1",
        devices=("0", "1"),
        lock_dir="/var/lock/triton-agent-npu",
        lock_timeout=30,
    )

    config_path = Path(__file__).parents[1] / "config" / "task_config.yaml"
    resolved = TaskConfigResolver.from_file(str(config_path)).resolve(tools["task"])
    sandbox = RemoteDockerSandbox.from_config(SandboxConfig.model_validate(resolved["sandbox"]))

    assert "--privileged" in sandbox.run_args
    assert "--network=host" in sandbox.run_args
    assert "--volume=/dev:/dev" in sandbox.run_args
    assert "/var/lock/triton-agent-npu:/var/lock/triton-agent-npu" in sandbox.run_args
