"""Recipe-local Docker provider for a pool of remote sandbox hosts."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from uni_agent.sandbox.base import ExecResult
from uni_agent.sandbox.docker import DockerSandbox
from uni_agent.sandbox.registry import register_sandbox

if TYPE_CHECKING:
    from uni_agent.sandbox.base import SandboxConfig


@register_sandbox("triton_remote_docker")
class RemoteDockerSandbox(DockerSandbox):
    """Use the stock Docker sandbox through one remote Docker endpoint."""

    def __init__(
        self,
        *,
        docker_host: str,
        image: str,
        npu_lock_dir: str,
        runtime_timeout: float = 3600.0,
        docker_binary: str = "docker",
        run_args: list[str] | None = None,
        pull_policy: str = "never",
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        if not docker_host:
            raise ValueError("docker_host is required")
        if not math.isfinite(runtime_timeout) or runtime_timeout <= 0:
            raise ValueError("runtime_timeout must be finite and positive")

        args = list(run_args or [])
        args.extend(["--volume", f"{npu_lock_dir}:{npu_lock_dir}"])
        if cwd:
            args.extend(["--workdir", cwd])
        for key, value in (env or {}).items():
            args.extend(["--env", f"{key}={value}"])

        self.docker_host = docker_host
        self.runtime_timeout = runtime_timeout
        super().__init__(
            image=image,
            docker_binary=docker_binary,
            run_args=args,
            pull_policy=pull_policy,
            entrypoint="sleep",
            command=[str(math.ceil(runtime_timeout))],
        )

    @classmethod
    def from_config(cls, config: SandboxConfig) -> RemoteDockerSandbox:
        return cls(image=config.image, runtime_timeout=config.runtime_timeout, **config.sandbox_kwargs)

    async def _run_docker(self, *args: str, timeout: float | None = None) -> ExecResult:
        return await super()._run_docker("--host", self.docker_host, *args, timeout=timeout)
