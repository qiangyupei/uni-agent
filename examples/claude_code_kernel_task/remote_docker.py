"""Remote Docker provider and per-session host/NPU configuration."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import math
import time
import uuid
from typing import TYPE_CHECKING, Any

from uni_agent.sandbox.base import ExecResult
from uni_agent.sandbox.docker import DockerSandbox
from uni_agent.sandbox.registry import register_sandbox

if TYPE_CHECKING:
    from uni_agent.sandbox.base import SandboxConfig

logger = logging.getLogger(__name__)

_LOCK_DIAGNOSTICS = """
import json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
print('identity', json.dumps({'uid': os.getuid(), 'gid': os.getgid(),
    'devices': os.environ.get('TRITON_EVAL_DEVICE_IDS'), 'configured_lock_dir': str(root),
    'env_lock_dir': os.environ.get('TRITON_EVAL_LOCK_DIR')}))
paths = [pathlib.Path('/var/lock'), root]
paths += [root / ('device-' + d + '.lock')
          for d in os.environ.get('TRITON_EVAL_DEVICE_IDS', '').split(',')[:64] if d]
for path in paths:
    try:
        s = path.lstat()
        print(json.dumps({'path': str(path), 'resolved': str(path.resolve()),
            'mode': oct(s.st_mode), 'uid': s.st_uid, 'gid': s.st_gid,
            'device': s.st_dev, 'inode': s.st_ino}))
    except OSError as exc:
        print(json.dumps({'path': str(path), 'error': str(exc)}))
"""


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
        session_id: str = "",
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
        self.session_id = session_id
        self.npu_lock_dir = npu_lock_dir
        self.runtime_timeout = runtime_timeout
        super().__init__(
            image=image,
            docker_binary=docker_binary,
            run_args=args,
            pull_policy=pull_policy,
            start_timeout=60,
            entrypoint="sleep",
            command=[str(math.ceil(runtime_timeout))],
        )

    @classmethod
    def from_config(cls, config: SandboxConfig) -> RemoteDockerSandbox:
        return cls(image=config.image, runtime_timeout=config.runtime_timeout, **config.sandbox_kwargs)

    async def _run_docker(self, *args: str, timeout: float | None = None) -> ExecResult:
        if timeout is None and (args[:2] == ("image", "inspect") or args[:1] == ("rm",)):
            timeout = 30
        container = self._container_name
        if args[:1] == ("run",) and "--name" in args:
            container = args[args.index("--name") + 1]
        elif args[:1] == ("rm",):
            container = args[-1]
        operation = args[0]
        if operation == "exec" and container in args:
            command_index = args.index(container) + 1
            if command_index < len(args):
                operation += ":" + args[command_index].rsplit("/", 1)[-1]
        context = (
            f"session={self.session_id} host={self.docker_host} container={container} "
            f"operation={operation} call={uuid.uuid4().hex[:8]} timeout={timeout}"
        )
        started = time.monotonic()
        # Long-running agent/verifier calls are expected; warn near their budget instead.
        slow_after = max(15, timeout * 0.8) if timeout and timeout > 120 else 15
        # A single pending warning also exposes commands with no configured timeout.
        pending = asyncio.get_running_loop().call_later(
            slow_after, logger.warning, "remote docker pending after %.1fs: %s", slow_after, context
        )
        try:
            result = await super()._run_docker("--host", self.docker_host, *args, timeout=timeout)
        except (Exception, asyncio.CancelledError) as exc:
            logger.warning(
                "remote docker failed: %s elapsed=%.2fs error=%s: %s",
                context,
                time.monotonic() - started,
                type(exc).__name__,
                exc,
            )
            raise
        finally:
            pending.cancel()
        elapsed = time.monotonic() - started
        if result.exit_code or elapsed >= slow_after:
            logger.warning(
                "remote docker completed: %s elapsed=%.2fs exit=%s stderr=%r",
                context,
                elapsed,
                result.exit_code,
                result.stderr[-2000:],
            )
        if result.exit_code and ("with_npu_lease" in result.stderr or "--check" in args):
            await self._diagnose_locks(context)
        return result

    async def _diagnose_locks(self, context: str) -> None:
        """Read failure-time state before teardown, without changing locks or retrying work."""
        if not self._container_name:
            return
        commands = (
            ("inspect", "--format", '{"mounts":{{json .Mounts}},"state":{{json .State}}}', self._container_name),
            ("exec", self._container_name, "python3", "-I", "-c", _LOCK_DIAGNOSTICS, self.npu_lock_dir),
        )
        for command in commands:
            try:
                # Bypass the wrapper to avoid recursive diagnostics on failure.
                result = await super()._run_docker("--host", self.docker_host, *command, timeout=10)
                logger.warning(
                    "NPU lock diagnostics: %s probe=%s exit=%s stdout=%r stderr=%r",
                    context,
                    command[0],
                    result.exit_code,
                    result.stdout[:16000],
                    result.stderr[-2000:],
                )
            except Exception as exc:
                logger.warning("NPU lock diagnostics unavailable: %s probe=%s error=%r", context, command[0], exc)


def _copy_sandbox_kwargs(tools_kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    copied = copy.deepcopy(tools_kwargs)
    task_config = copied.get("task")
    if not isinstance(task_config, dict):
        raise ValueError("run_triton_task requires tools_kwargs['task']")
    sandbox_config = task_config.setdefault("sandbox", {})
    if not isinstance(sandbox_config, dict):
        raise TypeError("tools_kwargs['task']['sandbox'] must be a mapping")
    sandbox_kwargs = sandbox_config.setdefault("sandbox_kwargs", {})
    if not isinstance(sandbox_kwargs, dict):
        raise TypeError("tools_kwargs['task']['sandbox']['sandbox_kwargs'] must be a mapping")
    return copied, sandbox_kwargs


def parse_device_ids(value: str) -> tuple[str, ...]:
    raw_devices = tuple(part.strip() for part in value.split(","))
    if not raw_devices or any(not part for part in raw_devices):
        raise ValueError(f"evaluator_npu_device_ids cannot contain empty entries; received {value!r}")
    devices = raw_devices
    if len(set(devices)) != len(devices):
        raise ValueError("evaluator_npu_device_ids cannot contain duplicates")
    if any(not part.replace("-", "").replace("_", "").isalnum() for part in devices):
        raise ValueError("evaluator_npu_device_ids contains an unsafe device ID")
    return devices


def bind_remote_sandbox(
    tools_kwargs: dict[str, Any],
    *,
    hosts: str,
    session_id: str,
    devices: tuple[str, ...],
    lock_dir: str,
    lock_timeout: float,
) -> dict[str, Any]:
    """Copy sample config and bind its Docker host and NPU lease contract."""

    docker_hosts = tuple(part.strip() for part in hosts.split(","))
    if not docker_hosts or any(not part for part in docker_hosts):
        raise ValueError("remote_docker_hosts cannot contain empty entries")
    if not math.isfinite(lock_timeout) or lock_timeout <= 0:
        raise ValueError("evaluator_npu_lock_timeout must be finite and positive")

    slot = int.from_bytes(hashlib.sha256(session_id.encode()).digest()[:8], "big") % len(docker_hosts)
    copied, sandbox_kwargs = _copy_sandbox_kwargs(tools_kwargs)
    sandbox_kwargs["docker_host"] = docker_hosts[slot]
    sandbox_kwargs["session_id"] = session_id
    sandbox_kwargs["npu_lock_dir"] = lock_dir
    env = sandbox_kwargs.setdefault("env", {})
    if not isinstance(env, dict):
        raise TypeError("sandbox_kwargs.env must be a mapping")
    env.update(
        {
            "TRITON_EVAL_DEVICE_IDS": ",".join(devices),
            "TRITON_EVAL_LOCK_DIR": lock_dir,
            "TRITON_EVAL_LOCK_TIMEOUT": str(lock_timeout),
        }
    )
    return copied
