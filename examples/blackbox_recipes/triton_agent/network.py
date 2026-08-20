"""Provider-specific, copy-on-write Gateway tunnel binding."""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from uni_agent.gateway.session import SessionHandle


def bind_gateway_tunnel(
    session: SessionHandle,
    tools_kwargs: dict[str, Any],
    *,
    proxy_port: int,
) -> tuple[SessionHandle, dict[str, Any]]:
    """Return copies configured for an OpenYuanRong per-sandbox tunnel.

    ``reward_info_url`` is deliberately left unchanged: reward reporting runs
    in the Ray task process, whereas only Claude Code runs inside the sandbox.
    """

    if not session.base_url:
        raise ValueError("sandbox_gateway_tunnel requires session.base_url")
    if not 1 <= proxy_port <= 65535:
        raise ValueError(f"invalid sandbox gateway proxy port: {proxy_port}")

    parsed = urlsplit(session.base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.port is None:
        raise ValueError(f"Gateway URL must contain an http(s) host and port: {session.base_url!r}")
    if parsed.username or parsed.password:
        raise ValueError("Gateway URL credentials cannot be forwarded as a sandbox upstream")

    upstream_host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    upstream = f"{upstream_host}:{parsed.port}"
    internal_url = urlunsplit(("http", f"127.0.0.1:{proxy_port}", parsed.path, parsed.query, ""))

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
    sandbox_kwargs["upstream"] = upstream
    sandbox_kwargs["proxy_port"] = proxy_port

    return replace(session, base_url=internal_url), copied


def parse_device_ids(value: str) -> tuple[str, ...]:
    raw_devices = tuple(part.strip() for part in value.split(","))
    if not raw_devices or any(not part for part in raw_devices):
        raise ValueError("evaluator_npu_device_ids cannot contain empty entries")
    devices = raw_devices
    if len(set(devices)) != len(devices):
        raise ValueError("evaluator_npu_device_ids cannot contain duplicates")
    if any(not part.replace("-", "").replace("_", "").isalnum() for part in devices):
        raise ValueError("evaluator_npu_device_ids contains an unsafe device ID")
    return devices


def bind_npu_lease(
    tools_kwargs: dict[str, Any],
    *,
    device_ids: str,
    lock_dir: str,
    lock_timeout: float,
) -> dict[str, Any]:
    """Copy sample config and inject the image-owned evaluator lease contract."""

    devices = parse_device_ids(device_ids)
    if not math.isfinite(lock_timeout) or lock_timeout <= 0:
        raise ValueError("evaluator_npu_lock_timeout must be finite and positive")
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
