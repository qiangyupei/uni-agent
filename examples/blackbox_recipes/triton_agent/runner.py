"""Thin adapter between Uni-Agent's generic Task runner and this example.

The adapter has two intentionally small responsibilities:

* import :mod:`task` so the example-local task is registered; and
* bind this session to a remote Docker host.

Task construction, reward reporting, and ``TaskResult.extra_info`` validation
remain owned by :func:`uni_agent.framework.task_runner.run_task`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from uni_agent.framework.task_runner import run_task

if TYPE_CHECKING:
    from uni_agent.gateway.session import SessionHandle

from . import remote_docker as _remote_docker  # noqa: F401 - registers the recipe provider
from . import task as _task  # noqa: F401 - import registers ``triton_operator``
from .network import bind_remote_sandbox, parse_device_ids


async def run_triton_task(
    *,
    session: SessionHandle,
    tools_kwargs: dict[str, Any] | None = None,
    raw_prompt: Any = None,
    sample_index: int | None = None,
    remote_docker_hosts: str,
    evaluator_npu_device_ids: str,
    evaluator_npu_lock_dir: str = "/var/lock/triton-agent-npu",
    evaluator_npu_lock_timeout: float = 1200.0,
    reward_post_strict: bool = True,
    **kwargs: Any,
):
    """Run one Triton task through Uni-Agent's generic Task runner."""

    devices = parse_device_ids(evaluator_npu_device_ids)
    copied = bind_remote_sandbox(
        tools_kwargs or {},
        hosts=remote_docker_hosts,
        session_id=session.session_id,
        devices=devices,
        lock_dir=evaluator_npu_lock_dir,
        lock_timeout=evaluator_npu_lock_timeout,
    )
    task_config = copied.get("task")
    if not isinstance(task_config, dict):
        raise ValueError("run_triton_task requires tools_kwargs['task']")
    metadata = task_config.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("tools_kwargs['task']['metadata'] must be a mapping")
    runtime = metadata.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        raise TypeError("tools_kwargs['task']['metadata']['runtime'] must be a mapping")
    runtime.update({"session_id": session.session_id, "sample_index": sample_index})

    runtime["evaluator_npu_device_count"] = len(devices)

    return await run_task(
        session=session,
        tools_kwargs=copied,
        raw_prompt=raw_prompt,
        sample_index=sample_index,
        reward_post_strict=reward_post_strict,
        **kwargs,
    )
