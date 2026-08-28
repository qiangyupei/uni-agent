from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from .. import runner as runner_module
from ..network import bind_remote_sandbox, parse_device_ids


@dataclass
class SessionHandle:
    session_id: str
    base_url: str | None = None
    reward_info_url: str | None = None


def test_runner_injects_shared_npu_lease_environment() -> None:
    tools = {"task": {"name": "triton_operator", "metadata": {}}}
    copied = bind_remote_sandbox(
        tools,
        hosts="ssh://sandbox-a",
        session_id="session-1",
        devices=("0", "2"),
        lock_dir="/shared/npu-locks",
        lock_timeout=45,
    )

    env = copied["task"]["sandbox"]["sandbox_kwargs"]["env"]
    assert env == {
        "TRITON_EVAL_DEVICE_IDS": "0,2",
        "TRITON_EVAL_LOCK_DIR": "/shared/npu-locks",
        "TRITON_EVAL_LOCK_TIMEOUT": "45",
    }
    assert copied["task"]["sandbox"]["sandbox_kwargs"]["npu_lock_dir"] == "/shared/npu-locks"
    assert "sandbox" not in tools["task"]


def test_remote_docker_binding_is_stable_and_copy_on_write() -> None:
    tools = {"task": {"name": "triton_operator", "metadata": {}}}

    kwargs = {
        "hosts": "ssh://sandbox-a,ssh://sandbox-b",
        "session_id": "session-1",
        "devices": ("0",),
        "lock_dir": "/shared/npu-locks",
        "lock_timeout": 45,
    }
    first = bind_remote_sandbox(tools, **kwargs)
    second = bind_remote_sandbox(tools, **kwargs)

    first_host = first["task"]["sandbox"]["sandbox_kwargs"]["docker_host"]
    assert first_host in {"ssh://sandbox-a", "ssh://sandbox-b"}
    assert second["task"]["sandbox"]["sandbox_kwargs"]["docker_host"] == first_host
    assert "sandbox" not in tools["task"]


@pytest.mark.parametrize("value", ["", ",0", "0,", "0,,1", "0, ,1"])
def test_device_parser_rejects_empty_entries(value: str) -> None:
    with pytest.raises(ValueError, match="empty entries"):
        parse_device_ids(value)


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_npu_lease_rejects_non_positive_or_non_finite_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        bind_remote_sandbox(
            {"task": {"name": "triton_operator"}},
            hosts="ssh://sandbox-a",
            session_id="session-1",
            devices=("0",),
            lock_dir="/shared/npu-locks",
            lock_timeout=timeout,
        )


def test_recipe_runner_requests_fail_closed_reward_post(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    async def fake_run_task(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(runner_module, "run_task", fake_run_task)
    session = SessionHandle(
        "session-1",
        base_url="http://gateway/sessions/session-1/v1",
        reward_info_url="http://gateway/sessions/session-1/reward_info",
    )
    result = asyncio.run(
        runner_module.run_triton_task(
            session=session,
            tools_kwargs={"task": {"name": "triton_operator", "metadata": {}}},
            remote_docker_hosts="ssh://sandbox-a",
            evaluator_npu_device_ids="0,1",
            report_reward=True,
        )
    )

    assert result == "ok"
    assert captured["reward_post_strict"] is True
    assert captured["tools_kwargs"]["task"]["metadata"]["runtime"]["session_id"] == "session-1"
    assert captured["tools_kwargs"]["task"]["sandbox"]["sandbox_kwargs"]["docker_host"] == "ssh://sandbox-a"
    assert captured["tools_kwargs"]["task"]["sandbox"]["sandbox_kwargs"]["npu_lock_dir"] == (
        "/var/lock/triton-agent-npu"
    )
