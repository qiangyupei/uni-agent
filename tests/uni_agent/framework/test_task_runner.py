import json
import logging
from types import SimpleNamespace

import aiohttp
import pytest

from uni_agent.framework import task_runner
from uni_agent.framework.task_runner import (
    _MAX_TASK_RESULT_EXTRA_INFO_BYTES,
    _REWARD_POST_TIMEOUT_SECONDS,
    _extract_upstream,
    _inject_gateway_tunnel,
    _reward_info_from_result,
    _rewrite_gateway_url,
)
from uni_agent.gateway.session import SessionHandle
from uni_agent.tasks import TaskConfig, TaskResult


def test_rewrite_gateway_url_replaces_host_with_tunnel_port():
    assert _rewrite_gateway_url("http://gateway.example:40169/sessions/abc/v1", 38197) == (
        "http://127.0.0.1:38197/sessions/abc/v1"
    )


def test_rewrite_gateway_url_custom_proxy_port():
    assert _rewrite_gateway_url("http://gateway:8000/v1", 4242) == "http://127.0.0.1:4242/v1"


def test_extract_upstream_returns_host_port():
    assert _extract_upstream("http://gateway.example:40169/sessions/abc/v1") == "gateway.example:40169"


def test_extract_upstream_none_without_port():
    assert _extract_upstream("http://gateway/v1") is None


def test_inject_gateway_tunnel_rewrites_upstream_and_base_url():
    task = {
        "sandbox": {"provider": "openyuanrong", "sandbox_kwargs": {"proxy_port": 38197, "image": "x"}},
        "agent": {"step_limit": 10},
    }
    merged = _inject_gateway_tunnel(task, "http://gateway.example:40169/sessions/abc/v1")

    assert merged["sandbox"]["sandbox_kwargs"]["upstream"] == "gateway.example:40169"
    assert merged["sandbox"]["sandbox_kwargs"]["proxy_port"] == 38197
    # The agent receives the tunnel-rewritten base_url; unrelated keys are preserved.
    assert merged["agent"]["model"]["base_url"] == "http://127.0.0.1:38197/sessions/abc/v1"
    assert merged["agent"]["step_limit"] == 10


def test_inject_gateway_tunnel_raises_without_port():
    task = {"sandbox": {"provider": "openyuanrong", "sandbox_kwargs": {"proxy_port": 38197}}}
    with pytest.raises(ValueError, match="cannot derive gateway tunnel upstream"):
        _inject_gateway_tunnel(task, "http://gateway.example/v1")


def test_inject_gateway_tunnel_rejects_non_yuanrong_sandbox():
    task = {"sandbox": {"provider": "local", "sandbox_kwargs": {"proxy_port": 38197}}}
    with pytest.raises(ValueError, match="supported only on 'openyuanrong'"):
        _inject_gateway_tunnel(task, "http://gateway.example:40169/v1")


def test_task_result_positional_field_order():
    result = TaskResult(0.5, 1.0, False, {"reason": "limit"})

    assert result.reward == 0.5
    assert result.accuracy == 1.0
    assert result.finished is False
    assert result.extra_info == {"reason": "limit"}


def test_reward_info_omits_unknown_agent_completion():
    result = TaskResult(reward=0.5, accuracy=1.0)

    assert _reward_info_from_result(result) == {
        "reward": 0.5,
        "acc": 1.0,
    }


@pytest.mark.parametrize("finished", [True, False])
def test_reward_info_forwards_agent_completion(finished):
    result = TaskResult(reward=0.0, finished=finished)

    assert _reward_info_from_result(result) == {
        "reward": 0.0,
        "finished": finished,
    }


def test_reward_info_rejects_non_boolean_agent_completion():
    result = TaskResult(reward=0.0, finished=0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="finished must be a bool or None"):
        _reward_info_from_result(result)


@pytest.mark.asyncio
async def test_run_task_binds_raw_prompt_to_sample_task_config(monkeypatch, tmp_path):
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(
        """
- name: test_task
""".strip()
    )
    captured = {}

    class _FakeTask:
        def __init__(self, config):
            self.config = TaskConfig(
                name=config["name"],
                sandbox={"provider": "local"},
                prompt=config["prompt"],
                metadata=config["metadata"],
            )

        async def run(self):
            captured["config"] = self.config
            return TaskResult(reward=1.0, accuracy=1.0, finished=True)

    monkeypatch.setattr(task_runner, "get_task", _FakeTask)
    source_prompt = [{"role": "user", "content": "Canonical source problem"}]

    await task_runner.run_task(
        session=SessionHandle(
            session_id="test-session",
            base_url="http://gateway/sessions/test/v1",
            reward_info_url=None,
        ),
        raw_prompt=source_prompt,
        tools_kwargs={
            "task": {
                "name": "test_task",
                "metadata": {"problem_statement": "METADATA PROBLEM"},
            }
        },
        task_config_path=str(config_path),
    )

    assert captured["config"].prompt == source_prompt


@pytest.mark.asyncio
async def test_run_task_forwards_strict_reward_policy(monkeypatch):
    result = TaskResult(reward=0.5)

    class _FakeResolver:
        def resolve(self, sample_config, *, runtime_model):
            return sample_config

    class _FakeTask:
        async def run(self):
            return result

    posted = []

    async def capture_post(url, task_result, *, strict=False):
        posted.append((url, task_result, strict))

    monkeypatch.setattr(task_runner, "TaskConfigResolver", _FakeResolver)
    monkeypatch.setattr(task_runner, "get_task", lambda _: _FakeTask())
    monkeypatch.setattr(task_runner, "_post_reward_info", capture_post)
    session = SimpleNamespace(base_url="http://gateway/v1", reward_info_url="http://gateway/reward_info")
    kwargs = {"session": session, "tools_kwargs": {"task": {"name": "fake"}}}

    await task_runner.run_task(**kwargs, report_reward=False, reward_post_strict=True)
    assert posted == []

    await task_runner.run_task(**kwargs, report_reward=True)
    assert posted == [(session.reward_info_url, result, False)]

    posted.clear()
    await task_runner.run_task(**kwargs, report_reward=True, reward_post_strict=True)
    assert posted == [(session.reward_info_url, result, True)]

    session.reward_info_url = None
    with pytest.raises(RuntimeError, match="strict reward posting requires"):
        await task_runner.run_task(**kwargs, report_reward=True, reward_post_strict=True)


class _FakeClientSession:
    def __init__(self, posts, *, post_error=None, response_error=None):
        self.posts = posts
        self.post_error = post_error
        self.response_error = response_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    def post(self, url, *, json):
        if self.post_error:
            raise self.post_error
        self.posts.append((url, json))
        return self

    def raise_for_status(self):
        if self.response_error:
            raise self.response_error


@pytest.mark.asyncio
async def test_strict_reward_post_succeeds_with_bounded_timeout_and_omitted_invalid_metadata(monkeypatch):
    posts = []
    timeouts = []

    def client_session(*, timeout):
        timeouts.append(timeout.total)
        return _FakeClientSession(posts)

    monkeypatch.setattr(aiohttp, "ClientSession", client_session)
    result = TaskResult(reward=0.5, accuracy=1.0, extra_info={"invalid": object()})

    await task_runner._post_reward_info("http://gateway/reward_info", result, strict=True)
    assert timeouts == [_REWARD_POST_TIMEOUT_SECONDS]
    assert posts == [
        (
            "http://gateway/reward_info",
            {"reward_info": {"reward": 0.5, "acc": 1.0}},
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["post", "response"])
async def test_reward_post_failure_is_best_effort_or_strict(monkeypatch, failure_stage):
    error = RuntimeError(f"{failure_stage} failed")

    def client_session(*, timeout):
        return _FakeClientSession(
            [],
            post_error=error if failure_stage == "post" else None,
            response_error=error if failure_stage == "response" else None,
        )

    monkeypatch.setattr(aiohttp, "ClientSession", client_session)
    result = TaskResult(reward=0.5)

    await task_runner._post_reward_info("http://gateway/reward_info", result)
    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        await task_runner._post_reward_info("http://gateway/reward_info", result, strict=True)


def test_reward_info_forwards_extra_info_without_replacing_reserved_keys(caplog):
    result = TaskResult(
        reward=0.75,
        accuracy=1.0,
        finished=True,
        extra_info={
            "reward": 99,
            "acc": 0,
            "finished": False,
            "train_best": 3,
            "verifier": {"passed": True, "speedup": 1.25},
        },
    )

    with caplog.at_level(logging.WARNING):
        reward_info = _reward_info_from_result(result)

    assert reward_info == {
        "reward": 0.75,
        "acc": 1.0,
        "finished": True,
        "train_best": 3,
        "verifier": {"passed": True, "speedup": 1.25},
    }
    assert "ignoring reserved TaskResult.extra_info keys: acc, finished, reward" in caplog.text


@pytest.mark.parametrize("extra_info", [{"invalid": object()}, {"not_finite": float("nan")}])
def test_reward_info_omits_invalid_extra_info(extra_info, caplog):
    result = TaskResult(reward=0.5, extra_info=extra_info)

    with caplog.at_level(logging.WARNING):
        assert _reward_info_from_result(result) == {"reward": 0.5}
    assert "metadata must be JSON serializable" in caplog.text


def test_reward_info_handles_json_recursion_error(monkeypatch):
    def raise_recursion_error(*args, **kwargs):
        raise RecursionError

    monkeypatch.setattr(task_runner.json, "dumps", raise_recursion_error)
    result = TaskResult(reward=0.5, extra_info={"nested": {}})

    assert _reward_info_from_result(result) == {"reward": 0.5}


def test_reward_info_enforces_serialized_size_limit():
    oversized = TaskResult(reward=0.5, extra_info={"text": "x" * _MAX_TASK_RESULT_EXTRA_INFO_BYTES})
    assert _reward_info_from_result(oversized) == {"reward": 0.5}

    empty_payload_size = len(json.dumps({"text": ""}).encode("utf-8"))
    at_limit = {"text": "x" * (_MAX_TASK_RESULT_EXTRA_INFO_BYTES - empty_payload_size)}
    assert _reward_info_from_result(TaskResult(reward=0.5, extra_info=at_limit)) == {
        "reward": 0.5,
        **at_limit,
    }
