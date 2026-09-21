import asyncio
from types import SimpleNamespace

import pytest

from examples.claude_code_kernel_task import runner

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.mark.parametrize(
    "split,overrides,expected",
    [
        ("train", {}, {"max_turns": 100, "run_timeout": 7200}),
        ("validation", {}, {"max_turns": 120, "run_timeout": 10800}),
        (
            "validation",
            {"validation_max_turns": 80, "validation_run_timeout": 9000},
            {"max_turns": 80, "run_timeout": 9000},
        ),
    ],
)
def test_split_budgets(monkeypatch, split, overrides, expected):
    original = {"task": {"metadata": {"split": split}, "agent": {"max_turns": 100, "run_timeout": 7200}}}
    extra_env = {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096"}
    original["task"]["agent"]["extra_env"] = extra_env

    async def run_task(**kwargs):
        return kwargs["tools_kwargs"]["task"]["agent"]

    monkeypatch.setattr(runner, "run_task", run_task)
    result = asyncio.run(
        runner.run_triton_task(
            session=SimpleNamespace(session_id="test"),
            tools_kwargs=original,
            remote_docker_hosts="ssh://test",
            evaluator_npu_device_ids="0",
            **overrides,
        )
    )
    assert result == {**expected, "extra_env": extra_env}
    assert original["task"]["agent"] == {"max_turns": 100, "run_timeout": 7200, "extra_env": extra_env}
