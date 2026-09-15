import json

import pytest

from examples.claude_code_kernel_task.framework import KernelAgentFramework
from examples.claude_code_kernel_task.trajectory_processor import process_trajectories
from uni_agent.gateway.session import Trajectory
from uni_agent.tasks import TaskResult


@pytest.mark.parametrize(
    "best_index,limit,applied", [(0, None, "best_prefix"), (1, 3, "best_prefix"), (None, None, "fallback")]
)
def test_diagnostics_reach_trajectory_json(tmp_path, best_index, limit, applied):
    trajectory = Trajectory(prompt_ids=[1], response_ids=[2, 3, 4], response_mask=[1, 0, 1], reward_score=0.3)
    extra = {
        "agent": {
            "verify_progress": {
                "verify_count": 8,
                "correctness_stale_verify_count": 7,
                "latency_stale_verify_count": 0,
            },
            "early_stop": {"reason": "no_verify_improvement"},
        },
    }
    if best_index is not None:
        extra["train_best"] = {"assistant_index": best_index}
    selected = process_trajectories(
        (trajectory,), task_result=TaskResult(extra_info=extra), selection="best", max_total_tokens=limit
    )
    framework = object.__new__(KernelAgentFramework)
    framework._dump_trajectories(tmp_path, "session", selected)
    meta = json.loads((tmp_path / "trajectory.json").read_text())["trajectories"][0]
    diagnostic = meta["kernel_bench"]
    assert diagnostic["best_assistant_index"] == best_index
    assert diagnostic["selection_applied"] == applied
    assert diagnostic["selected_assistant_index"] == (1 if best_index is None else 0)
    assert diagnostic["verify_count"] == 8
    assert diagnostic["correctness_stale_verify_count"] == 7
    assert diagnostic["latency_stale_verify_count"] == 0
    assert diagnostic["early_stop_reason"] == "no_verify_improvement"
    assert meta["reward_score"] == 0.3
    assert trajectory.extra_fields == {}


def test_missing_hook_state_is_unknown():
    trajectory = Trajectory(prompt_ids=[1], response_ids=[2], response_mask=[1])
    selected = process_trajectories((trajectory,), task_result=TaskResult(), selection="best")
    meta = object.__new__(KernelAgentFramework)._trajectory_meta(selected[0])["kernel_bench"]
    assert meta["selection_applied"] == "fallback"
    assert meta["verify_count"] is None
    assert meta["early_stop_reason"] is None
