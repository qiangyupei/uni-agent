import pytest

from examples.claude_code_kernel_task.trajectory_processor import process_trajectories
from uni_agent.gateway.session import Trajectory
from uni_agent.tasks import TaskResult


@pytest.mark.parametrize("best_index,limit,expected", [(0, None, [2]), (1, 3, [2]), (None, None, [2, 3, 4])])
def test_best_prefix_and_fallback_preserve_source(best_index, limit, expected):
    trajectory = Trajectory(prompt_ids=[1], response_ids=[2, 3, 4], response_mask=[1, 0, 1], reward_score=0.3)
    extra = {} if best_index is None else {"train_best": {"assistant_index": best_index}}
    selected = process_trajectories(
        (trajectory,), task_result=TaskResult(extra_info=extra), selection="best", max_total_tokens=limit
    )
    assert selected[0].response_ids == expected
    assert selected[0].reward_score == 0.3
    assert len(selected[0].response_mask) == len(expected)
    assert trajectory.response_ids == [2, 3, 4]
    assert trajectory.extra_fields == {}
