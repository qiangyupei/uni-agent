from __future__ import annotations

import pytest

from uni_agent.gateway.session import Trajectory
from uni_agent.tasks import TaskResult

from ..trajectory_processor import crop_to_assistant_prefix, process_trajectories


def trajectory(
    mask: list[int],
    *,
    prompt_len: int = 2,
) -> Trajectory:
    response = list(range(10, 10 + len(mask)))
    return Trajectory(
        prompt_ids=list(range(prompt_len)),
        response_ids=response,
        response_mask=mask,
        response_logprobs=[-0.1] * len(mask),
        num_turns=99,
        routed_experts="stale-routing",  # type: ignore[arg-type]
        extra_fields={"min_global_steps": 3, "max_global_steps": 4, "kept": "value"},
    )


def test_crop_is_aligned_and_clears_stale_metadata() -> None:
    source = trajectory([1, 1, 0, 1, 1])
    cropped = crop_to_assistant_prefix(source, 2, reason="test")

    assert cropped.response_ids == source.response_ids[:2]
    assert cropped.response_mask == [1, 1]
    assert cropped.response_logprobs == [-0.1, -0.1]
    assert cropped.num_turns == 3
    assert cropped.routed_experts is None
    assert cropped.extra_fields["kept"] == "value"
    assert "min_global_steps" not in cropped.extra_fields
    assert "max_global_steps" not in cropped.extra_fields
    assert source.response_ids == [10, 11, 12, 13, 14]


def test_best_index_is_not_applied_across_unordered_gateway_chains() -> None:
    first = trajectory([1, 1, 0, 1])
    final_info = {
        "metrics": {"correctness_ok": True},
        "train_best": {"assistant_index": 0, "source": "metrics_best"},
    }
    second = trajectory([1, 1, 0, 1, 1])

    selected = process_trajectories(
        (first, second),
        task_result=TaskResult(extra_info=final_info),
        selection="best",
    )

    assert [len(item.response_ids) for item in selected] == [4, 5]
    assert {item.extra_fields["trajectory_postprocess_reason"] for item in selected} == {"all_final"}


def test_best_crop_uses_previous_legal_assistant_boundary() -> None:
    info = {
        "metrics": {"correctness_ok": True},
        "train_best": {"assistant_index": 1},
    }
    source = trajectory([1, 1, 0, 1, 1], prompt_len=2)
    selected = process_trajectories(
        (source,),
        task_result=TaskResult(extra_info=info),
        selection="best",
        max_total_tokens=4,
    )
    assert len(selected) == 1
    assert len(selected[0].response_ids) == 2
    assert selected[0].extra_fields["trajectory_postprocess_reason"] == "best_previous_valid_assistant"


def test_partial_correctness_best_hint_preserves_legacy_selection() -> None:
    info = {
        "metrics": {"correctness_ok": False, "compile_ok": True, "passed_cases": 1},
        "train_best": {"assistant_index": 0},
    }
    source = trajectory([1, 1, 0, 1, 1])
    selected = process_trajectories((source,), task_result=TaskResult(extra_info=info), selection="best")

    assert len(selected) == 1
    assert len(selected[0].response_ids) == 2
    assert selected[0].extra_fields["trajectory_postprocess_reason"] == "best_assistant"


def test_no_impl_empty_policy_is_explicit() -> None:
    source = trajectory([1])
    assert (
        process_trajectories(
            (source,), task_result=TaskResult(extra_info={"no_impl_retry_failed": True}), empty_policy="drop"
        )
        == []
    )
    assert process_trajectories(
        (source,), task_result=TaskResult(extra_info={"no_impl_retry_failed": True}), empty_policy="keep_last"
    ) == [source]
    with pytest.raises(ValueError, match="no implementation"):
        process_trajectories(
            (source,), task_result=TaskResult(extra_info={"no_impl_retry_failed": True}), empty_policy="raise"
        )


def test_alignment_error_never_silently_crops_arrays() -> None:
    source = trajectory([1, 1])
    source.response_logprobs = [-0.1]
    with pytest.raises(ValueError, match="response_logprobs"):
        process_trajectories((source,), task_result=TaskResult())
    assert (
        process_trajectories(
            (source,),
            task_result=TaskResult(),
            alignment_error="drop",
            empty_policy="drop",
        )
        == []
    )


@pytest.mark.asyncio
async def test_framework_passes_task_result_to_best_prefix_processor() -> None:
    from types import SimpleNamespace

    from uni_agent.framework.framework import GatewayAgentFramework

    source = trajectory([1, 1, 0, 1, 1])
    source.finished = True
    source.reward_score = 0.5
    source.reward_metrics = {"acc": 0.25}
    result = TaskResult(reward=0.5, accuracy=0.25, finished=True, extra_info={"train_best": {"assistant_index": 0}})
    framework = SimpleNamespace(
        _trajectory_postprocessor=process_trajectories,
        _trajectory_postprocessor_kwargs={"selection": "best"},
    )
    selected = await GatewayAgentFramework._apply_trajectory_postprocessor(framework, [source], result)
    assert selected[0].response_ids == source.response_ids[:2]
    assert (selected[0].finished, selected[0].reward_score, selected[0].reward_metrics) == (True, 0.5, {"acc": 0.25})
    assert len(source.response_ids) == 5
    assert result.extra_info == {"train_best": {"assistant_index": 0}}
