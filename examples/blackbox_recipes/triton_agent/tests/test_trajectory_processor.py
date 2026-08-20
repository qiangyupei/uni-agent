from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import pytest

from ..trajectory_processor import crop_to_assistant_prefix, process_trajectories


@dataclass(frozen=True)
class Context:
    options: object


@dataclass
class Trajectory:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float] | None = None
    reward_info: dict[str, Any] = field(default_factory=dict)
    reward_score: float | None = None
    num_turns: int = 0
    routed_experts: Any = None
    multi_modal_data: dict[str, Any] | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)


def trajectory(
    mask: list[int],
    *,
    prompt_len: int = 2,
    reward_info: dict | None = None,
) -> Trajectory:
    response = list(range(10, 10 + len(mask)))
    return Trajectory(
        prompt_ids=list(range(prompt_len)),
        response_ids=response,
        response_mask=mask,
        response_logprobs=[-0.1] * len(mask),
        reward_info=reward_info or {},
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
    second = trajectory([1, 1, 0, 1, 1], reward_info=final_info)

    selected = process_trajectories(
        (first, second),
        context=Context(MappingProxyType({"selection": "best"})),
    )

    assert [len(item.response_ids) for item in selected] == [4, 5]
    assert {item.extra_fields["trajectory_postprocess_reason"] for item in selected} == {"all_final"}


def test_best_crop_uses_previous_legal_assistant_boundary() -> None:
    info = {
        "metrics": {"correctness_ok": True},
        "train_best": {"assistant_index": 1},
    }
    source = trajectory([1, 1, 0, 1, 1], prompt_len=2, reward_info=info)
    selected = process_trajectories(
        [source],
        context={"options": {"selection": "best", "max_total_tokens": 4}},
    )
    assert len(selected) == 1
    assert len(selected[0].response_ids) == 2
    assert selected[0].extra_fields["trajectory_postprocess_reason"] == "best_previous_valid_assistant"


def test_partial_correctness_best_hint_preserves_legacy_selection() -> None:
    info = {
        "metrics": {"correctness_ok": False, "compile_ok": True, "passed_cases": 1},
        "train_best": {"assistant_index": 0},
    }
    source = trajectory([1, 1, 0, 1, 1], reward_info=info)
    selected = process_trajectories([source], context={"options": {"selection": "best"}})

    assert len(selected) == 1
    assert len(selected[0].response_ids) == 2
    assert selected[0].extra_fields["trajectory_postprocess_reason"] == "best_assistant"


def test_no_impl_empty_policy_is_explicit() -> None:
    source = trajectory([1], reward_info={"no_impl_retry_failed": True})
    assert process_trajectories([source], context={"options": {"empty_policy": "drop"}}) == []
    assert process_trajectories([source], context={"options": {"empty_policy": "keep_last"}}) == [source]
    with pytest.raises(ValueError, match="no implementation"):
        process_trajectories([source], context={"options": {"empty_policy": "raise"}})


def test_alignment_error_never_silently_crops_arrays() -> None:
    source = trajectory([1, 1])
    source.response_logprobs = [-0.1]
    with pytest.raises(ValueError, match="response_logprobs"):
        process_trajectories([source], context={"options": {}})
    assert (
        process_trajectories(
            [source],
            context={"options": {"alignment_error": "drop", "empty_policy": "drop"}},
        )
        == []
    )
