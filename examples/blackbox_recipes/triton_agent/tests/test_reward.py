from __future__ import annotations

import pytest

from ..reward import attach_reward, normalize_metrics


def test_partial_correctness_and_speedup_reward() -> None:
    metrics = normalize_metrics(
        None,
        verify={"total_cases": 4, "passed_cases": 2, "failed_cases": 2, "compile_ok": True},
        perf={"speedup_vs_torch": 2.0},
        op_name="vector_add",
    )
    scored = attach_reward(metrics)
    assert scored["pass_rate"] == 0.5
    assert scored["reward_components"]["compile"] == 0.1
    assert scored["reward_components"]["correctness"] == 0.275
    assert scored["reward_components"]["speedup"] == 0.2
    assert scored["reward"] == pytest.approx(0.575)


def test_all_correct_preserves_original_uncapped_reward_shape() -> None:
    metrics = normalize_metrics(
        None,
        verify={"total_cases": 2, "passed_cases": 2, "failed_cases": 0},
        perf={"speedup": 2.0},
    )
    scored = attach_reward(metrics)
    assert scored["correctness_ok"] is True
    assert scored["reward"] == pytest.approx(1.15)


def test_untrusted_boolean_strings_and_non_finite_speedup_do_not_score() -> None:
    metrics = normalize_metrics(
        {"compile_ok": "false", "correctness_ok": "false", "total_cases": 1, "passed_cases": 0},
        perf={"speedup": float("inf")},
    )
    scored = attach_reward(metrics)

    assert scored["compile_ok"] is False
    assert scored["correctness_ok"] is False
    assert scored["reward_components"]["raw_speedup"] == 0.0
    assert scored["reward"] == 0.0
