"""Pure trajectory selection/cropping used by the framework hook PR.

The public entry point intentionally depends only on finalized ``Trajectory``
values and the hook's read-only context. It does not reach back into the
framework, Gateway, tokenizer, or environment variables.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uni_agent.gateway.session import Trajectory

logger = logging.getLogger(__name__)

_STALE_AFTER_CROP = {
    "materialization_reason",
    "min_global_steps",
    "max_global_steps",
}


def process_trajectories(
    trajectories: Sequence[Trajectory],
    *,
    context: Any,
) -> list[Trajectory]:
    """Select trainable assistant-boundary prefixes for one session.

    ``context.options`` supports:

    * ``selection``: ``all_final`` (default), ``best``, or ``final``;
    * ``best_fallback``: policy used when no valid assistant index is reported;
    * ``best_requires_correctness``: opt-in gate (default false preserves partial-credit best hints);
    * ``max_total_tokens``: optional prompt + response cap;
    * ``drop_no_impl`` / ``discard_pre_retry``: verifier-driven filters;
    * ``alignment_error``: ``raise`` (default) or ``drop``;
    * ``empty_policy``: ``drop`` (session failure), ``keep_last``, or ``raise``.

    The function never mutates the source trajectories or the hook context.
    """

    source = list(trajectories)
    if not source:
        return []
    options = _options(context)
    selection = str(options.get("selection", "all_final"))
    best_fallback = str(options.get("best_fallback", "all_final"))
    alignment_error = str(options.get("alignment_error", "raise"))
    empty_policy = str(options.get("empty_policy", "drop"))
    max_total_tokens = _optional_positive_int(options.get("max_total_tokens"), "max_total_tokens")

    _choice(selection, "selection", {"best", "all_final", "final"})
    _choice(best_fallback, "best_fallback", {"all_final", "final", "drop"})
    _choice(alignment_error, "alignment_error", {"raise", "drop"})
    _choice(empty_policy, "empty_policy", {"drop", "keep_last", "raise"})

    reward_info = source[-1].reward_info if isinstance(source[-1].reward_info, dict) else {}
    if _as_bool(options.get("drop_no_impl", True)) and (
        reward_info.get("no_impl_retry_filter") or reward_info.get("no_impl_retry_failed")
    ):
        return _on_empty(source, empty_policy, "no implementation produced")

    indexed = list(enumerate(source))
    if (
        _as_bool(options.get("discard_pre_retry", True))
        and reward_info.get("no_impl_retry_used")
        and reward_info.get("no_impl_retry_discard_previous")
    ):
        indexed = indexed[-1:]

    valid: list[tuple[int, Trajectory]] = []
    for index, trajectory in indexed:
        try:
            _validate_alignment(trajectory)
        except ValueError:
            if alignment_error == "raise":
                raise
            logger.warning("dropping trajectory %s with token-array misalignment", index, exc_info=True)
            continue
        valid.append((index, trajectory))
    if not valid:
        return _on_empty(source, empty_policy, "all trajectories were misaligned")

    if selection == "best":
        # Gateway materializes multiple chains in chain/update order, not as a
        # single chronological assistant-message stream. The stock Claude hook
        # supplies only a scalar assistant index and no chain identity, so it is
        # unsafe to consume that hint across more than one trajectory.
        if len(valid) > 1:
            if isinstance(reward_info.get("train_best"), Mapping):
                logger.warning(
                    "ignoring scalar best-assistant hint for %s finalized trajectories; falling back to %s",
                    len(valid),
                    best_fallback,
                )
            selected = _select_by_policy(valid, best_fallback, max_total_tokens)
        else:
            selected = _select_best(
                valid,
                reward_info,
                max_total_tokens,
                requires_correctness=_as_bool(options.get("best_requires_correctness", False)),
            )
            if selected is None:
                selected = _select_by_policy(valid, best_fallback, max_total_tokens)
    else:
        selected = _select_by_policy(valid, selection, max_total_tokens)

    if not selected:
        return _on_empty(source, empty_policy, "no assistant prefix fits the configured limit")
    return selected


def assistant_spans(trajectory: Trajectory) -> list[tuple[int, int]]:
    """Return half-open model-token runs in response token coordinates."""

    _validate_alignment(trajectory)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(trajectory.response_mask):
        active = bool(int(value))
        if active and start is None:
            start = index
        elif not active and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(trajectory.response_ids)))
    return spans


def crop_to_assistant_prefix(
    trajectory: Trajectory,
    response_end: int,
    *,
    reason: str,
) -> Trajectory:
    """Copy a trajectory through one assistant boundary and clear stale data."""

    _validate_alignment(trajectory)
    spans = assistant_spans(trajectory)
    if response_end not in {end for _start, end in spans}:
        raise ValueError(f"response_end={response_end} is not an assistant boundary")

    actually_cropped = response_end < len(trajectory.response_ids)
    extra_fields = dict(trajectory.extra_fields)
    if actually_cropped:
        for key in _STALE_AFTER_CROP:
            extra_fields.pop(key, None)
    kept_spans = sum(end <= response_end for _start, end in spans)
    extra_fields.update(
        {
            "trajectory_postprocessed": True,
            "trajectory_postprocess_reason": reason,
            "original_response_length": len(trajectory.response_ids),
            "assistant_spans_kept": kept_spans,
            # The Gateway's num_turns counts initial user + assistant/tool-user
            # alternation, plus its terminal offset. For an assistant-ending
            # prefix this is exactly 2*k+1 under the Claude Code protocol.
            "postprocess_num_turns_recomputed": True,
        }
    )
    logprobs = trajectory.response_logprobs
    return replace(
        trajectory,
        response_ids=list(trajectory.response_ids[:response_end]),
        response_mask=list(trajectory.response_mask[:response_end]),
        response_logprobs=None if logprobs is None else list(logprobs[:response_end]),
        num_turns=2 * kept_spans + 1,
        # Gateway routing captures do not expose per-turn slice boundaries. A
        # real response crop cannot prove routing alignment; an unchanged full
        # response can retain its original capture.
        routed_experts=None if actually_cropped else trajectory.routed_experts,
        extra_fields=extra_fields,
    )


def _select_best(
    indexed: list[tuple[int, Trajectory]],
    reward_info: Mapping[str, Any],
    max_total_tokens: int | None,
    *,
    requires_correctness: bool,
) -> list[Trajectory] | None:
    if len(indexed) != 1:
        raise ValueError("a scalar best-assistant hint requires exactly one finalized trajectory")
    metrics = reward_info.get("metrics")
    if requires_correctness and isinstance(metrics, Mapping) and not bool(metrics.get("correctness_ok")):
        return None
    train_best = reward_info.get("train_best")
    if not isinstance(train_best, Mapping):
        return None
    best_index = _as_int_or_none(train_best.get("assistant_index"))
    if best_index is None:
        messages_seen = _as_int_or_none(train_best.get("assistant_messages_seen"))
        best_index = None if messages_seen is None else messages_seen - 1
    if best_index is None or best_index < 0:
        return None

    trajectory = indexed[0][1]
    spans = assistant_spans(trajectory)
    if best_index >= len(spans):
        return None
    legal_ends = [end for _start, end in spans[: best_index + 1] if _within_limit(trajectory, end, max_total_tokens)]
    if not legal_ends:
        return []
    reason = "best_assistant" if legal_ends[-1] == spans[best_index][1] else "best_previous_valid_assistant"
    return [crop_to_assistant_prefix(trajectory, legal_ends[-1], reason=reason)]


def _select_by_policy(
    indexed: list[tuple[int, Trajectory]],
    policy: str,
    max_total_tokens: int | None,
) -> list[Trajectory]:
    if policy == "drop":
        return []
    candidates = indexed[-1:] if policy == "final" else indexed
    selected = []
    for _index, trajectory in candidates:
        cropped = _last_legal_prefix(trajectory, max_total_tokens, reason=policy)
        if cropped is not None:
            selected.append(cropped)
    return selected


def _last_legal_prefix(
    trajectory: Trajectory,
    max_total_tokens: int | None,
    *,
    reason: str,
) -> Trajectory | None:
    legal_ends = [
        end for _start, end in assistant_spans(trajectory) if _within_limit(trajectory, end, max_total_tokens)
    ]
    if not legal_ends:
        return None
    return crop_to_assistant_prefix(trajectory, legal_ends[-1], reason=reason)


def _within_limit(trajectory: Trajectory, response_end: int, limit: int | None) -> bool:
    return limit is None or len(trajectory.prompt_ids) + response_end <= limit


def _validate_alignment(trajectory: Trajectory) -> None:
    response_len = len(trajectory.response_ids)
    if len(trajectory.response_mask) != response_len:
        raise ValueError(f"response_ids/response_mask misaligned: {response_len} != {len(trajectory.response_mask)}")
    if trajectory.response_logprobs is not None and len(trajectory.response_logprobs) != response_len:
        raise ValueError(
            f"response_ids/response_logprobs misaligned: {response_len} != {len(trajectory.response_logprobs)}"
        )


def _on_empty(source: list[Trajectory], policy: str, reason: str) -> list[Trajectory]:
    if policy == "drop":
        logger.warning("trajectory postprocessor returned an empty session: %s", reason)
        return []
    if policy == "raise":
        raise ValueError(f"trajectory postprocessing produced an empty session: {reason}")
    logger.warning("trajectory postprocessor kept the last unmodified trajectory: %s", reason)
    return [source[-1]]


def _options(context: Any) -> Mapping[str, Any]:
    options = context.get("options", {}) if isinstance(context, Mapping) else getattr(context, "options", {})
    if not isinstance(options, Mapping):
        raise TypeError("trajectory postprocessor context.options must be a mapping")
    return options


def _choice(value: str, name: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ValueError(f"invalid {name}={value!r}; expected one of {sorted(allowed)}")


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value in (None, ""):
        return None
    parsed = _as_int_or_none(value)
    if parsed is None or parsed <= 0:
        raise ValueError(f"{name} must be a positive integer or null")
    return parsed


def _as_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)
