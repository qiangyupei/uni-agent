from __future__ import annotations

import json
from pathlib import Path

import pytest

from ..assets.track_verify_snapshot import post, pre


def hook_payload(workspace: Path, transcript: Path) -> dict:
    return {
        "tool_name": "Bash",
        "tool_input": {"command": "bash tools/verify_once.sh smoke"},
        "cwd": str(workspace),
        "transcript_path": str(transcript),
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_hook_annotates_only_a_changed_best_pair(tmp_path: Path) -> None:
    write_json(tmp_path / "TASK_METADATA.json", {"op_name": "smoke"})
    transcript = tmp_path / "transcript.jsonl"
    events = [
        {"type": "assistant", "message": {"id": "a1", "content": []}},
        {"type": "assistant", "message": {"id": "a1", "content": []}},
        {"type": "assistant", "message": {"id": "a2", "content": []}},
    ]
    transcript.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    best_metrics = tmp_path / "metrics_best.json"
    best_impl = tmp_path / "src/smoke_triton_ascend_impl_best.py"
    write_json(best_metrics, {"correctness_ok": True, "speedup": 1.0})
    best_impl.parent.mkdir(parents=True)
    best_impl.write_text("# old\n", encoding="utf-8")
    payload = hook_payload(tmp_path, transcript)

    pre(payload)
    write_json(best_metrics, {"correctness_ok": True, "speedup": 1.2})
    best_impl.write_text("# new\n", encoding="utf-8")
    post(payload)

    annotated = json.loads(best_metrics.read_text(encoding="utf-8"))
    assert annotated["assistant_index"] == 1
    assert annotated["assistant_messages_seen"] == 2
    assert annotated["assistant_index_source"] == "claude_code_transcript_hook"
    assert len(annotated["assistant_snapshot_impl_sha256"]) == 64


def test_hook_does_not_relabel_unchanged_snapshot(tmp_path: Path) -> None:
    write_json(tmp_path / "TASK_METADATA.json", {"op_name": "smoke"})
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"type": "assistant", "message": {"id": "a1", "content": []}}) + "\n",
        encoding="utf-8",
    )
    best_metrics = tmp_path / "metrics_best.json"
    best_impl = tmp_path / "src/smoke_triton_ascend_impl_best.py"
    write_json(best_metrics, {"correctness_ok": True})
    best_impl.parent.mkdir(parents=True)
    best_impl.write_text("# unchanged\n", encoding="utf-8")
    payload = hook_payload(tmp_path, transcript)

    pre(payload)
    post(payload)

    assert "assistant_index" not in json.loads(best_metrics.read_text(encoding="utf-8"))


@pytest.mark.parametrize("change_metrics", [False, True])
def test_hook_does_not_relabel_when_only_half_the_pair_changes(
    tmp_path: Path,
    change_metrics: bool,
) -> None:
    write_json(tmp_path / "TASK_METADATA.json", {"op_name": "smoke"})
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"type": "assistant", "message": {"id": "a1", "content": []}}) + "\n",
        encoding="utf-8",
    )
    best_metrics = tmp_path / "metrics_best.json"
    best_impl = tmp_path / "src/smoke_triton_ascend_impl_best.py"
    write_json(best_metrics, {"speedup": 1.0})
    best_impl.parent.mkdir(parents=True)
    best_impl.write_text("# old\n", encoding="utf-8")
    payload = hook_payload(tmp_path, transcript)

    pre(payload)
    if change_metrics:
        write_json(best_metrics, {"speedup": 1.1})
    else:
        best_impl.write_text("# new\n", encoding="utf-8")
    post(payload)

    assert "assistant_index" not in json.loads(best_metrics.read_text(encoding="utf-8"))
