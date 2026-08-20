"""Claude Code hook: bind a verifier best snapshot to its assistant turn.

The command receives Claude Code's hook JSON on stdin. ``pre`` records the
current transcript assistant count and hashes of the paired best artifacts;
``post`` annotates metrics_best.json only when that pair changed during the
matching verifier invocation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_STATE = ".triton_verify_assistant_pending.json"
_VERIFY_COMMAND = re.compile(r"(?:^|[;&|()\s])(?:bash\s+)?(?:\./)?tools/verify_once\.sh(?:\s|$)")


def _payload() -> dict[str, Any]:
    try:
        value = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_verifier(payload: dict[str, Any]) -> bool:
    if str(payload.get("tool_name", "")).lower() != "bash":
        return False
    tool_input = payload.get("tool_input")
    command = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
    return bool(_VERIFY_COMMAND.search(str(command)))


def _workspace(payload: dict[str, Any]) -> Path:
    cwd = payload.get("cwd") or os.getcwd()
    return Path(str(cwd)).resolve()


def _operator_name(workspace: Path) -> str | None:
    try:
        metadata = json.loads((workspace / "TASK_METADATA.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = metadata.get("op_name") if isinstance(metadata, dict) else None
    return str(value) if value else None


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _assistant_count(transcript_path: Any) -> int:
    if not transcript_path:
        return 0
    seen_ids: set[str] = set()
    count = 0
    try:
        lines = Path(str(transcript_path)).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return 0
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        message_id = str(message.get("id") or "")
        if message_id:
            if message_id in seen_ids:
                continue
            seen_ids.add(message_id)
        count += 1
    return count


def pre(payload: dict[str, Any]) -> None:
    if not _is_verifier(payload):
        return
    workspace = _workspace(payload)
    op_name = _operator_name(workspace)
    if not op_name:
        return
    messages_seen = _assistant_count(payload.get("transcript_path"))
    state = {
        "assistant_messages_seen": messages_seen,
        "assistant_index": messages_seen - 1 if messages_seen > 0 else None,
        "metrics_digest": _digest(workspace / "metrics_best.json"),
        "implementation_digest": _digest(workspace / "src" / f"{op_name}_triton_ascend_impl_best.py"),
    }
    _atomic_json(workspace / _STATE, state)


def post(payload: dict[str, Any]) -> None:
    if not _is_verifier(payload):
        return
    workspace = _workspace(payload)
    state_path = workspace / _STATE
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    try:
        state_path.unlink(missing_ok=True)
    except OSError:
        pass
    op_name = _operator_name(workspace)
    if not op_name or not isinstance(state, dict):
        return
    metrics_path = workspace / "metrics_best.json"
    implementation_path = workspace / "src" / f"{op_name}_triton_ascend_impl_best.py"
    current_metrics = _digest(metrics_path)
    current_impl = _digest(implementation_path)
    pair_exists = current_metrics is not None and current_impl is not None
    pair_changed = current_metrics != state.get("metrics_digest") and current_impl != state.get("implementation_digest")
    assistant_index = state.get("assistant_index")
    if not pair_exists or not pair_changed or not isinstance(assistant_index, int) or assistant_index < 0:
        return
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(metrics, dict):
        return
    metrics.update(
        {
            "assistant_index": assistant_index,
            "assistant_messages_seen": assistant_index + 1,
            "assistant_index_source": "claude_code_transcript_hook",
            "assistant_snapshot_impl_sha256": current_impl,
        }
    )
    _atomic_json(metrics_path, metrics)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    payload = _payload()
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "pre":
        pre(payload)
    elif mode == "post":
        post(payload)


if __name__ == "__main__":
    main()
