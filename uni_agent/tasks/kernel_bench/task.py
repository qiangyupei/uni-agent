"""KernelBench task using Uni-Agent Task, Sandbox, and Claude Code APIs."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import math
import re
import shlex
import time
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import Field, field_validator

from uni_agent.sandbox import ExecResult, Sandbox
from uni_agent.tasks import Task, TaskConfig, TaskResult
from uni_agent.tasks.registry import register_task

from .reward import attach_reward, normalize_metrics

logger = logging.getLogger(__name__)

_SAFE_OPERATOR = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_ARTIFACT_PATHS = (
    "metrics.json",
    "metrics_best.json",
    "output/verify/verify_result.json",
    "output/verify/verify_result_summary.json",
    "output/verify/perf_result.json",
)
_MAX_JSON_BYTES = 1024 * 1024
_MAX_IMPLEMENTATION_BYTES = 2 * 1024 * 1024
_FILE_READ_TIMEOUT = 30.0
_EARLY_STOP_PATH = ".triton_verify_stop.json"


class TritonOperatorTaskConfig(TaskConfig):
    name: str = "triton_operator"
    workspace_dir: str = Field(default="/workspace", description="Fresh per-session workspace inside the sandbox.")
    template_dir: str | None = Field(
        default="/opt/triton-agent-template",
        description="Root-owned task/verifier template baked into the sandbox image; None stages only sample files.",
    )
    npu_lease_command: str = "/opt/triton-agent-tools/with_npu_lease.py"
    npu_lease_required: bool = True
    agent_verify_entrypoint: str = "tools/verify_once.sh"
    trusted_agent_verify_command: str = "/opt/triton-agent-tools/verify_once.sh"
    enforce_trusted_image: bool = True
    trusted_tool_paths: tuple[str, ...] = (
        "/opt/triton-agent-tools/with_npu_lease.py",
        "/opt/triton-agent-tools/verify_once.sh",
        "/opt/triton-agent-tools/cleanup_task_processes.sh",
    )
    setup_timeout: float = 120.0
    cleanup_timeout: float = 30.0
    cleanup_command: str | None = Field(
        default="/opt/triton-agent-tools/cleanup_task_processes.sh",
        description=("Optional image-owned process cleanup command; sandbox.stop remains the mandatory final cleanup."),
    )
    artifact_dir: str | None = Field(
        default=None,
        description="Optional runner-local destination, disabled by default.",
    )
    artifact_timeout: float = 120.0
    max_artifact_bytes: int = 2 * 1024 * 1024
    reward_weights: dict[str, float] = Field(default_factory=dict)
    retry_on_missing_impl: bool = False
    install_transcript_hooks: bool = True
    verify_early_stop_patience: int = Field(default=7, ge=0)
    verify_early_stop_min_reward: float = Field(default=0.15, ge=0)
    latency_optimize_patience: int = Field(default=3, ge=0)

    @field_validator("workspace_dir")
    @classmethod
    def _safe_workspace(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or str(path) in {"/", "/root", "/home", "/workspace/.."}:
            raise ValueError("workspace_dir must be a dedicated absolute sandbox path")
        if ".." in path.parts:
            raise ValueError("workspace_dir cannot contain '..'")
        return str(path)


@register_task("triton_operator")
class TritonOperatorTask(Task):
    """Stage one task, run stock Claude Code, select its metrics, then destroy its sandbox.

    A missing-implementation retry gets a *new* context-managed sandbox while
    retaining the same Gateway session. This produces a new chain without any
    private session reset and proves workspace/process isolation by construction.
    """

    config_model = TritonOperatorTaskConfig

    async def run(self) -> TaskResult:
        cfg: TritonOperatorTaskConfig = self.config  # type: ignore[assignment]
        metadata = dict(cfg.metadata)
        op_name = _operator_name(metadata)
        started = time.monotonic()
        agent_info: dict[str, Any] = {}
        evaluation: dict[str, Any] = {}
        finished: bool | None = False
        attempts: list[dict[str, Any]] = []
        attempt_limit = 2 if cfg.retry_on_missing_impl else 1
        for attempt_index in range(attempt_limit):
            evaluation, agent_info, finished, timing = await self._run_attempt(
                cfg,
                metadata,
                op_name,
                attempt_index=attempt_index,
            )
            attempts.append({"attempt": attempt_index + 1, **timing, "has_impl": evaluation.get("_has_impl", False)})
            if evaluation.get("_has_impl"):
                break
            if attempt_index + 1 < attempt_limit:
                logger.info("%s produced no implementation; retrying in a fresh sandbox", op_name)

        retried = len(attempts) > 1
        evaluation.pop("_has_impl", None)
        if retried:
            evaluation.update(
                {
                    "no_impl_retry_used": True,
                    "no_impl_retry_attempts": len(attempts),
                    "no_impl_retry_reason": "missing_impl",
                    "no_impl_retry_discard_previous": True,
                }
            )
        evaluation["timing_ms"] = {"total": _elapsed_ms(started), "attempts": attempts}
        agent_info["attempt_count"] = len(attempts)

        metrics = evaluation.get("metrics") if isinstance(evaluation.get("metrics"), dict) else {}
        reward = float(metrics.get("reward", 0.0))
        pass_rate = float(metrics.get("pass_rate", 0.0))
        extra_info = _compact_extra_info(op_name, evaluation, agent_info, metadata=metadata)
        logger.info(
            "triton task done: op=%s reward=%.4f pass_rate=%.4f source=%s total_ms=%s",
            op_name,
            reward,
            pass_rate,
            extra_info.get("selected_metrics_source"),
            extra_info.get("timing_ms", {}).get("total"),
        )
        return TaskResult(reward=reward, accuracy=pass_rate, finished=finished, extra_info=extra_info)

    async def _run_attempt(
        self,
        cfg: TritonOperatorTaskConfig,
        metadata: dict[str, Any],
        op_name: str,
        *,
        attempt_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any], bool | None, dict[str, Any]]:
        attempt_started = time.monotonic()
        agent_info: dict[str, Any]
        finished: bool | None
        # __aexit__ executes for each attempt on success, error, and
        # cancellation. A retry cannot inherit workspace files or processes.
        async with self.build_sandbox() as sandbox:
            workspace = cfg.workspace_dir
            try:
                setup_started = time.monotonic()
                await self._prepare_workspace(sandbox, cfg, metadata, op_name)
                setup_ms = _elapsed_ms(setup_started)
                initial_impl_digests = await _implementation_digests(sandbox, workspace, op_name)

                agent_started = time.monotonic()
                try:
                    agent_result = await self.build_agent().run(sandbox=sandbox, messages=cfg.prompt)
                    finished = agent_result.finished
                    agent_info = {
                        "exit_code": agent_result.info.get("exit_code"),
                        "finished": agent_result.finished,
                    }
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # preserve partial workspace and agent-time verifier artifacts
                    logger.exception("Claude Code failed for %s; evaluating partial workspace", op_name)
                    agent_info = {"finished": False, "error_type": type(exc).__name__, "error": str(exc)[:512]}
                    finished = False
                agent_ms = _elapsed_ms(agent_started)

                # Stop agent-time verifier children before reading their snapshots.
                # This also releases any evaluator NPU lease.
                await self._cleanup_task_processes(sandbox, cfg, workspace, required=True)
                early_stop = await _read_early_stop(sandbox, workspace)
                evaluate_started = time.monotonic()
                evaluation = await self._evaluate_workspace(
                    sandbox,
                    cfg,
                    workspace,
                    op_name,
                    initial_impl_digests=initial_impl_digests,
                )
                if early_stop is not None:
                    agent_info["early_stop"] = early_stop
                    if evaluation.get("train_best"):
                        finished = True
                        agent_info["finished"] = True
                evaluate_ms = _elapsed_ms(evaluate_started)
                await self._collect_artifacts(
                    sandbox,
                    cfg,
                    metadata,
                    workspace,
                    op_name,
                    attempt_index=attempt_index,
                )
            finally:
                # Provider stop is authoritative. An image may additionally own
                # a bounded PID/cgroup cleanup script for accelerator processes.
                try:
                    await self._cleanup_task_processes(sandbox, cfg, workspace, required=False)
                except Exception:  # noqa: BLE001 - best effort before mandatory sandbox.stop
                    logger.warning("task process cleanup failed for %s", op_name, exc_info=True)
        timing = {
            "setup": setup_ms,
            "agent": agent_ms,
            "evaluate": evaluate_ms,
            "total": _elapsed_ms(attempt_started),
        }
        return evaluation, agent_info, finished, timing

    async def _cleanup_task_processes(
        self,
        sandbox: Sandbox,
        cfg: TritonOperatorTaskConfig,
        workspace: str,
        *,
        required: bool,
    ) -> None:
        if not cfg.cleanup_command:
            if required:
                raise RuntimeError("cleanup_command is required before reading agent-time verifier artifacts")
            return
        result = await asyncio.shield(
            sandbox.exec_shell(
                cfg.cleanup_command,
                timeout=cfg.cleanup_timeout,
                workdir=workspace,
            )
        )
        if required:
            _require_success(result, "pre-evaluation process cleanup")

    async def _prepare_workspace(
        self,
        sandbox: Sandbox,
        cfg: TritonOperatorTaskConfig,
        metadata: dict[str, Any],
        op_name: str,
    ) -> None:
        workspace = shlex.quote(cfg.workspace_dir)
        commands = [f"mkdir -p {workspace}/src {workspace}/output/verify {workspace}/.triton_case_sidecars"]
        if cfg.template_dir:
            template = shlex.quote(cfg.template_dir)
            commands.insert(0, f"test -d {template}")
            commands.append(f"cp -a --no-preserve=ownership,mode {template}/. {workspace}/")
        result = await sandbox.exec_shell(" && ".join(commands), timeout=cfg.setup_timeout)
        _require_success(result, "workspace setup")
        # Stock ClaudeCodeAgent intentionally uses the sandbox provider's
        # process cwd. Fail before rollout if the image/provider contract would
        # make Claude inspect or edit a different directory.
        pwd = await sandbox.exec(["pwd"], timeout=10)
        _require_success(pwd, "sandbox working-directory check")
        if pwd.stdout.strip().rstrip("/") != cfg.workspace_dir.rstrip("/"):
            raise RuntimeError(
                "sandbox process cwd must equal workspace_dir for stock ClaudeCodeAgent: "
                f"expected {cfg.workspace_dir!r}, got {pwd.stdout.strip()!r}"
            )
        if cfg.enforce_trusted_image:
            await _validate_trusted_image(sandbox, cfg.trusted_tool_paths)
            await _validate_agent_verify_entrypoint(
                sandbox,
                cfg.workspace_dir,
                cfg.agent_verify_entrypoint,
                cfg.trusted_agent_verify_command,
            )
        if cfg.npu_lease_required:
            lease_check = await sandbox.exec([cfg.npu_lease_command, "--check"], timeout=10)
            _require_success(lease_check, "evaluator NPU lease contract check")

        task_code = metadata.get("task_code")
        if not isinstance(task_code, str) or not task_code.strip():
            raise ValueError("task metadata requires non-empty task_code")
        await sandbox.write_file(f"{cfg.workspace_dir}/src/{op_name}.py", task_code)

        support_files = metadata.get("support_files", [])
        for filename, content in _support_file_entries(support_files):
            safe_name = _support_filename(filename)
            if not isinstance(content, str | bytes):
                raise TypeError(f"support file {safe_name!r} must contain text or bytes")
            await sandbox.write_file(f"{cfg.workspace_dir}/src/{safe_name}", content)
            protected_copy = f"{cfg.workspace_dir}/.triton_case_sidecars/{safe_name}"
            await sandbox.write_file(protected_copy, content)
            protected = await sandbox.exec(
                ["chmod", "0444", f"{cfg.workspace_dir}/src/{safe_name}", protected_copy],
                timeout=10,
            )
            _require_success(protected, f"support file protection for {safe_name}")

        public_metadata = _public_metadata(metadata)
        await sandbox.write_file(
            f"{cfg.workspace_dir}/TASK_METADATA.json",
            json.dumps(public_metadata, ensure_ascii=False, indent=2) + "\n",
        )
        if cfg.install_transcript_hooks:
            await self._install_transcript_hooks(sandbox, cfg)

    async def _install_transcript_hooks(self, sandbox: Sandbox, cfg: TritonOperatorTaskConfig) -> None:
        workspace = cfg.workspace_dir
        hook_source = (Path(__file__).with_name("assets") / "track_verify_snapshot.py").read_bytes()
        hook_path = f"{workspace}/.claude/hooks/track_verify_snapshot.py"
        policy_path = f"{workspace}/.claude/hooks/triton_verify_policy.json"
        created = await sandbox.exec(["mkdir", "-p", f"{workspace}/.claude/hooks"], timeout=10)
        _require_success(created, "Claude transcript hook directory setup")
        await sandbox.write_file(hook_path, hook_source)
        await sandbox.write_file(
            policy_path,
            json.dumps(
                {
                    "correctness_patience": cfg.verify_early_stop_patience,
                    "correctness_min_reward": cfg.verify_early_stop_min_reward,
                    "latency_patience": cfg.latency_optimize_patience,
                },
                separators=(",", ":"),
            )
            + "\n",
        )
        settings_path = f"{workspace}/.claude/settings.json"
        settings = await _read_json(sandbox, settings_path) or {}
        hooks = settings.setdefault("hooks", {})
        if not isinstance(hooks, dict):
            raise TypeError(".claude/settings.json 'hooks' must be a mapping")
        quoted_hook = shlex.quote(hook_path)
        for event, mode in (
            ("PreToolUse", "pre"),
            ("PostToolUse", "post"),
            ("PostToolUseFailure", "post"),
        ):
            entries = hooks.setdefault(event, [])
            if not isinstance(entries, list):
                raise TypeError(f".claude/settings.json hooks.{event} must be a list")
            command = f"python3 {quoted_hook} {mode}"
            if not any(command in json.dumps(entry, sort_keys=True) for entry in entries):
                entries.append(
                    {
                        "matcher": "Bash",
                        "hooks": [{"type": "command", "command": command, "timeout": 30}],
                    }
                )
        await sandbox.write_file(settings_path, json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
        result = await sandbox.exec(["chmod", "0444", hook_path, policy_path, settings_path], timeout=10)
        _require_success(result, "Claude transcript hook protection")

    async def _evaluate_workspace(
        self,
        sandbox: Sandbox,
        cfg: TritonOperatorTaskConfig,
        workspace: str,
        op_name: str,
        *,
        initial_impl_digests: dict[str, str | None],
    ) -> dict[str, Any]:
        """Reuse the same agent-time metric selection order as the legacy recipe."""

        if not await _plain_directory(sandbox, workspace) or not await _plain_directory(sandbox, f"{workspace}/src"):
            raise RuntimeError("workspace and src must be real directories before metric evaluation")
        best_impl_path = f"{workspace}/src/{op_name}_triton_ascend_impl_best.py"
        current_impl_path = f"{workspace}/src/{op_name}_triton_ascend_impl.py"
        staged_impl_path = f"{workspace}/output/verify/{op_name}_triton_ascend_impl.py"
        best_status = await _implementation_status(
            sandbox,
            best_impl_path,
            baseline_digest=initial_impl_digests.get("best"),
        )
        current_status = await _implementation_status(
            sandbox,
            current_impl_path,
            baseline_digest=initial_impl_digests.get("current"),
        )
        artifact_dirs_safe = await _plain_directory(sandbox, f"{workspace}/output") and await _plain_directory(
            sandbox, f"{workspace}/output/verify"
        )
        staged_status = (
            await _implementation_status(sandbox, staged_impl_path, baseline_digest=None)
            if artifact_dirs_safe
            else {"substantive": False, "digest": None, "reason": "unsafe_artifact_directory"}
        )

        def finish(
            raw_metrics: dict[str, Any],
            *,
            source: str,
            implementation_path: str,
            implementation_digest: str,
            used_best: bool,
            binding: str,
            best_source: str | None = None,
        ) -> dict[str, Any]:
            canonical = _canonical_agent_metric_snapshot(raw_metrics, op_name=op_name)
            if canonical is None:
                raise ValueError(f"invalid agent-time metrics selected from {source}")
            # Artifact recovery follows verify_once's validated staged source.
            # Paired legacy files retain their explicit AST result instead of
            # silently turning a recorded failure into partial reward.
            canonical["ast_check_ok"] = (
                True if best_source == "verifier_artifacts" else raw_metrics.get("ast_check_ok") is True
            )
            metrics = normalize_metrics(canonical, op_name=op_name)
            metrics = attach_reward(metrics, cfg.reward_weights)
            metrics.update(
                {
                    "agent_time_metrics_reused": True,
                    "metrics_impl_path": implementation_path,
                    "metrics_impl_sha256": implementation_digest,
                    "metrics_impl_binding": binding,
                }
            )
            evaluation: dict[str, Any] = {
                "metrics": metrics,
                "selected_metrics_source": source,
                "used_best_metrics": used_best,
                "_has_impl": True,
            }
            if best_source is not None:
                evaluation["best_source"] = best_source
            if used_best:
                train_best = _train_best(
                    raw_metrics,
                    source,
                    best_impl_digest=implementation_digest,
                )
                if train_best:
                    evaluation["train_best"] = train_best
            return evaluation

        # Legacy priority 1: a paired best implementation and metrics snapshot.
        best = await _read_json(sandbox, f"{workspace}/metrics_best.json")
        best_digest = best_status.get("digest") if best_status["substantive"] else None
        if isinstance(best, dict) and best_digest and _has_verify_signal(best):
            binding = _metric_impl_binding(best, best_digest)
            if binding is not None and _canonical_agent_metric_snapshot(best, op_name=op_name) is not None:
                return finish(
                    best,
                    source="metrics_best",
                    implementation_path=f"src/{op_name}_triton_ascend_impl_best.py",
                    implementation_digest=best_digest,
                    used_best=True,
                    binding=binding,
                    best_source="best_pair",
                )
            logger.warning("ignoring metrics_best with a mismatched implementation digest for %s", op_name)

        # Legacy priority 2: recover the latest verified staged implementation
        # and metrics from the verifier artifacts left by verify_once.sh.
        summary = (
            await _read_json(sandbox, f"{workspace}/output/verify/verify_result_summary.json")
            if artifact_dirs_safe
            else None
        )
        verify = (
            await _read_json(sandbox, f"{workspace}/output/verify/verify_result.json") if artifact_dirs_safe else None
        )
        perf = await _read_json(sandbox, f"{workspace}/output/verify/perf_result.json") if artifact_dirs_safe else None
        artifact_metrics = _metrics_from_agent_verify_artifacts(
            op_name=op_name,
            summary=summary,
            verify=verify,
            perf=perf,
        )
        staged_digest = staged_status.get("digest") if staged_status["substantive"] else None
        artifact_binding = _combined_metric_impl_binding(
            (summary, verify, perf),
            staged_digest or "",
        )
        artifacts_not_older = await _staged_implementation_precedes_artifacts(
            sandbox,
            staged_impl_path,
            (
                f"{workspace}/output/verify/verify_result_summary.json",
                f"{workspace}/output/verify/verify_result.json",
                f"{workspace}/output/verify/perf_result.json",
            ),
        )
        if artifact_metrics is not None and staged_digest and artifact_binding is not None and artifacts_not_older:
            recovered = await _recover_agent_time_best(
                sandbox,
                workspace,
                op_name,
                staged_impl_path=staged_impl_path,
                staged_digest=staged_digest,
                metrics=artifact_metrics,
            )
            if recovered:
                return finish(
                    artifact_metrics,
                    source="metrics_best",
                    implementation_path=f"src/{op_name}_triton_ascend_impl_best.py",
                    implementation_digest=staged_digest,
                    used_best=True,
                    binding=artifact_binding,
                    best_source="verifier_artifacts",
                )

        # Legacy priority 3: the flat metrics.json paired with current source.
        current = await _read_json(sandbox, f"{workspace}/metrics.json")
        current_digest = current_status.get("digest") if current_status["substantive"] else None
        if isinstance(current, dict) and current_digest and _has_verify_signal(current):
            binding = _metric_impl_binding(current, current_digest)
            canonical_current = _canonical_agent_metric_snapshot(current, op_name=op_name)
        else:
            binding = None
            canonical_current = None
        if canonical_current is not None and current_digest and binding is not None:
            return finish(
                current,
                source="metrics",
                implementation_path=f"src/{op_name}_triton_ascend_impl.py",
                implementation_digest=current_digest,
                used_best=False,
                binding=binding,
            )

        # A staged source counts only when its verifier artifacts were valid
        # enough to recover the pair above. Otherwise preserve the legacy
        # missing-implementation retry semantics.
        has_impl = bool(best_status["substantive"] or current_status["substantive"])
        if not has_impl:
            metrics = normalize_metrics(None, op_name=op_name)
            metrics.update(
                {
                    "success": False,
                    "compile_ok": False,
                    "correctness_ok": False,
                    "pass_rate": 0.0,
                    "error_type": "missing_impl",
                    "error": (
                        "no substantive ModelNew implementation changed from the fresh workspace baseline: "
                        f"current={current_status['reason']}, best={best_status['reason']}, "
                        f"staged={staged_status['reason']}"
                    ),
                }
            )
            return {
                "metrics": attach_reward(metrics, cfg.reward_weights),
                "selected_metrics_source": "not_run_missing_impl",
                "used_best_metrics": False,
                "_has_impl": False,
                "no_impl_retry_filter": True,
                "no_impl_retry_failed": True,
                "no_impl_retry_failed_reason": "missing_impl",
            }

        metrics = normalize_metrics(None, op_name=op_name)
        metrics.update(
            {
                "success": False,
                "compile_ok": False,
                "correctness_ok": False,
                "pass_rate": 0.0,
                "error_type": "missing_metrics",
                "error": "no reusable metrics_best, verifier artifacts, or metrics.json were found",
            }
        )
        return {
            "metrics": attach_reward(metrics, cfg.reward_weights),
            "selected_metrics_source": "missing_metrics",
            "used_best_metrics": False,
            "_has_impl": True,
        }

    async def _collect_artifacts(
        self,
        sandbox: Sandbox,
        cfg: TritonOperatorTaskConfig,
        metadata: dict[str, Any],
        workspace: str,
        op_name: str,
        *,
        attempt_index: int,
    ) -> None:
        if not cfg.artifact_dir:
            return
        runtime = metadata.get("runtime") if isinstance(metadata.get("runtime"), dict) else {}
        session_id = _safe_component(runtime.get("session_id") or "unknown-session")
        uid = _safe_component(metadata.get("uid") or op_name)
        destination = Path(cfg.artifact_dir).expanduser() / uid / session_id / f"attempt-{attempt_index + 1}"
        candidates = [
            *_ARTIFACT_PATHS,
            f"src/{op_name}_triton_ascend_impl.py",
            f"src/{op_name}_triton_ascend_impl_best.py",
        ]
        for relative in candidates:
            remote = f"{workspace}/{relative}"
            try:
                size = await _regular_file_size(sandbox, remote, max_bytes=cfg.max_artifact_bytes)
                if size is None:
                    continue
                await asyncio.wait_for(
                    sandbox.download_file(remote, destination / relative),
                    timeout=cfg.artifact_timeout,
                )
            except Exception:  # noqa: BLE001 - artifact export must not change reward
                logger.warning("artifact download failed: %s", remote, exc_info=True)


def _operator_name(metadata: dict[str, Any]) -> str:
    value = str(metadata.get("op_name") or "")
    if not _SAFE_OPERATOR.fullmatch(value):
        raise ValueError(f"unsafe or missing metadata.op_name: {value!r}")
    return value


def _support_filename(value: Any) -> str:
    text = str(value)
    path = PurePosixPath(text)
    if path.name != text or text in {"", ".", ".."}:
        raise ValueError(f"support filename must be a basename: {text!r}")
    return text


def _support_file_entries(value: Any) -> list[tuple[Any, Any]]:
    if value in (None, {}):
        return []
    if isinstance(value, dict):  # backward compatibility with early prepared rows
        return list(value.items())
    if not isinstance(value, list | tuple):
        raise TypeError("metadata.support_files must be a list of {name, content} records")
    entries: list[tuple[Any, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"name", "content"}:
            raise TypeError(f"metadata.support_files[{index}] must contain exactly name and content")
        entries.append((item["name"], item["content"]))
    return entries


def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in {"task_code", "support_files", "runtime"} and _json_safe(value)
    }


async def _validate_trusted_image(sandbox: Sandbox, paths: tuple[str, ...]) -> None:
    uid_result = await sandbox.exec(["id", "-u"], timeout=10)
    _require_success(uid_result, "sandbox user check")
    try:
        uid = int(uid_result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"sandbox user check returned invalid uid: {uid_result.stdout!r}") from exc
    if uid == 0:
        raise RuntimeError("Claude Code sandbox process must run as a non-root user")
    checked_directories: set[str] = set()
    for path in paths:
        tool_path = PurePosixPath(path)
        if not tool_path.is_absolute():
            raise RuntimeError(f"trusted tool path must be absolute: {path}")
        result = await sandbox.exec(
            ["stat", "-c", "%u:%a:%F", path],
            timeout=10,
            env={"LC_ALL": "C"},
        )
        _require_success(result, f"trusted tool stat for {path}")
        try:
            owner_text, mode_text, file_type = result.stdout.strip().split(":", maxsplit=2)
            owner = int(owner_text)
            mode = int(mode_text, 8)
        except ValueError as exc:
            raise RuntimeError(f"invalid trusted tool ownership for {path}: {result.stdout!r}") from exc
        if file_type != "regular file" or owner != 0 or mode & 0o111 == 0 or mode & 0o022:
            raise RuntimeError(f"trusted tool must be root-owned, executable, and not group/other writable: {path}")
        for parent in tool_path.parents:
            parent_text = parent.as_posix()
            if parent_text in checked_directories:
                continue
            checked_directories.add(parent_text)
            result = await sandbox.exec(
                ["stat", "-c", "%u:%a:%F", parent_text],
                timeout=10,
                env={"LC_ALL": "C"},
            )
            _require_success(result, f"trusted tool parent stat for {parent_text}")
            try:
                owner_text, mode_text, file_type = result.stdout.strip().split(":", maxsplit=2)
                owner = int(owner_text)
                mode = int(mode_text, 8)
            except ValueError as exc:
                raise RuntimeError(
                    f"invalid trusted tool parent ownership for {parent_text}: {result.stdout!r}"
                ) from exc
            if file_type != "directory" or owner != 0 or mode & 0o022:
                raise RuntimeError(
                    "trusted tool parent must be a root-owned, non-symlink, non-group/other-writable "
                    f"directory: {parent_text}"
                )


async def _validate_agent_verify_entrypoint(
    sandbox: Sandbox,
    workspace: str,
    relative_entrypoint: str,
    trusted_command: str,
) -> None:
    relative = PurePosixPath(relative_entrypoint)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("agent_verify_entrypoint must be a safe workspace-relative path")
    entrypoint = f"{workspace.rstrip('/')}/{relative.as_posix()}"
    resolved = await sandbox.exec(["readlink", "-f", entrypoint], timeout=10)
    _require_success(resolved, "agent-time verifier entrypoint resolution")
    if resolved.stdout.strip() != trusted_command:
        raise RuntimeError(
            "agent-time verifier entrypoint must resolve to the immutable trusted command: "
            f"expected {trusted_command!r}, got {resolved.stdout.strip()!r}"
        )


def _require_success(result: ExecResult, operation: str) -> None:
    if result.exit_code != 0:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise RuntimeError(f"{operation} failed with exit code {result.exit_code}: {detail}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


async def _regular_file_size(
    sandbox: Sandbox,
    path: str,
    *,
    max_bytes: int | None = None,
) -> int | None:
    """Return lstat size only for a bounded, non-symlink regular file."""

    result = await sandbox.exec(["stat", "-c", "%F:%s", "--", path], timeout=10, env={"LC_ALL": "C"})
    if result.exit_code != 0:
        return None
    try:
        file_type, size_text = result.stdout.strip().rsplit(":", maxsplit=1)
        size = int(size_text)
    except ValueError:
        return None
    if file_type != "regular file" or size < 0 or (max_bytes is not None and size > max_bytes):
        return None
    return size


async def _read_regular_file(
    sandbox: Sandbox,
    path: str,
    *,
    max_bytes: int,
    timeout: float = _FILE_READ_TIMEOUT,
) -> bytes | None:
    size = await _regular_file_size(sandbox, path, max_bytes=max_bytes)
    if size is None:
        return None
    try:
        raw = await asyncio.wait_for(sandbox.read_file(path), timeout=timeout)
    except Exception:
        return None
    data = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    return data if len(data) == size and len(data) <= max_bytes else None


async def _read_json(
    sandbox: Sandbox,
    path: str,
    *,
    max_bytes: int = _MAX_JSON_BYTES,
) -> dict[str, Any] | None:
    raw = await _read_regular_file(sandbox, path, max_bytes=max_bytes)
    if raw is None:
        return None
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


async def _read_early_stop(sandbox: Sandbox, workspace: str) -> dict[str, Any] | None:
    value = await _read_json(sandbox, f"{workspace}/{_EARLY_STOP_PATH}", max_bytes=4096)
    if not value or value.get("reason") not in {"no_verify_improvement", "no_latency_improvement"}:
        return None
    patience = value.get("patience")
    stale = value.get("stale_verify_count")
    verify_count = value.get("verify_count")
    if (
        not all(type(item) is int for item in (patience, stale, verify_count))
        or not 0 < patience <= stale <= verify_count
    ):
        return None
    return {
        "reason": value["reason"],
        "mode": value.get("mode"),
        "patience": patience,
        "stale_verify_count": stale,
        "verify_count": verify_count,
    }


async def _plain_directory(sandbox: Sandbox, path: str) -> bool:
    result = await sandbox.exec_shell(
        f"test -d {shlex.quote(path)} && test ! -L {shlex.quote(path)}",
        timeout=10,
    )
    if result.exit_code != 0:
        return False
    resolved = await sandbox.exec(["readlink", "-f", "--", path], timeout=10)
    return resolved.exit_code == 0 and resolved.stdout.strip().rstrip("/") == path.rstrip("/")


def _canonical_agent_metric_snapshot(metrics: dict[str, Any], *, op_name: str) -> dict[str, Any] | None:
    claimed_op = metrics.get("op_name")
    if claimed_op not in (None, "", op_name) or not _has_verify_signal(metrics):
        return None
    result = dict(metrics)
    for key in ("reward", "reward_components", "reward_score", "all_correct_bonus"):
        result.pop(key, None)
    count_keys = ("total_cases", "passed_cases", "failed_cases")
    if any(key in result for key in count_keys):
        counts = _agent_time_case_counts(result)
        if counts is None:
            return None
        total, passed, failed = counts
        result.update(
            {
                "total_cases": total,
                "passed_cases": passed,
                "failed_cases": failed,
                "pass_rate": round(passed / total, 6),
                "correctness_ok": passed == total and failed == 0,
                "success": passed == total and failed == 0,
            }
        )
    else:
        # Legacy flat files without counts may retain compile partial credit,
        # but may not claim correctness or performance credit.
        result.update(
            {
                "total_cases": 0,
                "passed_cases": 0,
                "failed_cases": 0,
                "pass_rate": 0.0,
                "correctness_ok": False,
                "success": False,
            }
        )
    passed = result["passed_cases"]
    compile_ok = result.get("compile_ok") is True or result.get("output_observed") is True or passed > 0
    result["compile_ok"] = compile_ok
    result["output_observed"] = result.get("output_observed") is True or passed > 0
    speedup = _strict_metric_speedup(result) if result["correctness_ok"] is True else None
    for key in ("speedup_vs_torch", "speedup", "geomean_speedup", "perf_data"):
        result.pop(key, None)
    if speedup is not None:
        result["perf_data"] = {"speedup": speedup}
    return result


def _strict_metric_speedup(metrics: dict[str, Any]) -> float | None:
    containers = [metrics]
    perf_data = metrics.get("perf_data")
    if isinstance(perf_data, dict):
        containers.append(perf_data)
        implementation = perf_data.get("implementation")
        if isinstance(implementation, dict):
            containers.append(implementation)
    for container in containers:
        for key in ("speedup_vs_torch", "speedup", "geomean_speedup"):
            if key not in container:
                continue
            value = container[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                return None
            converted = float(value)
            return converted if math.isfinite(converted) and converted >= 0 else None
    return None


def _metric_impl_binding(metrics: dict[str, Any], implementation_digest: str) -> str | None:
    claims = [
        metrics[key]
        for key in ("assistant_snapshot_impl_sha256", "verified_impl_sha256", "implementation_sha256")
        if key in metrics
    ]
    if not claims:
        return "legacy_unbound"
    if not implementation_digest or any(
        not isinstance(claim, str) or claim.lower() != implementation_digest for claim in claims
    ):
        return None
    return "matched"


def _combined_metric_impl_binding(
    snapshots: tuple[dict[str, Any] | None, ...],
    implementation_digest: str,
) -> str | None:
    """Require every artifact digest claim to agree with the staged bytes."""

    bindings = [_metric_impl_binding(item, implementation_digest) for item in snapshots if isinstance(item, dict)]
    if any(binding is None for binding in bindings):
        return None
    return "matched" if "matched" in bindings else "legacy_unbound"


async def _file_mtime(sandbox: Sandbox, path: str) -> int | None:
    if await _regular_file_size(sandbox, path) is None:
        return None
    result = await sandbox.exec(["stat", "-c", "%Y", "--", path], timeout=10, env={"LC_ALL": "C"})
    if result.exit_code != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


async def _staged_implementation_precedes_artifacts(
    sandbox: Sandbox,
    staged_path: str,
    artifact_paths: tuple[str, ...],
) -> bool:
    """Keep the legacy guard against pairing newer source with older results."""

    staged_mtime = await _file_mtime(sandbox, staged_path)
    if staged_mtime is None:
        return False
    artifact_mtimes = [await _file_mtime(sandbox, path) for path in artifact_paths]
    valid_mtimes = [mtime for mtime in artifact_mtimes if mtime is not None]
    return bool(valid_mtimes) and staged_mtime <= max(valid_mtimes)


def _has_verify_signal(metrics: dict[str, Any] | None) -> bool:
    if not isinstance(metrics, dict):
        return False
    return any(
        key in metrics
        for key in (
            "verified_success",
            "verify_exit",
            "total_cases",
            "passed_cases",
            "failed_cases",
            "pass_rate",
            "compile_ok",
            "correctness_ok",
        )
    )


def _agent_time_case_counts(verify: dict[str, Any] | None) -> tuple[int, int, int] | None:
    if not isinstance(verify, dict):
        return None
    values = tuple(verify.get(key) for key in ("total_cases", "passed_cases", "failed_cases"))
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return None
    total, passed, failed = values
    if total <= 0 or passed < 0 or failed < 0 or passed + failed != total:
        return None
    return total, passed, failed


def _agent_time_output_observed(verify: dict[str, Any], passed: int) -> bool:
    if passed > 0:
        return True
    failures = verify.get("failures")
    if not isinstance(failures, list):
        return False
    markers = (
        "输出不一致",
        "输出形状不一致",
        "NaN 位置不匹配",
        "Inf 位置",
        "布尔值不匹配",
        "output shape",
        "MERE=",
        "MARE=",
        "compare(fw_out, impl_out",
    )
    return any(marker in str(failure) for failure in failures for marker in markers)


def _metrics_from_agent_verify_artifacts(
    *,
    op_name: str,
    summary: dict[str, Any] | None,
    verify: dict[str, Any] | None,
    perf: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize the legacy verify_once.sh artifacts without rerunning it."""

    counts = _agent_time_case_counts(verify)
    if counts is None or verify is None:
        return None
    total, passed, failed = counts
    summary_perf = summary.get("perf_data") if isinstance(summary, dict) else None
    perf_data = perf if isinstance(perf, dict) else summary_perf if isinstance(summary_perf, dict) else None
    compile_raw = verify.get("compile_ok")
    output_raw = verify.get("output_observed")
    output_observed = (
        output_raw if type(output_raw) is bool else bool(perf_data) or _agent_time_output_observed(verify, passed)
    )
    compile_ok = compile_raw if type(compile_raw) is bool else output_observed
    if passed > 0:
        compile_ok = True
        output_observed = True
    correctness_ok = passed == total and failed == 0
    metrics: dict[str, Any] = {
        "op_name": op_name,
        "success": correctness_ok,
        "ast_check_ok": True,
        "compile_ok": compile_ok,
        "output_observed": output_observed,
        "correctness_ok": correctness_ok,
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "pass_rate": round(passed / total, 6),
    }
    speedup = _strict_metric_speedup({"perf_data": perf_data}) if correctness_ok and perf_data else None
    if speedup is not None:
        metrics["perf_data"] = {"speedup": speedup}
    if not correctness_ok:
        metrics["error_type"] = "compilation_failed" if not compile_ok else "correctness_failed"
    return metrics


async def _recover_agent_time_best(
    sandbox: Sandbox,
    workspace: str,
    op_name: str,
    *,
    staged_impl_path: str,
    staged_digest: str,
    metrics: dict[str, Any],
) -> bool:
    """Recreate the legacy best pair from the implementation staged by verify_once."""

    staged = await _read_regular_file(
        sandbox,
        staged_impl_path,
        max_bytes=_MAX_IMPLEMENTATION_BYTES,
    )
    if staged is None or hashlib.sha256(staged).hexdigest() != staged_digest:
        return False
    best_path = f"{workspace}/src/{op_name}_triton_ascend_impl_best.py"
    metrics_path = f"{workspace}/metrics_best.json"
    temporary_best = f"{best_path}.triton-recover"
    temporary_metrics = f"{metrics_path}.triton-recover"
    snapshot = {
        **metrics,
        "implementation_path": f"src/{op_name}_triton_ascend_impl_best.py",
        "implementation_sha256": staged_digest,
    }
    try:
        removed = await sandbox.exec(["rm", "-f", "--", temporary_best, temporary_metrics], timeout=10)
        _require_success(removed, "stale agent-time recovery temporary cleanup")
        await sandbox.write_file(temporary_best, staged)
        await sandbox.write_file(
            temporary_metrics,
            json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
        )
        moved = await sandbox.exec(["mv", "-f", "--", temporary_best, best_path], timeout=10)
        _require_success(moved, "agent-time best implementation recovery")
        moved = await sandbox.exec(["mv", "-f", "--", temporary_metrics, metrics_path], timeout=10)
        _require_success(moved, "agent-time best metrics recovery")
    except (TypeError, ValueError):
        return False
    recovered = await _read_regular_file(sandbox, best_path, max_bytes=_MAX_IMPLEMENTATION_BYTES)
    recovered_metrics = await _read_json(sandbox, metrics_path)
    return bool(
        recovered is not None
        and hashlib.sha256(recovered).hexdigest() == staged_digest
        and isinstance(recovered_metrics, dict)
        and _metric_impl_binding(recovered_metrics, staged_digest) == "matched"
    )


async def _file_sha256(
    sandbox: Sandbox,
    path: str,
    *,
    max_bytes: int | None = None,
) -> str | None:
    if await _regular_file_size(sandbox, path, max_bytes=max_bytes) is None:
        return None
    result = await sandbox.exec(["sha256sum", path], timeout=10)
    if result.exit_code != 0:
        return None
    digest = result.stdout.strip().split(maxsplit=1)[0].lower()
    return digest if re.fullmatch(r"[0-9a-f]{64}", digest) else None


async def _nonempty_file_sha256(
    sandbox: Sandbox,
    path: str,
    *,
    max_bytes: int | None = None,
) -> str | None:
    size = await _regular_file_size(sandbox, path, max_bytes=max_bytes)
    if size is None or size == 0:
        return None
    return await _file_sha256(sandbox, path, max_bytes=max_bytes)


async def _implementation_digests(sandbox: Sandbox, workspace: str, op_name: str) -> dict[str, str | None]:
    return {
        "current": await _nonempty_file_sha256(
            sandbox,
            f"{workspace}/src/{op_name}_triton_ascend_impl.py",
            max_bytes=_MAX_IMPLEMENTATION_BYTES,
        ),
        "best": await _nonempty_file_sha256(
            sandbox,
            f"{workspace}/src/{op_name}_triton_ascend_impl_best.py",
            max_bytes=_MAX_IMPLEMENTATION_BYTES,
        ),
    }


async def _implementation_status(
    sandbox: Sandbox,
    path: str,
    *,
    baseline_digest: str | None,
) -> dict[str, Any]:
    """Preserve the old no-implementation signals without a protocol shim."""

    digest = await _nonempty_file_sha256(sandbox, path, max_bytes=_MAX_IMPLEMENTATION_BYTES)
    if digest is None:
        return {"substantive": False, "digest": None, "reason": "missing_or_empty"}
    if digest == baseline_digest:
        return {"substantive": False, "digest": digest, "reason": "unchanged_template"}
    raw = await _read_regular_file(sandbox, path, max_bytes=_MAX_IMPLEMENTATION_BYTES)
    if raw is None or hashlib.sha256(raw).hexdigest() != digest:
        return {"substantive": False, "digest": digest, "reason": "unreadable"}
    try:
        tree = ast.parse(raw.decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return {"substantive": False, "digest": digest, "reason": "invalid_python"}
    model_classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelNew"]
    if not model_classes:
        reason = "missing_model_new"
    elif not any(_class_has_substantive_body(node) for node in model_classes):
        reason = "not_implemented"
    else:
        return {"substantive": True, "digest": digest, "reason": "valid_impl_shape"}
    return {"substantive": False, "digest": digest, "reason": reason}


def _class_has_substantive_body(node: ast.ClassDef) -> bool:
    """Distinguish a real ``ModelNew`` AST from template-only stubs.

    Comments and string literals mentioning a class or ``NotImplementedError``
    never affect this check.  Constructor/setup code alone is insufficient: at
    least one ``forward``/``__call__`` execution entrypoint must contain a
    statement beyond a docstring, ``pass``, ellipsis, or a direct
    ``raise NotImplementedError``.
    """

    entrypoints = (
        statement
        for statement in node.body
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef) and statement.name in {"forward", "__call__"}
    )
    return any(any(not _is_stub_statement(child) for child in method.body) for method in entrypoints)


def _is_stub_statement(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Pass):
        return True
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
        return statement.value.value is Ellipsis or isinstance(statement.value.value, str)
    if isinstance(statement, ast.Raise):
        exception = statement.exc
        if isinstance(exception, ast.Call):
            exception = exception.func
        return isinstance(exception, ast.Name) and exception.id == "NotImplementedError"
    return False


def _train_best(
    metrics: dict[str, Any],
    source: str,
    *,
    best_impl_digest: str | None,
) -> dict[str, Any]:
    recorded_digest = str(metrics.get("assistant_snapshot_impl_sha256") or "").lower()
    if not best_impl_digest or recorded_digest != best_impl_digest:
        return {}
    assistant_index = metrics.get("assistant_index")
    messages_seen = metrics.get("assistant_messages_seen")
    index_source = metrics.get("assistant_index_source")
    if (
        type(assistant_index) is not int
        or assistant_index < 0
        or type(messages_seen) is not int
        or messages_seen != assistant_index + 1
        or index_source != "claude_code_transcript_hook"
    ):
        return {}
    return {
        "source": source,
        "used_best_metrics": source == "metrics_best",
        "assistant_index": assistant_index,
        "assistant_messages_seen": messages_seen,
        "assistant_index_source": index_source,
    }


def _compact_extra_info(
    op_name: str,
    evaluation: dict[str, Any],
    agent_info: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    metrics = evaluation.get("metrics") if isinstance(evaluation.get("metrics"), dict) else {}
    compact_metrics = {
        key: metrics[key]
        for key in (
            "success",
            "compile_ok",
            "correctness_ok",
            "total_cases",
            "passed_cases",
            "failed_cases",
            "pass_rate",
            "reward",
            "reward_components",
            "agent_time_metrics_reused",
            "metrics_impl_path",
            "metrics_impl_sha256",
            "metrics_impl_binding",
            "error_type",
        )
        if key in metrics
    }
    result: dict[str, Any] = {
        "op_name": op_name,
        "reward_score": float(metrics.get("reward", 0.0)),
        "metrics": compact_metrics,
        "selected_metrics_source": evaluation.get("selected_metrics_source"),
        "used_best_metrics": bool(evaluation.get("used_best_metrics")),
        "agent": agent_info,
        "provenance": {
            key: str(metadata[key])[:256]
            for key in (
                "dataset_name",
                "dataset_revision",
                "dataset_kind",
                "source_id",
                "source_fingerprint",
                "split",
                "arch",
                "benchmark_name",
                "benchmark_problem_id",
                "benchmark_level",
            )
            if metadata.get(key) not in (None, "")
        },
        "timing_ms": evaluation.get("timing_ms", {}),
    }
    for key in (
        "best_source",
        "train_best",
        "no_impl_retry_used",
        "no_impl_retry_attempts",
        "no_impl_retry_reason",
        "no_impl_retry_discard_previous",
        "no_impl_retry_filter",
        "no_impl_retry_failed",
        "no_impl_retry_failed_reason",
    ):
        if key in evaluation:
            result[key] = evaluation[key]
    return result


def _safe_component(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("._")[:128] or "unknown"


def _json_safe(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)
