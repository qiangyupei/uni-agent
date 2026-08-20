"""Per-session NPU operator task using Uni-Agent Task/Sandbox/ClaudeCodeAgent."""

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


class TritonOperatorTaskConfig(TaskConfig):
    name: str = "triton_operator"
    workspace_dir: str = Field(default="/workspace", description="Fresh per-session workspace inside the sandbox.")
    template_dir: str | None = Field(
        default="/opt/triton-agent-template",
        description="Root-owned task/verifier template baked into the sandbox image; None stages only sample files.",
    )
    verify_command: str = Field(
        default=(
            "/opt/triton-agent-tools/with_npu_lease.py -- "
            "/opt/triton-agent-tools/final_verify.sh {op_name} {workspace_dir}"
        ),
        description="Final verifier command. Only quoted {op_name} and {workspace_dir} placeholders are supported.",
    )
    npu_lease_command: str = "/opt/triton-agent-tools/with_npu_lease.py"
    npu_lease_required: bool = True
    agent_verify_entrypoint: str = "tools/verify_once.sh"
    trusted_agent_verify_command: str = "/opt/triton-agent-tools/verify_once.sh"
    enforce_trusted_image: bool = True
    trusted_tool_paths: tuple[str, ...] = (
        "/opt/triton-agent-tools/with_npu_lease.py",
        "/opt/triton-agent-tools/verify_once.sh",
        "/opt/triton-agent-tools/final_verify.sh",
        "/opt/triton-agent-tools/cleanup_task_processes.sh",
    )
    setup_timeout: float = 120.0
    verify_timeout: float = 1200.0
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
    """Stage one task, run stock Claude Code, verify it, then destroy its sandbox.

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
                except Exception as exc:  # preserve partial workspace for final verification
                    logger.exception("Claude Code failed for %s; evaluating partial workspace", op_name)
                    agent_info = {"finished": False, "error_type": type(exc).__name__, "error": str(exc)[:512]}
                    finished = False
                agent_ms = _elapsed_ms(agent_started)

                # No agent/verifier child may race the trusted final verifier or
                # keep an evaluator NPU context alive. This command is an
                # immutable image asset and failure aborts the attempt.
                await self._cleanup_task_processes(sandbox, cfg, workspace, required=True)
                verify_started = time.monotonic()
                evaluation = await self._verify_workspace(
                    sandbox,
                    cfg,
                    workspace,
                    op_name,
                    metadata=metadata,
                    initial_impl_digests=initial_impl_digests,
                )
                verify_ms = _elapsed_ms(verify_started)
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
            "verify": verify_ms,
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
                raise RuntimeError("cleanup_command is required before trusted final verification")
            return
        result = await asyncio.shield(
            sandbox.exec_shell(
                cfg.cleanup_command,
                timeout=cfg.cleanup_timeout,
                workdir=workspace,
            )
        )
        if required:
            _require_success(result, "pre-verification process cleanup")

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
            commands.append(f"cp -a {template}/. {workspace}/")
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
            await self._install_transcript_hooks(sandbox, cfg.workspace_dir)

    async def _install_transcript_hooks(self, sandbox: Sandbox, workspace: str) -> None:
        hook_source = (Path(__file__).with_name("assets") / "track_verify_snapshot.py").read_bytes()
        hook_path = f"{workspace}/.claude/hooks/track_verify_snapshot.py"
        created = await sandbox.exec(["mkdir", "-p", f"{workspace}/.claude/hooks"], timeout=10)
        _require_success(created, "Claude transcript hook directory setup")
        await sandbox.write_file(hook_path, hook_source)
        settings_path = f"{workspace}/.claude/settings.json"
        settings = await _read_json(sandbox, settings_path) or {}
        hooks = settings.setdefault("hooks", {})
        if not isinstance(hooks, dict):
            raise TypeError(".claude/settings.json 'hooks' must be a mapping")
        quoted_hook = shlex.quote(hook_path)
        for event, mode in (("PreToolUse", "pre"), ("PostToolUse", "post")):
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
        result = await sandbox.exec(["chmod", "0444", hook_path, settings_path], timeout=10)
        _require_success(result, "Claude transcript hook protection")

    async def _verify_workspace(
        self,
        sandbox: Sandbox,
        cfg: TritonOperatorTaskConfig,
        workspace: str,
        op_name: str,
        *,
        metadata: dict[str, Any],
        initial_impl_digests: dict[str, str | None],
    ) -> dict[str, Any]:
        expected_inputs = await _restore_verifier_inputs(
            sandbox,
            workspace,
            metadata,
            op_name,
        )
        input_manifest_digest = _digest_manifest(expected_inputs)
        # Capture only the assistant hint from the agent-time best pair. Its
        # numerical metrics are never trusted for reward.
        best = await _read_json(sandbox, f"{workspace}/metrics_best.json")
        best_impl_path = f"{workspace}/src/{op_name}_triton_ascend_impl_best.py"
        generated_digests = await _implementation_digests(sandbox, workspace, op_name)
        best_impl_digest = generated_digests["best"]
        current_impl_path = f"{workspace}/src/{op_name}_triton_ascend_impl.py"
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
        has_best_impl = bool(best_status["substantive"])
        has_current_impl = bool(current_status["substantive"])
        has_impl = has_best_impl or has_current_impl
        if not has_impl:
            metrics = normalize_metrics(None, op_name=op_name)
            metrics.update(
                {
                    "success": False,
                    "compile_ok": False,
                    "correctness_ok": False,
                    "pass_rate": 0.0,
                    "verifier_attested": False,
                    "verifier_inputs_intact": True,
                    "verifier_input_manifest_sha256": input_manifest_digest,
                    "error_type": "missing_impl",
                    "error": (
                        "no substantive ModelNew implementation changed from the fresh workspace baseline: "
                        f"current={current_status['reason']}, best={best_status['reason']}"
                    ),
                }
            )
            return {
                "metrics": attach_reward(metrics, cfg.reward_weights),
                "selected_metrics_source": "not_run_missing_impl",
                "used_best_metrics": False,
                "verifier_input_manifest_sha256": input_manifest_digest,
                "_has_impl": False,
                "no_impl_retry_filter": True,
                "no_impl_retry_failed": True,
                "no_impl_retry_failed_reason": "missing_impl",
            }

        # The agent controls the workspace and may replace output/verify with a
        # symlink. Remove the whole known output subtree without following a
        # symlink argument, then recreate the verifier directory after process
        # cleanup so stale or out-of-workspace JSON cannot satisfy attestation.
        await _reset_verifier_output_dir(sandbox, workspace)

        command = cfg.verify_command.format_map(
            {"op_name": shlex.quote(op_name), "workspace_dir": shlex.quote(workspace)}
        )
        result = await sandbox.exec_shell(command, timeout=cfg.verify_timeout, workdir=workspace)
        # A trusted verifier must be synchronous, but enforce quiescence again
        # before reading attestation/input digests in case it leaked children.
        await self._cleanup_task_processes(sandbox, cfg, workspace, required=True)
        verify = await _read_json(sandbox, f"{workspace}/output/verify/verify_result.json")
        perf = await _read_json(sandbox, f"{workspace}/output/verify/perf_result.json")
        inputs_intact = await _files_match(sandbox, expected_inputs)

        attested_path, attested_digest = await _verified_implementation(sandbox, workspace, op_name, verify)
        valid_counts = _verifier_case_counts_valid(verify)
        substantive_digests = {
            f"src/{op_name}_triton_ascend_impl.py": current_status["digest"] if current_status["substantive"] else None,
            f"src/{op_name}_triton_ascend_impl_best.py": best_status["digest"] if best_status["substantive"] else None,
        }
        attests_substantive_impl = bool(
            attested_path and attested_digest and substantive_digests.get(attested_path) == attested_digest
        )
        attested = (
            result.exit_code == 0
            and attested_path is not None
            and attested_digest is not None
            and valid_counts
            and attests_substantive_impl
            and inputs_intact
        )
        attested_perf = _attested_perf(perf, workspace, attested_path, attested_digest) if attested else None
        metrics = normalize_metrics(None, verify=verify, perf=attested_perf, op_name=op_name)
        metrics["verify_exit_code"] = result.exit_code
        metrics["verifier_attested"] = attested
        metrics["perf_attested"] = bool(attested_perf)
        metrics["verifier_inputs_intact"] = inputs_intact
        metrics["verifier_input_manifest_sha256"] = input_manifest_digest
        if attested:
            metrics["verified_impl_path"] = attested_path
            metrics["verified_impl_sha256"] = attested_digest
        else:
            metrics.update(
                {
                    "success": False,
                    "compile_ok": False,
                    "correctness_ok": False,
                    "pass_rate": 0.0,
                    "error_type": "untrusted_verifier_output",
                    "error": (result.stderr or result.stdout).strip()[-512:]
                    or (
                        "final verifier did not emit a matching implementation attestation "
                        "with positive, internally consistent case counts"
                    ),
                }
            )
        metrics = attach_reward(metrics, cfg.reward_weights)

        used_best = bool(attested and attested_digest == best_impl_digest)
        evaluation: dict[str, Any] = {
            "metrics": metrics,
            "selected_metrics_source": "final_verifier",
            "used_best_metrics": used_best,
            "verifier_input_manifest_sha256": input_manifest_digest,
            "_has_impl": has_impl,
        }
        train_best = _train_best(
            best or {},
            "metrics_best",
            best_impl_digest=attested_digest if used_best else None,
        )
        if train_best:
            evaluation["train_best"] = train_best
        return evaluation

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


async def _restore_verifier_inputs(
    sandbox: Sandbox,
    workspace: str,
    metadata: dict[str, Any],
    op_name: str,
) -> dict[str, str]:
    """Reset agent-visible verifier inputs after process cleanup and hash them."""

    task_code = metadata.get("task_code")
    if not isinstance(task_code, str) or not task_code.strip():
        raise ValueError("task metadata requires non-empty task_code")
    src_dir = f"{workspace}/src"
    hidden_dir = f"{workspace}/.triton_case_sidecars"
    src_check = await sandbox.exec_shell(
        f"test -d {shlex.quote(src_dir)} && test ! -L {shlex.quote(src_dir)}",
        timeout=10,
    )
    _require_success(src_check, "verifier source directory integrity check")
    recreated = await sandbox.exec(["rm", "-rf", "--", hidden_dir], timeout=30)
    _require_success(recreated, "stale hidden verifier input cleanup")
    recreated = await sandbox.exec(["mkdir", "-p", hidden_dir], timeout=10)
    _require_success(recreated, "hidden verifier input recreation")

    files: dict[str, str | bytes] = {f"{src_dir}/{op_name}.py": task_code}
    reserved = {
        f"{op_name}.py",
        f"{op_name}_triton_ascend_impl.py",
        f"{op_name}_triton_ascend_impl_best.py",
    }
    for filename, content in _support_file_entries(metadata.get("support_files", [])):
        safe_name = _support_filename(filename)
        if safe_name in reserved:
            raise ValueError(f"support file conflicts with a task/implementation path: {safe_name}")
        if not isinstance(content, str | bytes):
            raise TypeError(f"support file {safe_name!r} must contain text or bytes")
        files[f"{src_dir}/{safe_name}"] = content
        files[f"{hidden_dir}/{safe_name}"] = content
    public_metadata = json.dumps(_public_metadata(metadata), ensure_ascii=False, indent=2) + "\n"
    files[f"{workspace}/TASK_METADATA.json"] = public_metadata

    expected: dict[str, str] = {}
    for path, content in files.items():
        raw = content.encode() if isinstance(content, str) else content
        temporary = f"{path}.triton-restore"
        removed = await sandbox.exec(["rm", "-f", "--", temporary], timeout=10)
        _require_success(removed, f"temporary verifier input cleanup for {path}")
        await sandbox.write_file(temporary, raw)
        replaced = await sandbox.exec(["mv", "-f", "--", temporary, path], timeout=10)
        _require_success(replaced, f"atomic verifier input reset for {path}")
        expected[path] = hashlib.sha256(raw).hexdigest()
    protected = await sandbox.exec(["chmod", "0444", *expected], timeout=10)
    _require_success(protected, "verifier input protection")
    return expected


async def _reset_verifier_output_dir(sandbox: Sandbox, workspace: str) -> None:
    output_dir = f"{workspace.rstrip('/')}/output"
    removed = await sandbox.exec(["rm", "-rf", "--", output_dir], timeout=30)
    _require_success(removed, "stale verifier output subtree cleanup")
    created = await sandbox.exec(["mkdir", "-p", f"{output_dir}/verify"], timeout=10)
    _require_success(created, "verifier output directory recreation")
    checked = await sandbox.exec_shell(
        " && ".join(
            (
                f"test -d {shlex.quote(output_dir)}",
                f"test ! -L {shlex.quote(output_dir)}",
                f"test -d {shlex.quote(output_dir)}/verify",
                f"test ! -L {shlex.quote(output_dir)}/verify",
            )
        ),
        timeout=10,
    )
    _require_success(checked, "verifier output directory integrity check")


async def _files_match(sandbox: Sandbox, expected: dict[str, str]) -> bool:
    for path, digest in expected.items():
        if await _file_sha256(sandbox, path) != digest:
            return False
    return True


def _digest_manifest(expected: dict[str, str]) -> str:
    payload = "\n".join(f"{path}\0{digest}" for path, digest in sorted(expected.items()))
    return hashlib.sha256(payload.encode()).hexdigest()


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


def _verifier_case_counts_valid(verify: dict[str, Any] | None) -> bool:
    if not isinstance(verify, dict):
        return False
    values = [verify.get(key) for key in ("total_cases", "passed_cases", "failed_cases")]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return False
    total, passed, failed = values
    counts_valid = total > 0 and 0 <= passed <= total and 0 <= failed <= total and passed + failed == total
    if not counts_valid or type(verify.get("compile_ok")) is not bool:
        return False
    if passed > 0 and verify["compile_ok"] is not True:
        return False
    for optional_bool in ("correctness_ok", "output_observed"):
        if optional_bool in verify and type(verify[optional_bool]) is not bool:
            return False
    if "correctness_ok" in verify and verify["correctness_ok"] != (passed == total and failed == 0):
        return False
    return True


def _attested_perf(
    perf: dict[str, Any] | None,
    workspace: str,
    attested_path: str | None,
    attested_digest: str | None,
) -> dict[str, Any] | None:
    """Accept finite performance data only when bound to the verified bytes."""

    if not isinstance(perf, dict) or not attested_path or not attested_digest:
        return None
    claimed_path = str(perf.get("verified_impl_path") or "")
    allowed_paths = {
        attested_path,
        f"./{attested_path}",
        f"{workspace.rstrip('/')}/{attested_path}",
    }
    if claimed_path not in allowed_paths or str(perf.get("verified_impl_sha256") or "").lower() != attested_digest:
        return None
    containers = [perf]
    implementation = perf.get("implementation")
    if isinstance(implementation, dict):
        containers.append(implementation)
    for container in containers:
        for key in ("speedup_vs_torch", "speedup", "geomean_speedup"):
            value = container.get(key)
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            speedup = float(value)
            if math.isfinite(speedup) and speedup >= 0:
                return perf
    return None


async def _verified_implementation(
    sandbox: Sandbox,
    workspace: str,
    op_name: str,
    verify: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    """Validate the trusted final verifier's path + content digest attestation."""

    if not isinstance(verify, dict):
        return None, None
    claimed_path = str(verify.get("verified_impl_path") or "")
    claimed_digest = str(verify.get("verified_impl_sha256") or "").lower()
    allowed_relative = {
        f"src/{op_name}_triton_ascend_impl.py",
        f"src/{op_name}_triton_ascend_impl_best.py",
    }
    if claimed_path.startswith(f"{workspace.rstrip('/')}/"):
        relative = claimed_path[len(workspace.rstrip("/")) + 1 :]
    elif claimed_path.startswith("./"):
        relative = claimed_path[2:]
    else:
        relative = claimed_path
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return None, None
    relative = relative_path.as_posix()
    if relative not in allowed_relative or not re.fullmatch(r"[0-9a-f]{64}", claimed_digest):
        return None, None
    actual_digest = await _nonempty_file_sha256(
        sandbox,
        f"{workspace}/{relative}",
        max_bytes=_MAX_IMPLEMENTATION_BYTES,
    )
    if actual_digest != claimed_digest:
        return None, None
    return relative, actual_digest


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
            "verify_exit_code",
            "verifier_attested",
            "perf_attested",
            "verifier_inputs_intact",
            "verifier_input_manifest_sha256",
            "verified_impl_path",
            "verified_impl_sha256",
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
