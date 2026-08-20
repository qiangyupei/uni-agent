from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from uni_agent.agents import AgentResult
from uni_agent.sandbox import ExecResult

from ..task import (
    TritonOperatorTask,
    TritonOperatorTaskConfig,
    _attested_perf,
    _implementation_status,
    _read_json,
    _train_best,
    _validate_agent_verify_entrypoint,
    _validate_trusted_image,
)


class FakeSandbox:
    def __init__(
        self,
        *,
        has_impl: bool = True,
        events: list[str] | None = None,
        label: str = "sandbox",
        verifier_counts: tuple[int, int, int] = (2, 2, 0),
        tamper_inputs: bool = False,
        mutate_input_after_verify: bool = False,
        uid: int = 1000,
        verify_exit_code: int = 0,
        trusted_file_type: str = "regular file",
        trusted_parent_mode: str = "755",
        verify_entrypoint_target: str = "/opt/triton-agent-tools/verify_once.sh",
    ) -> None:
        self.started = False
        self.stopped = False
        self.cleanup_calls = 0
        self.verify_saw_cleanup = False
        self.events = events
        self.label = label
        self.produce_impl = has_impl
        self.verifier_counts = verifier_counts
        self.tamper_inputs = tamper_inputs
        self.mutate_input_after_verify = mutate_input_after_verify
        self.verify_reference: bytes | None = None
        self.verify_cases: bytes | None = None
        self.uid = uid
        self.verify_exit_code = verify_exit_code
        self.trusted_file_type = trusted_file_type
        self.trusted_parent_mode = trusted_parent_mode
        self.verify_entrypoint_target = verify_entrypoint_target
        self.files: dict[str, bytes] = {}
        self.file_types: dict[str, str] = {}
        self.read_file_calls: list[str] = []
        self.download_file_calls: list[str] = []

    async def __aenter__(self):
        self.started = True
        if self.events is not None:
            self.events.append(f"{self.label}:enter")
        return self

    async def __aexit__(self, *exc):
        self.stopped = True
        if self.events is not None:
            self.events.append(f"{self.label}:exit")

    async def exec_shell(self, script, *, timeout=None, workdir=None, env=None):
        del timeout, workdir, env
        if script == "/trusted/cleanup":
            self.cleanup_calls += 1
        if script.startswith("/trusted/final_verify"):
            self.verify_saw_cleanup = self.cleanup_calls > 0
            self.verify_reference = self.files.get("/workspace/src/smoke.py")
            self.verify_cases = next(
                (
                    content
                    for path, content in self.files.items()
                    if path.startswith("/workspace/src/smoke_") and path.endswith(".json")
                ),
                None,
            )
            implementation = self.files.get("/workspace/src/smoke_triton_ascend_impl.py")
            if implementation is None:
                return ExecResult(1, "", "missing implementation")
            digest = hashlib.sha256(implementation).hexdigest()
            self.files["/workspace/output/verify/verify_result.json"] = json.dumps(
                {
                    "total_cases": self.verifier_counts[0],
                    "passed_cases": self.verifier_counts[1],
                    "failed_cases": self.verifier_counts[2],
                    "compile_ok": True,
                    "verified_impl_path": "src/smoke_triton_ascend_impl.py",
                    "verified_impl_sha256": digest,
                }
            ).encode()
            self.files["/workspace/output/verify/perf_result.json"] = json.dumps(
                {
                    "speedup_vs_torch": 1.0,
                    "verified_impl_path": "src/smoke_triton_ascend_impl.py",
                    "verified_impl_sha256": digest,
                }
            ).encode()
            if self.mutate_input_after_verify:
                self.files["/workspace/src/smoke.py"] = b"# verifier tampered\n"
            return ExecResult(self.verify_exit_code, "", "verification infrastructure failed")
        return ExecResult(0, "", "")

    async def exec(self, argv, *, timeout=None, workdir=None, env=None):
        del timeout, workdir, env
        if argv == ["pwd"]:
            return ExecResult(0, "/workspace\n", "")
        if argv == ["id", "-u"]:
            return ExecResult(0, f"{self.uid}\n", "")
        if argv[:2] == ["readlink", "-f"]:
            return ExecResult(0, f"{self.verify_entrypoint_target}\n", "")
        if argv[:2] == ["test", "-f"]:
            return ExecResult(int(argv[2] not in self.files), "", "")
        if argv[:2] == ["test", "-s"]:
            return ExecResult(int(not self.files.get(argv[2])), "", "")
        if argv[:2] == ["rm", "-f"]:
            for path in argv[2:]:
                if path != "--":
                    self.files.pop(path, None)
            return ExecResult(0, "", "")
        if argv[:3] == ["rm", "-rf", "--"]:
            prefix = argv[3].rstrip("/") + "/"
            for path in list(self.files):
                if path == argv[3] or path.startswith(prefix):
                    self.files.pop(path, None)
            return ExecResult(0, "", "")
        if argv[:3] == ["mv", "-f", "--"]:
            self.files[argv[4]] = self.files.pop(argv[3])
            return ExecResult(0, "", "")
        if argv[:3] == ["stat", "-c", "%F:%s"]:
            content = self.files.get(argv[-1])
            return (
                ExecResult(0, f"{self.file_types.get(argv[-1], 'regular file')}:{len(content)}\n", "")
                if content is not None
                else ExecResult(1, "", "missing")
            )
        if argv[:3] == ["stat", "-c", "%u:%a:%F"]:
            is_tool = argv[3].endswith((".py", ".sh", "final_verify"))
            if is_tool:
                return ExecResult(0, f"0:755:{self.trusted_file_type}\n", "")
            return ExecResult(0, f"0:{self.trusted_parent_mode}:directory\n", "")
        if argv and argv[0] == "sha256sum":
            content = self.files.get(argv[1])
            if content is None:
                return ExecResult(1, "", "missing")
            return ExecResult(0, f"{hashlib.sha256(content).hexdigest()}  {argv[1]}\n", "")
        return ExecResult(0, "", "")

    async def write_file(self, path, content):
        self.files[path] = content.encode() if isinstance(content, str) else content

    async def read_file(self, path):
        self.read_file_calls.append(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def download_file(self, remote_path, local_path):
        self.download_file_calls.append(remote_path)
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[remote_path])


class FakeAgent:
    async def run(self, *, sandbox, messages):
        del messages
        if sandbox.produce_impl:
            sandbox.files["/workspace/src/smoke_triton_ascend_impl.py"] = (
                b"class ModelNew:\n    def forward(self, value):\n        return value\n"
            )
        if sandbox.tamper_inputs:
            sandbox.files["/workspace/src/smoke.py"] = b"# agent tampered\n"
            for path in list(sandbox.files):
                if path.startswith("/workspace/src/smoke_") and path.endswith(".json"):
                    sandbox.files[path] = b'{"tampered":true}\n'
        return AgentResult(info={"exit_code": 0}, finished=True)


def test_task_owns_sandbox_lifecycle_and_reports_compact_extra_info(tmp_path: Path) -> None:
    sandbox = FakeSandbox()
    sandbox.files["/workspace/output/verify/agent-controlled.json"] = b"{}\n"
    sandbox.files["/workspace/metrics.json"] = b'{"agent":"controlled"}\n'
    sandbox.file_types["/workspace/metrics.json"] = "symbolic link"
    config = TritonOperatorTaskConfig(
        sandbox={"provider": "local"},
        agent={"name": "claude_code", "model": {"base_url": "http://unused", "model_name": "unused"}},
        prompt=[{"role": "user", "content": "implement"}],
        metadata={"uid": "uid-1", "op_name": "smoke", "task_code": "class Model: pass\n"},
        template_dir=None,
        verify_command="/trusted/final_verify {op_name} {workspace_dir}",
        cleanup_command="/trusted/cleanup",
        artifact_dir=str(tmp_path),
    )
    task = TritonOperatorTask(config)
    task.build_sandbox = lambda: sandbox  # type: ignore[method-assign]
    task.build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    result = asyncio.run(task.run())

    assert sandbox.started and sandbox.stopped
    assert sandbox.cleanup_calls == 3
    assert sandbox.verify_saw_cleanup is True
    assert result.finished is True
    assert result.accuracy == 1.0
    assert result.reward == 0.95
    assert result.extra_info is not None
    assert result.extra_info["metrics"]["correctness_ok"] is True
    assert "task_code" not in result.extra_info
    assert sandbox.files["/workspace/src/smoke.py"].startswith(b"class Model")
    assert "/workspace/output/verify/agent-controlled.json" not in sandbox.files
    assert "/workspace/metrics.json" not in sandbox.download_file_calls


def test_stub_or_missing_modelnew_is_not_a_generated_implementation() -> None:
    sandbox = FakeSandbox()
    path = "/workspace/src/smoke_triton_ascend_impl.py"
    sandbox.files[path] = b"class ModelNew:\n    raise NotImplementedError('Replace this stub')\n"

    status = asyncio.run(_implementation_status(sandbox, path, baseline_digest=None))
    assert status["substantive"] is False
    assert status["reason"] == "not_implemented"

    sandbox.files[path] = b"def helper():\n    return 1\n"
    status = asyncio.run(_implementation_status(sandbox, path, baseline_digest=None))
    assert status["substantive"] is False
    assert status["reason"] == "missing_model_new"

    sandbox.files[path] = b"class ModelNew:\n    pass\n"
    status = asyncio.run(_implementation_status(sandbox, path, baseline_digest=None))
    assert status["substantive"] is False
    assert status["reason"] == "not_implemented"


def test_implementation_shape_uses_ast_not_comments_or_substrings() -> None:
    sandbox = FakeSandbox()
    path = "/workspace/src/smoke_triton_ascend_impl.py"
    sandbox.files[path] = (
        b"""\n# class ModelNew: pass\ndef helper():\n    return "class ModelNew and NotImplementedError"\n"""
    )
    status = asyncio.run(_implementation_status(sandbox, path, baseline_digest=None))
    assert status["reason"] == "missing_model_new"

    sandbox.files[path] = (
        b"\nclass ModelNew:\n"
        b'    """NotImplementedError is discussed here, not raised."""\n'
        b"    def forward(self, value):\n"
        b"        return value\n"
    )
    status = asyncio.run(_implementation_status(sandbox, path, baseline_digest=None))
    assert status["substantive"] is True
    assert status["reason"] == "valid_impl_shape"


def test_template_constructor_does_not_hide_stub_forward() -> None:
    sandbox = FakeSandbox()
    path = "/workspace/src/smoke_triton_ascend_impl.py"
    sandbox.files[path] = (
        b"class ModelNew(nn.Module):\n"
        b"    def __init__(self):\n"
        b"        super().__init__()\n"
        b"    def forward(self, value):\n"
        b"        raise NotImplementedError('replace this stub')\n"
        b"# agent changed only this comment\n"
    )

    status = asyncio.run(_implementation_status(sandbox, path, baseline_digest="0" * 64))

    assert status["substantive"] is False
    assert status["reason"] == "not_implemented"


def test_agent_symlink_implementation_is_rejected_without_reading_target() -> None:
    sandbox = FakeSandbox()
    path = "/workspace/src/smoke_triton_ascend_impl.py"
    sandbox.files[path] = b"class ModelNew: pass\n"
    sandbox.file_types[path] = "symbolic link"

    status = asyncio.run(_implementation_status(sandbox, path, baseline_digest=None))

    assert status["substantive"] is False
    assert status["reason"] == "missing_or_empty"
    assert path not in sandbox.read_file_calls


def test_json_reads_reject_symlinks_oversize_and_non_finite_values() -> None:
    sandbox = FakeSandbox()
    path = "/workspace/metrics_best.json"
    sandbox.files[path] = b'{"assistant_index": 1}'
    sandbox.file_types[path] = "symbolic link"
    assert asyncio.run(_read_json(sandbox, path)) is None
    assert path not in sandbox.read_file_calls

    sandbox.file_types[path] = "regular file"
    sandbox.files[path] = b"x" * (1024 * 1024 + 1)
    assert asyncio.run(_read_json(sandbox, path)) is None
    assert path not in sandbox.read_file_calls

    sandbox.files[path] = b'{"speedup": Infinity}'
    assert asyncio.run(_read_json(sandbox, path)) is None


def test_performance_data_must_be_finite_and_bound_to_verified_bytes() -> None:
    digest = "a" * 64
    base = {
        "verified_impl_path": "src/smoke_triton_ascend_impl.py",
        "verified_impl_sha256": digest,
        "speedup": 1.5,
    }
    assert _attested_perf(base, "/workspace", "src/smoke_triton_ascend_impl.py", digest) == base
    assert _attested_perf({**base, "speedup": float("inf")}, "/workspace", base["verified_impl_path"], digest) is None
    assert _attested_perf({**base, "speedup": "1.5"}, "/workspace", base["verified_impl_path"], digest) is None
    assert (
        _attested_perf({**base, "verified_impl_sha256": "b" * 64}, "/workspace", base["verified_impl_path"], digest)
        is None
    )


def test_best_hint_requires_consistent_native_hook_metadata() -> None:
    digest = "a" * 64
    valid = {
        "assistant_snapshot_impl_sha256": digest,
        "assistant_index": 2,
        "assistant_messages_seen": 3,
        "assistant_index_source": "claude_code_transcript_hook",
    }
    assert _train_best(valid, "metrics_best", best_impl_digest=digest)["assistant_index"] == 2
    assert _train_best({**valid, "assistant_messages_seen": 4}, "metrics_best", best_impl_digest=digest) == {}
    assert _train_best({**valid, "assistant_index": float("inf")}, "metrics_best", best_impl_digest=digest) == {}


def test_trusted_image_rejects_root_agent_user() -> None:
    with pytest.raises(RuntimeError, match="non-root"):
        asyncio.run(_validate_trusted_image(FakeSandbox(uid=0), ("/trusted/final_verify",)))


def test_trusted_image_rejects_symlink_tool_or_writable_parent() -> None:
    with pytest.raises(RuntimeError, match="root-owned, executable"):
        asyncio.run(
            _validate_trusted_image(
                FakeSandbox(trusted_file_type="symbolic link"),
                ("/opt/triton-agent-tools/final_verify.sh",),
            )
        )


def test_agent_verify_entrypoint_must_resolve_to_trusted_command() -> None:
    with pytest.raises(RuntimeError, match="immutable trusted command"):
        asyncio.run(
            _validate_agent_verify_entrypoint(
                FakeSandbox(verify_entrypoint_target="/workspace/agent-script.sh"),
                "/workspace",
                "tools/verify_once.sh",
                "/opt/triton-agent-tools/verify_once.sh",
            )
        )
    with pytest.raises(RuntimeError, match="trusted tool parent"):
        asyncio.run(
            _validate_trusted_image(
                FakeSandbox(trusted_parent_mode="777"),
                ("/opt/triton-agent-tools/final_verify.sh",),
            )
        )


def test_missing_impl_retry_uses_a_fresh_sandbox() -> None:
    events: list[str] = []
    sandboxes = [
        FakeSandbox(has_impl=False, events=events, label="first"),
        FakeSandbox(has_impl=True, events=events, label="second"),
    ]
    config = TritonOperatorTaskConfig(
        sandbox={"provider": "local"},
        agent={"name": "claude_code", "model": {"base_url": "http://unused", "model_name": "unused"}},
        prompt=[{"role": "user", "content": "implement"}],
        metadata={"uid": "uid-1", "op_name": "smoke", "task_code": "class Model: pass\n"},
        template_dir=None,
        verify_command="/trusted/final_verify {op_name} {workspace_dir}",
        cleanup_command="/trusted/cleanup",
        retry_on_missing_impl=True,
    )
    task = TritonOperatorTask(config)
    remaining = iter(sandboxes)
    task.build_sandbox = lambda: next(remaining)  # type: ignore[method-assign]
    task.build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    result = asyncio.run(task.run())

    assert events == ["first:enter", "first:exit", "second:enter", "second:exit"]
    assert result.reward == 0.95
    assert result.extra_info is not None
    assert result.extra_info["no_impl_retry_used"] is True
    assert result.extra_info["no_impl_retry_discard_previous"] is True


def test_zero_case_verifier_output_is_untrusted() -> None:
    sandbox = FakeSandbox(verifier_counts=(0, 0, 0))
    config = TritonOperatorTaskConfig(
        sandbox={"provider": "local"},
        agent={"name": "claude_code", "model": {"base_url": "http://unused", "model_name": "unused"}},
        prompt=[{"role": "user", "content": "implement"}],
        metadata={"uid": "uid-1", "op_name": "smoke", "task_code": "class Model: pass\n"},
        template_dir=None,
        verify_command="/trusted/final_verify {op_name} {workspace_dir}",
        cleanup_command="/trusted/cleanup",
        retry_on_missing_impl=False,
    )
    task = TritonOperatorTask(config)
    task.build_sandbox = lambda: sandbox  # type: ignore[method-assign]
    task.build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    result = asyncio.run(task.run())

    assert result.reward == 0.0
    assert result.extra_info is not None
    assert result.extra_info["metrics"]["verifier_attested"] is False
    assert result.extra_info["metrics"]["error_type"] == "untrusted_verifier_output"


def test_partial_correctness_keeps_reward_when_evaluation_completed() -> None:
    sandbox = FakeSandbox(verifier_counts=(2, 1, 1))
    config = TritonOperatorTaskConfig(
        sandbox={"provider": "local"},
        agent={"name": "claude_code", "model": {"base_url": "http://unused", "model_name": "unused"}},
        prompt=[{"role": "user", "content": "implement"}],
        metadata={"uid": "uid-1", "op_name": "smoke", "task_code": "class Model: pass\n"},
        template_dir=None,
        verify_command="/trusted/final_verify {op_name} {workspace_dir}",
        cleanup_command="/trusted/cleanup",
        retry_on_missing_impl=False,
    )
    task = TritonOperatorTask(config)
    task.build_sandbox = lambda: sandbox  # type: ignore[method-assign]
    task.build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    result = asyncio.run(task.run())

    assert result.accuracy == 0.5
    assert result.reward == 0.475
    assert result.extra_info is not None
    assert result.extra_info["metrics"]["verifier_attested"] is True


def test_nonzero_final_verifier_exit_is_infrastructure_failure() -> None:
    sandbox = FakeSandbox(verifier_counts=(2, 1, 1), verify_exit_code=3)
    config = TritonOperatorTaskConfig(
        sandbox={"provider": "local"},
        agent={"name": "claude_code", "model": {"base_url": "http://unused", "model_name": "unused"}},
        prompt=[{"role": "user", "content": "implement"}],
        metadata={"uid": "uid-1", "op_name": "smoke", "task_code": "class Model: pass\n"},
        template_dir=None,
        verify_command="/trusted/final_verify {op_name} {workspace_dir}",
        cleanup_command="/trusted/cleanup",
        retry_on_missing_impl=False,
    )
    task = TritonOperatorTask(config)
    task.build_sandbox = lambda: sandbox  # type: ignore[method-assign]
    task.build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    result = asyncio.run(task.run())

    assert result.reward == 0.0
    assert result.extra_info is not None
    assert result.extra_info["metrics"]["verifier_attested"] is False


def test_agent_tampered_reference_and_cases_are_restored_before_verify() -> None:
    sandbox = FakeSandbox(tamper_inputs=True)
    config = TritonOperatorTaskConfig(
        sandbox={"provider": "local"},
        agent={"name": "claude_code", "model": {"base_url": "http://unused", "model_name": "unused"}},
        prompt=[{"role": "user", "content": "implement"}],
        metadata={
            "uid": "uid-1",
            "op_name": "smoke",
            "task_code": "class Model: pass\n",
            "support_files": [{"name": "smoke_cases.json", "content": '{"case":1}\n'}],
        },
        template_dir=None,
        verify_command="/trusted/final_verify {op_name} {workspace_dir}",
        cleanup_command="/trusted/cleanup",
        retry_on_missing_impl=False,
    )
    task = TritonOperatorTask(config)
    task.build_sandbox = lambda: sandbox  # type: ignore[method-assign]
    task.build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    result = asyncio.run(task.run())

    assert result.reward == 0.95
    assert sandbox.verify_reference == b"class Model: pass\n"
    assert sandbox.verify_cases == b'{"case":1}\n'
    assert result.extra_info is not None
    assert result.extra_info["metrics"]["verifier_inputs_intact"] is True


def test_verifier_input_mutation_after_reset_invalidates_attestation() -> None:
    sandbox = FakeSandbox(mutate_input_after_verify=True)
    config = TritonOperatorTaskConfig(
        sandbox={"provider": "local"},
        agent={"name": "claude_code", "model": {"base_url": "http://unused", "model_name": "unused"}},
        prompt=[{"role": "user", "content": "implement"}],
        metadata={"uid": "uid-1", "op_name": "smoke", "task_code": "class Model: pass\n"},
        template_dir=None,
        verify_command="/trusted/final_verify {op_name} {workspace_dir}",
        cleanup_command="/trusted/cleanup",
        retry_on_missing_impl=False,
    )
    task = TritonOperatorTask(config)
    task.build_sandbox = lambda: sandbox  # type: ignore[method-assign]
    task.build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    result = asyncio.run(task.run())

    assert result.reward == 0.0
    assert result.extra_info is not None
    assert result.extra_info["metrics"]["verifier_inputs_intact"] is False
    assert result.extra_info["metrics"]["verifier_attested"] is False
