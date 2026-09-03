from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.skipif(
    sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() != 0,
    reason="lease integration requires Linux root to create the image-contract lock fixture",
)
def test_lease_kills_background_process_group_before_lock_reuse(tmp_path: Path) -> None:
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    lock_dir.chmod(0o1777)
    lock_file = lock_dir / "device-0.lock"
    lock_file.touch()
    lock_file.chmod(0o666)
    pid_file = tmp_path / "descendant.pid"
    helper = tmp_path / "leak_descendant.py"
    helper.write_text(
        """\
import subprocess
import sys

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
open(sys.argv[1], "w", encoding="utf-8").write(str(child.pid))
""",
        encoding="utf-8",
    )
    wrapper = Path(__file__).parents[1] / "sandbox" / "tools" / "with_npu_lease.py"
    env = {
        **os.environ,
        "TRITON_EVAL_DEVICE_IDS": "0",
        "TRITON_EVAL_LOCK_DIR": str(lock_dir),
        "TRITON_EVAL_LOCK_TIMEOUT": "2",
    }

    first = subprocess.run(
        [sys.executable, str(wrapper), "--", sys.executable, str(helper), str(pid_file)],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert first.returncode == 0, first.stderr
    descendant = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2
    while Path(f"/proc/{descendant}").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not Path(f"/proc/{descendant}").exists()

    second = subprocess.run(
        [sys.executable, str(wrapper), "--", sys.executable, "-c", "pass"],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert second.returncode == 0, second.stderr
