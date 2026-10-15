"""A dead codex app-server's final stderr lines must reach ``stderr_tail``.

Regression for the e2e flake in
``test_crash_mid_item_surfaced_once_without_partial_content_or_orphan``: stderr
is read on a separate thread, so the session could format the crash error from
``stderr_tail()`` after the process exited but before the reader appended the
crash reason.
"""

from __future__ import annotations

import stat
import sys

import pytest

from agent.transports.codex_app_server import CodexAppServerClient


@pytest.mark.platforms("posix")  # shebang-executable fake codex binary
def test_stderr_tail_after_exit_includes_lines_still_in_the_pipe(tmp_path):
    # The fake app-server exits at once, but a child sharing its stderr pipe
    # writes the crash reason a moment later, so the line is guaranteed to
    # land in the pipe after the root process is reaped.
    fake_codex = tmp_path / "fake_codex.py"
    fake_codex.write_text(
        f"#!{sys.executable}\n"
        """
import subprocess
import sys

subprocess.Popen([
    sys.executable, "-c",
    "import sys, time; time.sleep(0.5); sys.stderr.write('fatal: CRASH-MARKER-77\\\\n')",
])
sys.exit(3)
""".lstrip()
    )
    fake_codex.chmod(fake_codex.stat().st_mode | stat.S_IXUSR)

    client = CodexAppServerClient(codex_bin=str(fake_codex))
    try:
        assert client._proc.wait(timeout=10) == 3
        assert not client.is_alive()
        assert "fatal: CRASH-MARKER-77" in client.stderr_tail()
    finally:
        client.close()
