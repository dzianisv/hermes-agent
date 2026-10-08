"""A gateway restart must not kill kanban workers mid-task.

Behaviour test, not source reading. It uses a real temp HERMES_HOME, a real kanban DB, and a real
"gateway" process that spawns its worker with the production ``_default_spawn``. The worker is a fake
``hermes`` binary. The test then does what launchd ``kickstart -k`` does to a job: it signals the
gateway's whole process group, SIGTERM and then SIGKILL. The worker must still be alive. A fresh
dispatcher pass must re-adopt it and leave it alone. Once the worker exits, the same pass must close the run.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("posix")

FAKE_HERMES = textwrap.dedent("""\
    #!{python}
    import os, sys, time, pathlib
    d = pathlib.Path({gate!r})
    (d / "worker.pid").write_text(str(os.getpid()))
    while not (d / "finish").exists():
        time.sleep(0.05)
    sys.exit(0)
""")

GATEWAY = textwrap.dedent("""\
    import json, sys, time
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    conn = kbc.connect()
    tid = sys.argv[1]
    task = kb.claim_task(conn, tid)
    pid = kbd._default_spawn(task, sys.argv[2])
    kbd._set_worker_pid(conn, tid, pid)
    print(json.dumps({"pid": pid}), flush=True)
    time.sleep(600)
""")


def _wait(pred, timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A reaped-elsewhere zombie still answers kill(0); ask ps.
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout
    return bool(out.strip()) and "Z" not in out


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / "hh"
    (root / "profiles" / "coder").mkdir(parents=True)
    (root / "config.yaml").write_text("{}\n")
    (root / "profiles" / "coder" / "config.yaml").write_text("{}\n")
    gate = tmp_path / "gate"
    gate.mkdir()
    fake = tmp_path / "fake-hermes"
    fake.write_text(FAKE_HERMES.format(python=sys.executable, gate=str(gate)))
    fake.chmod(0o755)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_BIN", str(fake))
    # No launch grace: the re-adoption check must hold on liveness alone, not on a timer.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kbc.init_db()
    conn = kbc.connect()
    tid = kb.create_task(conn, title="long job", assignee="coder")
    conn.close()
    ws = tmp_path / "ws"
    ws.mkdir()
    return {"root": root, "gate": gate, "tid": str(tid), "ws": ws}


def _start_gateway(home) -> tuple[subprocess.Popen, int]:
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo))
    gw = subprocess.Popen(
        [sys.executable, "-c", GATEWAY, home["tid"], str(home["ws"])],
        stdout=subprocess.PIPE, text=True, env=env, cwd=str(repo),
        start_new_session=True,  # its own group, as a launchd job has
    )
    line = gw.stdout.readline()
    assert line, "fake gateway failed to spawn a worker"
    pid = json.loads(line)["pid"]
    assert _wait(lambda: (home["gate"] / "worker.pid").exists())
    return gw, pid


def test_kickstart_style_group_kill_of_gateway_leaves_worker_running(home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli.gateway_kanban_workers import open_host_worker_runs

    gw, wpid = _start_gateway(home)
    try:
        assert os.getpgid(wpid) != os.getpgid(gw.pid), "worker shares the gateway's process group"
        assert os.getsid(wpid) != os.getsid(gw.pid), "worker shares the gateway's session"

        # launchd kickstart -k: SIGTERM the job's group, then SIGKILL it.
        os.killpg(gw.pid, signal.SIGTERM)
        try:
            os.killpg(gw.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass  # group already gone (macOS answers EPERM for a zombie-only group)
        gw.wait(timeout=10)
        time.sleep(0.3)
        assert _alive(wpid), "worker died with the gateway"

        # The new gateway's dispatcher re-adopts the worker: the run stays open and the card stays running.
        conn = kbc.connect()
        assert kbd.detect_crashed_workers(conn) == []
        assert conn.execute("SELECT status FROM tasks WHERE id=?", (home["tid"],)).fetchone()[0] == "running"
        runs = open_host_worker_runs()
        assert [(r.task_id, r.worker_pid, r.alive) for r in runs] == [(home["tid"], wpid, True)]

        # The worker finishes on its own. The next pass sees it gone and closes the run.
        (home["gate"] / "finish").write_text("")
        assert _wait(lambda: not _alive(wpid))
        kbd.detect_crashed_workers(conn)
        row = conn.execute(
            "SELECT ended_at FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (home["tid"],)
        ).fetchone()
        assert row[0] is not None
        assert open_host_worker_runs() == []
        conn.close()
    finally:
        (home["gate"] / "finish").write_text("")
        if gw.poll() is None:
            gw.kill()
            gw.wait(timeout=10)


def test_restart_announces_running_workers(home, capsys):
    from hermes_cli.gateway_kanban_workers import announce_surviving_workers

    gw, wpid = _start_gateway(home)
    try:
        runs = announce_surviving_workers("restart")
        out = capsys.readouterr().out
        assert [r.worker_pid for r in runs] == [wpid]
        assert home["tid"] in out and f"pid {wpid}" in out and "keep running" in out
    finally:
        (home["gate"] / "finish").write_text("")
        gw.kill()
        gw.wait(timeout=10)


def test_no_running_workers_prints_nothing(home, capsys):
    from hermes_cli.gateway_kanban_workers import announce_surviving_workers

    assert announce_surviving_workers("stop") == []
    assert capsys.readouterr().out == ""
