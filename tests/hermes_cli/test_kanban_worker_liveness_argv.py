"""A live kanban worker is never reclaimed and duplicated because its start-time fingerprint drifted.

The spawn-time fingerprint is the first identity; when it no longer matches a live pid, the
worker's own ``-q work kanban task <id>`` argv token is the second. Only a readable cmdline WITHOUT
the token makes the pid foreign; an unreadable one holds the claim (and is never signalled).
"""

import json
import logging
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _claimed_running(conn, *, pid: int, started_at) -> str:
    tid = kb.create_task(conn, title="job", assignee="worker")
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = ?, started_at = ? WHERE id = ?",
                     (started_at, old, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))
    return tid


def _drift_start_time(monkeypatch, delta: int) -> None:
    from gateway import status

    real = status.get_process_start_time
    monkeypatch.setattr(status, "get_process_start_time",
                        lambda pid: (lambda s: None if s is None else s + delta)(real(pid)))


def _cmdline(monkeypatch, value) -> None:
    from gateway import status

    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: value)


def _worker_cmdline(tid: str) -> str:
    return f"/x/python -m hermes_cli.main -p software-engineer chat -q work kanban task {tid}"


def _events(conn, tid, kind):
    return [e for e in kb.list_events(conn, tid) if e.kind == kind]


def _live_fp_running(conn):
    pid = os.getpid()
    fp = kbd._process_fingerprint(pid)
    assert fp is not None and "|" in fp
    return pid, fp, _claimed_running(conn, pid=pid, started_at=fp)


def test_drift_beyond_gateway_tolerance_keeps_worker_and_spawns_nothing(board, monkeypatch, all_assignees_spawnable):
    conn = board
    pid, _fp, tid = _live_fp_running(conn)
    _drift_start_time(monkeypatch, 300)
    _cmdline(monkeypatch, _worker_cmdline(tid))

    sweep = kbd._reclaim_dead_workers(conn)
    assert tid not in sweep.crashed
    assert kb.get_task(conn, tid).status == "running"
    assert not _events(conn, tid, "crashed")

    spawned: list = []
    kbd.dispatch_once(conn, spawn_fn=lambda task, ws, board=None: spawned.append(task.id) or 99999)
    assert tid not in spawned
    task = kb.get_task(conn, tid)
    assert task.status == "running" and task.worker_pid == pid


def test_far_drift_with_argv_identity_is_held_once_per_run(board, monkeypatch):
    conn = board
    _pid, _fp, tid = _live_fp_running(conn)
    _drift_start_time(monkeypatch, 1000)
    _cmdline(monkeypatch, _worker_cmdline(tid))

    kbd._reclaim_dead_workers(conn)
    kbd._reclaim_dead_workers(conn)
    assert kb.get_task(conn, tid).status == "running"
    held = _events(conn, tid, "worker_held")
    assert len(held) == 1
    run_id = conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
    assert held[0].run_id == run_id
    assert held[0].payload["liveness"]["argv_match"] is True
    assert held[0].payload["liveness"]["alive"] is True


@pytest.mark.parametrize("cmdline", ["{worker}9", "/usr/sbin/sshd -D"])
def test_far_drift_without_token_is_reclaimed_with_evidence_and_never_signalled(board, monkeypatch, cmdline):
    conn = board
    pid, fp, tid = _live_fp_running(conn)
    lock = conn.execute("SELECT claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
    _drift_start_time(monkeypatch, 1000)
    _cmdline(monkeypatch, cmdline.format(worker=_worker_cmdline(tid)))

    sweep = kbd._reclaim_dead_workers(conn)
    assert tid in sweep.crashed
    assert kb.get_task(conn, tid).status != "running"
    crashed = _events(conn, tid, "crashed")
    assert len(crashed) == 1
    liveness = crashed[0].payload["liveness"]
    assert liveness["fp_match"] is False and liveness["argv_match"] is False
    assert liveness["fp_recorded"] == fp and liveness["fp_current"] and liveness["fp_current"] != fp
    assert liveness["alive"] is False
    meta = conn.execute("SELECT metadata FROM task_runs WHERE id = ?", (crashed[0].run_id,)).fetchone()[0]
    assert json.loads(meta)["liveness"] == liveness

    rec: list = []
    info = kbd._terminate_reclaimed_worker(pid, lock, signal_fn=lambda p, s: rec.append((p, s)),
                                           started_at=fp, task_id=tid)
    assert rec == []
    assert info["terminated"] is True and info["pid_recycled"] is True


def test_unreadable_identity_is_held_and_never_signalled(board, monkeypatch, caplog):
    from gateway import status

    conn = board
    pid, fp, tid = _live_fp_running(conn)
    lock = conn.execute("SELECT claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: None)
    _cmdline(monkeypatch, None)

    with caplog.at_level(logging.WARNING):
        sweep = kbd._reclaim_dead_workers(conn)
    assert tid not in sweep.crashed
    assert kb.get_task(conn, tid).status == "running"
    held = _events(conn, tid, "worker_held")
    assert len(held) == 1 and held[0].payload["reason"] == "identity_unreadable_held"
    assert held[0].payload["liveness"]["fp_match"] is None
    assert any("holding" in r.getMessage() and tid in r.getMessage() for r in caplog.records
               if r.levelno == logging.WARNING)

    rec: list = []
    info = kbd._terminate_reclaimed_worker(pid, lock, signal_fn=lambda p, s: rec.append((p, s)),
                                           started_at=fp, task_id=tid)
    assert rec == []
    assert info["terminated"] is False
    assert kbd._worker_survived_termination(info) is True


@pytest.mark.platforms("posix")
def test_pid_alive_without_psutil_treats_permission_error_as_alive(monkeypatch):
    def denied(pid, sig):
        raise PermissionError

    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(os, "kill", denied)
    assert kb._pid_alive(os.getpid()) is True


def test_argv_token_is_matched_as_whole_tokens(monkeypatch):
    long_id, short_id = "t_007efb64", "t_00"
    _cmdline(monkeypatch, _worker_cmdline(long_id))
    assert kbd._pid_carries_task(os.getpid(), short_id) is False
    assert kbd._pid_carries_task(os.getpid(), long_id) is True
    # Single argv element (one ``-q`` value) and ``ps``-style joined forms are the same string shape.
    _cmdline(monkeypatch, kbd.worker_task_argv_token(long_id))
    assert kbd._pid_carries_task(os.getpid(), long_id) is True
    _cmdline(monkeypatch, "hermes chat -q   work  kanban task t_007efb64 --x")
    assert kbd._pid_carries_task(os.getpid(), long_id) is True
    _cmdline(monkeypatch, None)
    assert kbd._pid_carries_task(os.getpid(), long_id) is None


def test_worker_argv_carries_the_matchable_token(monkeypatch):
    task = SimpleNamespace(id="t_abc", skills=None, model_override=None,
                           provider_override=None, reasoning_effort=None)
    monkeypatch.setattr(kbd, "_resolve_worker_cli_toolsets", lambda home: [])
    argv = kbd._worker_argv(task, "worker", None)
    _cmdline(monkeypatch, " ".join(argv))
    assert kbd._pid_carries_task(os.getpid(), "t_abc") is True


@pytest.mark.platforms("posix")
def test_real_worker_process_drift_then_exit(board, monkeypatch):
    conn = board
    tid = kb.create_task(conn, title="job", assignee="worker")
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)", "-q",
                             kbd.worker_task_argv_token(tid)])
    try:
        kb.claim_task(conn, tid)
        real_fp = None
        for _ in range(50):
            real_fp = kbd._process_fingerprint(proc.pid)
            if real_fp:
                break
            time.sleep(0.05)
        assert real_fp and "|" in real_fp
        epoch, start = real_fp.rsplit("|", 1)
        kbd._set_worker_pid(conn, tid, proc.pid)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET worker_started_at = ? WHERE id = ?",
                         (f"{epoch}|{int(start) + 300}", tid))
        monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

        assert tid not in kbd._reclaim_dead_workers(conn).crashed
        assert kb.get_task(conn, tid).status == "running"

        proc.terminate()
        proc.wait(timeout=10)
        sweep = kbd._reclaim_dead_workers(conn)
        assert tid in sweep.crashed + sweep.rate_limited
        assert kb.get_task(conn, tid).status != "running"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


@pytest.mark.parametrize("reopened, expected", [(True, None), (False, "recent_success")])
def test_done_reopened_after_recent_success_releases_the_guard(board, reopened, expected):
    conn = board
    tid = kb.create_task(conn, title="job", assignee="worker")
    ended = int(time.time()) + 5  # after every creation event
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (?, 'worker', 'completed', 'completed', ?, ?)", (tid, ended - 60, ended))
        if reopened:
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'done_reopened', ?, ?)",
                (tid, json.dumps({"status": "ready"}), ended + 1))
    assert kbd.check_respawn_guard(conn, tid) == expected
