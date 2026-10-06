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


# --- Unknown identity = no signal AND no release (review follow-ups) ---------------------------------

def _spawn_sleeper(token_task_id: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)", "-q", kbd.worker_task_argv_token(token_task_id)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _real_fingerprint(pid: int) -> str:
    fp = None
    for _ in range(100):
        fp = kbd._process_fingerprint(pid)
        if fp:
            break
        time.sleep(0.05)
    assert fp and "|" in fp
    return fp


def _shifted(fp: str, delta: int) -> str:
    epoch, start = fp.rsplit("|", 1)
    return f"{epoch}|{int(start) + delta}"


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


@pytest.mark.platforms("posix")
def test_real_child_far_drift_identity_comes_from_its_argv(board):
    conn = board
    tid = kb.create_task(conn, title="job", assignee="worker")
    ours = _spawn_sleeper(tid)
    other_tid = kb.create_task(conn, title="other", assignee="worker")
    foreign = _spawn_sleeper("t_someone_else")
    try:
        for task_id, proc in ((tid, ours), (other_tid, foreign)):
            kb.claim_task(conn, task_id)
            kbd._set_worker_pid(conn, task_id, proc.pid)
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET worker_started_at = ? WHERE id = ?",
                             (_shifted(_real_fingerprint(proc.pid), 1000), task_id))

        sweep = kbd._reclaim_dead_workers(conn)

        assert tid not in sweep.crashed
        assert kb.get_task(conn, tid).status == "running"
        assert not _events(conn, tid, "crashed")
        assert other_tid in sweep.crashed
        assert kb.get_task(conn, other_tid).status != "running"
    finally:
        _stop(ours)
        _stop(foreign)


def _signal_recorder(on_sigterm=None):
    import signal as _signal

    rec: list = []

    def fn(pid, sig):
        rec.append((pid, sig))
        if sig == _signal.SIGTERM and on_sigterm is not None:
            on_sigterm()

    return rec, fn


def _identity_goes_unreadable(monkeypatch):
    from gateway import status

    def flip():
        monkeypatch.setattr(status, "get_process_start_time", lambda pid: None)
        monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: None)

    return flip


def _sigkill_sent(rec) -> bool:
    import signal as _signal

    return any(sig == getattr(_signal, "SIGKILL", None) for _pid, sig in rec)


def test_sigkill_escalation_refused_once_identity_is_unreadable(board, monkeypatch):
    import signal as _signal

    conn = board
    pid, fp, tid = _live_fp_running(conn)
    lock = conn.execute("SELECT claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
    monkeypatch.setattr(kbd, "_poll_worker_exit", lambda *a, **k: False)  # SIGTERM ignored
    rec, fn = _signal_recorder(on_sigterm=_identity_goes_unreadable(monkeypatch))

    info = kbd._terminate_reclaimed_worker(pid, lock, signal_fn=fn, started_at=fp, task_id=tid)

    assert any(sig == _signal.SIGTERM for _p, sig in rec)
    assert not _sigkill_sent(rec)
    assert info["terminated"] is False and info["signal_refused"] is True
    assert kbd._worker_survived_termination(info) is True


@pytest.mark.parametrize("identity_unreadable_after_sigterm", [False, True])
def test_max_runtime_never_releases_a_claim_beside_a_surviving_worker(
        board, monkeypatch, identity_unreadable_after_sigterm):
    conn = board
    pid, _fp, tid = _live_fp_running(conn)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET max_runtime_seconds = 1 WHERE id = ?", (tid,))
    lock = conn.execute("SELECT claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
    monkeypatch.setattr(kbd, "_poll_worker_exit", lambda *a, **k: False)
    rec, fn = _signal_recorder(
        on_sigterm=_identity_goes_unreadable(monkeypatch) if identity_unreadable_after_sigterm else None)

    assert kbd.enforce_max_runtime(conn, signal_fn=fn) == []

    assert rec, "the over-runtime worker is signalled while its identity is confirmed"
    assert _sigkill_sent(rec) is (not identity_unreadable_after_sigterm)
    task = kb.get_task(conn, tid)
    assert task.status == "running" and task.worker_pid == pid
    assert conn.execute("SELECT claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()[0] == lock
    assert not _events(conn, tid, "timed_out")


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("successor_owns_pid", [True, False])
def test_terminal_reaper_identity_is_run_scoped_not_task_scoped(board, successor_owns_pid):
    conn = board
    tid = kb.create_task(conn, title="job", assignee="worker")
    proc = _spawn_sleeper(tid)  # argv carries the TASK token — it cannot name the closed run
    try:
        kb.claim_task(conn, tid, claimer=kb._claimer_id())
        run_b = kb._current_run_id(conn, tid)
        lock_b = conn.execute("SELECT claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
        kbd._set_worker_pid(conn, tid, proc.pid if successor_owns_pid else os.getpid())
        stale_fp = _shifted(_real_fingerprint(proc.pid), 1000)
        ended = int(time.time()) - 3600
        with kb.write_txn(conn):
            run_a = conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, claim_lock, worker_pid, "
                "worker_started_at, started_at, ended_at) VALUES (?, 'worker', 'reclaimed', 'reclaimed', "
                "?, ?, ?, ?, ?)",
                (tid, lock_b, proc.pid, stale_fp, ended - 60, ended)).lastrowid
        rec, fn = _signal_recorder()

        assert kbd.reap_terminal_workers(conn, signal_fn=fn) == []

        assert rec == []
        assert proc.poll() is None
        task = kb.get_task(conn, tid)
        assert task.status == "running" and task.current_run_id == run_b
        assert conn.execute("SELECT claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()[0] == lock_b
        evidence = conn.execute("SELECT worker_pid FROM task_runs WHERE id = ?", (run_a,)).fetchone()[0]
        # A successor's pid keeps its evidence; otherwise the fingerprint mismatch makes it foreign.
        assert evidence == (proc.pid if successor_owns_pid else None)
    finally:
        _stop(proc)


def test_manual_reclaim_keeps_claim_when_identity_unreadable(board, monkeypatch):
    conn = board
    pid, _fp, tid = _live_fp_running(conn)
    lock = conn.execute("SELECT claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
    _drift_start_time(monkeypatch, 1000)
    _cmdline(monkeypatch, None)
    rec, fn = _signal_recorder()

    assert kb.reclaim_task(conn, tid, reason="op", signal_fn=fn) is False

    assert rec == []
    task = kb.get_task(conn, tid)
    assert task.status == "running" and task.worker_pid == pid
    assert conn.execute("SELECT claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()[0] == lock


def test_manual_reclaim_signals_drifted_worker_proven_by_argv(board, monkeypatch):
    import signal as _signal

    conn = board
    pid, _fp, tid = _live_fp_running(conn)
    _drift_start_time(monkeypatch, 1000)
    _cmdline(monkeypatch, _worker_cmdline(tid))
    monkeypatch.setattr(kbd, "_poll_worker_exit", lambda *a, **k: True)  # exits on SIGTERM
    rec, fn = _signal_recorder()

    assert kb.reclaim_task(conn, tid, reason="op", signal_fn=fn) is True

    assert any(sig == _signal.SIGTERM for _p, sig in rec)
    assert kb.get_task(conn, tid).status != "running"


def test_dashboard_status_change_terminates_with_task_id(tmp_path, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    import importlib.util
    from pathlib import Path

    from fastapi.testclient import TestClient

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    plugin_file = Path(__file__).resolve().parents[2] / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_dashboard_plugin_kanban_liveness_test", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    app = fastapi.FastAPI()
    app.include_router(mod.router, prefix="/api/plugins/kanban")

    with kbc.connect() as c:
        tid = kb.create_task(c, title="job", assignee="worker")
        kb.claim_task(c, tid)
        kbd._set_worker_pid(c, tid, 424242)
    calls: list = []
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker",
                        lambda pid, lock, **kw: calls.append((pid, kw)) or {"terminated": True})

    r = TestClient(app).patch(f"/api/plugins/kanban/tasks/{tid}", json={"status": "ready"})

    assert r.status_code == 200, r.text
    assert calls and calls[0][0] == 424242 and calls[0][1]["task_id"] == tid


@pytest.mark.platforms("macos")
def test_pid_alive_keeps_existence_answer_when_ps_fails(monkeypatch):
    def ps(argv, **kw):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", ps)
    assert kbd._pid_alive(os.getpid()) is True

    monkeypatch.setattr(subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout="Z+\n", stderr=""))
    assert kbd._pid_alive(os.getpid()) is False
