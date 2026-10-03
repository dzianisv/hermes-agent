"""A live worker is never reclaimed or duplicated because its start-time fingerprint drifted.

macOS ``psutil`` create_time drifts by seconds on a healthy worker (measured: exactly the
drift-tolerance edge). Fingerprint mismatch alone then looks like PID recycle, the dispatcher
marks the task crashed, and the next tick spawns a second worker beside the one still running.
A descendant of this process cannot be a recycled stranger, and a live process whose argv
names the task id is proof a worker is still in flight.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

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


def _sleep_with_token(token: str) -> subprocess.Popen:
    """Child of this process, so it is a dispatcher descendant, with ``token`` as its own argv token."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", token],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=5)


def _drifted_fingerprint(pid: int) -> str:
    """Composed fingerprint for ``pid`` shifted far past the start-time tolerance."""
    recorded = kbd._process_fingerprint(pid)
    assert recorded is not None and "|" in recorded
    epoch, start = recorded.rsplit("|", 1)
    return f"{epoch}|{int(start) + 10_000}"


def _claim_running(conn, tid: str, pid: int, started_at) -> None:
    kb.claim_task(conn, tid)
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, worker_started_at = ?, started_at = ?, "
            "claim_expires = ? WHERE id = ?",
            (pid, started_at, old, old, tid),
        )


def _dead_pid() -> int:
    candidate = 2_147_483_647
    while candidate > 1 and kbd._pid_alive(candidate):
        candidate -= 1
    return candidate


def test_drifted_descendant_is_not_reclaimed(board):
    """A real child whose fingerprint is past the drift tolerance is still our worker.

    Old code: ``_pid_recycled`` is true, ``_worker_alive`` is false, and
    ``_reclaim_dead_workers`` releases the claim. The descendant guard must hold the claim.
    """
    conn = board
    tid = kb.create_task(conn, title="job", assignee="worker")
    proc = _sleep_with_token(tid)
    try:
        drifted = _drifted_fingerprint(proc.pid)
        assert kbd._pid_recycled(proc.pid, drifted) is True
        assert kbd._is_dispatcher_descendant(proc.pid) is True
        assert kbd._worker_alive(proc.pid, drifted) is True
        _claim_running(conn, tid, proc.pid, drifted)

        sweep = kbd._reclaim_dead_workers(conn)
        task = kb.get_task(conn, tid)
        assert tid not in sweep.crashed
        assert task.status == "running"
        assert task.worker_pid == proc.pid
        assert task.claim_lock
    finally:
        _stop(proc)


def test_dispatch_refuses_spawn_while_live_worker_exists(board, all_assignees_spawnable):
    """A ready task whose id is already a live argv token must not be spawned again."""
    conn = board
    tid = kb.create_task(conn, title="job", assignee="worker")
    proc = _sleep_with_token(tid)
    calls = {"n": 0}

    def spawn_fn(*_args, **_kwargs):
        calls["n"] += 1
        return 4242

    try:
        result = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        assert calls["n"] == 0
        assert not any(row[0] == tid for row in result.spawned)
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.claim_lock is None
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None
        events = [e for e in kb.list_events(conn, tid) if e.kind == "spawn_refused_live_worker"]
        assert events
        assert proc.pid in events[-1].payload["pids"]
    finally:
        _stop(proc)


def test_dead_worker_still_reclaimed_and_prefix_id_does_not_match(board):
    """No live worker -> reclaim proceeds. ``t_abc`` must not match a ``t_abcd`` token."""
    conn = board
    tid = kb.create_task(conn, title="job", assignee="worker")
    prefix_proc = _sleep_with_token(tid + "x")
    try:
        assert prefix_proc.pid not in kbd._live_worker_pids_for_task(tid)
        assert kbd._argv_has_exact_task_token([tid + "x"], tid) is False
        assert kbd._argv_has_exact_task_token(["work kanban task " + tid], tid) is True
        _claim_running(conn, tid, _dead_pid(), "1|1")

        sweep = kbd._reclaim_dead_workers(conn)
        task = kb.get_task(conn, tid)
        assert tid in sweep.crashed
        assert task.status == "ready"
        assert task.worker_pid is None
    finally:
        _stop(prefix_proc)
