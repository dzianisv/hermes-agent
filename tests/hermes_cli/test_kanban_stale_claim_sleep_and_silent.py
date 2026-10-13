"""Harness gap #30b: a live worker that NEVER heartbeated was extended until max-runtime; its silence now counts from
claim/start time. Reclaim must never signal the process running the reclaim (os.getpid()).
"""

import os
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    for var in ("HERMES_KANBAN_CLAIM_TTL_SECONDS", "HERMES_KANBAN_CLAIM_HEARTBEAT_MAX_STALE_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _running(conn, *, age: int, heartbeat: bool, pid: int = 4498) -> str:
    tid = kb.create_task(conn, title="w", assignee="worker", allow_duplicate=True)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    t0 = int(time.time()) - age
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET claim_expires = ?, last_heartbeat_at = ?, started_at = ? WHERE id = ?",
                     (t0 + kb.DEFAULT_CLAIM_TTL_SECONDS, t0 if heartbeat else None, t0, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = "
                     "(SELECT current_run_id FROM tasks WHERE id = ?)", (t0, tid))
    return tid


def _kinds(conn, tid):
    return [e.kind for e in kb.list_events(conn, tid)]


def test_awake_silence_past_window_still_reclaimed(board, monkeypatch):
    monkeypatch.setattr(kb, "_worker_alive", lambda _p, _s: True)
    tid = _running(board, age=35 * 60, heartbeat=True)
    assert kb.release_stale_claims(board, signal_fn=lambda *a: None) == 1
    assert kb.get_task(board, tid).status == "ready"


def test_never_heartbeated_live_worker_reclaimed_after_window(board, monkeypatch):
    monkeypatch.setattr(kb, "_worker_alive", lambda _p, _s: True)
    tid = _running(board, age=35 * 60, heartbeat=False)
    assert kb.release_stale_claims(board, signal_fn=lambda *a: None) == 1
    assert kb.get_task(board, tid).status == "ready"
    assert "claim_extended" not in _kinds(board, tid)


def test_never_heartbeated_live_worker_within_window_extended(board, monkeypatch):
    monkeypatch.setattr(kb, "_worker_alive", lambda _p, _s: True)
    tid = _running(board, age=20 * 60, heartbeat=False)
    assert kb.release_stale_claims(board, signal_fn=lambda *a: None) == 0
    assert "claim_extended" in _kinds(board, tid)


def test_reclaim_never_signals_own_process(board, monkeypatch):
    """Default signal path with pid == os.getpid(): refused, claim held, runner not SIGTERM'd."""
    sent = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(os, "killpg", lambda pg, sig: sent.append((pg, sig)))
    tid = _running(board, age=35 * 60, heartbeat=False, pid=os.getpid())
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET worker_started_at = NULL WHERE id = ?", (tid,))
    assert kb.release_stale_claims(board) == 0
    assert [s for s in sent if s[1] != 0] == []
    assert kb.get_task(board, tid).status == "running"
    assert "reclaim_deferred" in _kinds(board, tid)
