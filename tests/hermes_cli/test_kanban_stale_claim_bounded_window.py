"""Harness gap #25: stale-lock reclaims wasted 1-4h per wedged/dead worker.

Live shape (kanban.db, runs 6685/6693/6795/6801...): host-local worker, PID still alive,
``last_heartbeat_at`` frozen at ``claim_expires - 900`` (the last heartbeat that renewed the claim),
then ``claim_extended`` (reason ``pid_alive``) on every pass until the fixed 1h staleness backstop
finally fired — 49-253 min per run. The window is now 2x TTL (30 min, configurable), and a
host-local PID that is gone is reclaimed on the next pass with ``exit_reason.reason = pid_dead``.
"""

import json
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


def _live_shape(conn, *, hb_age: int, pid: int = 4498, heartbeat: bool = True) -> str:
    """Running host-local card: claim expired ``hb_age - 900`` s ago, heartbeat at claim time."""
    tid = kb.create_task(conn, title="wedged", assignee="worker", allow_duplicate=True)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    now = int(time.time())
    hb = now - hb_age
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_expires = ?, last_heartbeat_at = ?, started_at = ? WHERE id = ?",
            (hb + kb.DEFAULT_CLAIM_TTL_SECONDS, hb if heartbeat else None, hb, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = "
                     "(SELECT current_run_id FROM tasks WHERE id = ?)", (hb, tid))
    return tid


def _reclaim_payload(conn, tid):
    row = conn.execute("SELECT payload FROM task_events WHERE task_id = ? AND kind = 'reclaimed'",
                       (tid,)).fetchone()
    return None if row is None else json.loads(row["payload"])


def _exit_reason(conn, tid):
    row = conn.execute("SELECT metadata FROM task_runs WHERE task_id = ? AND outcome = 'reclaimed'",
                       (tid,)).fetchone()
    return json.loads(row["metadata"])["exit_reason"]


def test_live_pid_silent_past_two_ttl_is_reclaimed_not_extended(board, monkeypatch):
    """35 min without a heartbeat (run 6687 / 6799 / 6801 shape: heartbeat_stale=false at 1h) —
    previously ``claim_extended`` again; now reclaimed with reason ``heartbeat_stale``."""
    monkeypatch.setattr(kb, "_worker_alive", lambda _pid, _s: True)
    tid = _live_shape(board, hb_age=35 * 60)
    assert kb.release_stale_claims(board, signal_fn=lambda *_a: None) == 1
    assert kb.get_task(board, tid).status == "ready"
    assert "claim_extended" not in [e.kind for e in kb.list_events(board, tid)]
    assert _reclaim_payload(board, tid)["heartbeat_stale"] is True
    assert _exit_reason(board, tid)["reason"] == "heartbeat_stale"


def test_host_local_dead_pid_reclaimed_next_pass_with_pid_dead(board, monkeypatch):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    tid = _live_shape(board, hb_age=16 * 60)
    assert kb.release_stale_claims(board, signal_fn=lambda *_a: None) == 1
    assert kb.get_task(board, tid).status == "ready"
    assert _reclaim_payload(board, tid)["pid_dead"] is True
    assert _exit_reason(board, tid)["reason"] == "pid_dead"


def test_active_worker_within_window_is_still_extended(board, monkeypatch):
    """Existing semantics: a live worker that heartbeated within 2x TTL keeps its claim."""
    monkeypatch.setattr(kb, "_worker_alive", lambda _pid, _s: True)
    tid = _live_shape(board, hb_age=20 * 60)
    assert kb.release_stale_claims(board, signal_fn=lambda *_a: None) == 0
    assert kb.get_task(board, tid).status == "running"
    assert "claim_extended" in [e.kind for e in kb.list_events(board, tid)]


def test_window_is_configurable(board, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_HEARTBEAT_MAX_STALE_SECONDS", str(2 * 3600))
    monkeypatch.setattr(kb, "_worker_alive", lambda _pid, _s: True)
    tid = _live_shape(board, hb_age=35 * 60)
    assert kb.release_stale_claims(board, signal_fn=lambda *_a: None) == 0
    assert kb.get_task(board, tid).status == "running"
