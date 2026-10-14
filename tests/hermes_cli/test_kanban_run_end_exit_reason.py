"""Every way a kanban run ends records ``metadata.exit_reason`` (row #3).

Runs 6329/6330/6409 were closed by ``archive_task`` with no exit_reason; the
class is "a terminal transition that closes the run without the shared helper".
Each case drives a real transition on a real temp DB, then checks EVERY closed
run on the board, so a path that closes runs on the side is caught too.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kbd, "_worker_alive", lambda *_a: False)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _no_signal(*_a, **_k):
    return None


def _claimed(conn):
    tid = kb.create_task(conn, title="t", assignee="a")
    assert kb.claim_task(conn, tid) is not None
    return tid


def _archive(conn, tid):
    kb.archive_task(conn, tid, signal_fn=_no_signal)


def _expire_lease(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET claim_expires = 1, worker_pid = NULL WHERE id = ?", (tid,))
        conn.execute("UPDATE task_runs SET claim_expires = 1 WHERE task_id = ?", (tid,))
    kb.release_stale_claims(conn)


def _crash(conn, tid):
    past = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_pid = 99991, started_at = ? WHERE id = ?", (past, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE task_id = ?", (past, tid))
    kbd._record_worker_exit(99991, 1 << 8)
    kbd.detect_crashed_workers(conn)


def _dangling_unblock(conn, tid):
    # Leaked open run on a blocked card: unblock closes it via _reclaim_dangling_run.
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'blocked', claim_lock = NULL WHERE id = ?", (tid,))
    kb.unblock_task(conn, tid)


TRANSITIONS = {
    "complete": lambda c, t: kb.complete_task(c, t, result="ok"),
    "block": lambda c, t: kb.block_task(c, t, reason="need input"),
    "manual_reclaim": lambda c, t: kb.reclaim_task(c, t, reason="op", signal_fn=_no_signal),
    "reassign": lambda c, t: kb.reassign_task(c, t, "b", reclaim_first=True),
    "archive": _archive,
    "lease_expiry": _expire_lease,
    "crash": _crash,
    "dangling_on_unblock": _dangling_unblock,
}


@pytest.mark.parametrize("name", sorted(TRANSITIONS))
def test_every_run_end_records_exit_reason(conn, name):
    tid = _claimed(conn)
    TRANSITIONS[name](conn, tid)
    runs = [r for r in kb.list_runs(conn, tid) if r.ended_at is not None]
    assert runs, f"{name}: no run was closed"
    for r in runs:
        er = (r.metadata or {}).get("exit_reason")
        assert er and er.get("reason"), f"{name}: run {r.id} ({r.outcome}) ended without exit_reason"


def test_archive_reason_names_the_archive(conn):
    tid = _claimed(conn)
    _archive(conn, tid)
    assert kb.list_runs(conn, tid)[-1].metadata["exit_reason"]["reason"] == "archived"

