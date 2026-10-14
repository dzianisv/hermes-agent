"""Harness row #30 recurrence (run 6854, 2026-10-13 14:47): a worker with no tracked pid kept heartbeating
(last beat 21 min before) but its claim was reclaimed because ``heartbeat_worker`` never extended
``claim_expires``. A heartbeat is a lease renewal, as in Temporal activity heartbeats.
"""

import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in ("HERMES_KANBAN_CLAIM_TTL_SECONDS", "HERMES_KANBAN_CLAIM_HEARTBEAT_MAX_STALE_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def test_heartbeat_renews_expired_claim_of_pidless_worker(board):
    tid = kb.create_task(board, title="w", assignee="worker", allow_duplicate=True)
    kb.claim_task(board, tid)
    past = int(time.time()) - 60
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (past, tid))
    assert kbd.heartbeat_worker(board, tid).ok
    assert kb.release_stale_claims(board, signal_fn=lambda *a: None) == 0
    assert kb.get_task(board, tid).status == "running"


def test_heartbeat_never_shortens_a_longer_claim(board):
    tid = kb.create_task(board, title="w", assignee="worker", allow_duplicate=True)
    kb.claim_task(board, tid)
    far = int(time.time()) + 4 * 3600
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (far, tid))
    kbd.heartbeat_worker(board, tid)
    assert board.execute("SELECT claim_expires FROM tasks WHERE id = ?", (tid,)).fetchone()[0] == far
