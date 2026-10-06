"""Issue #3: the kanban stop guard reads the bound run's persisted disposition, not the
transcript. An attempted terminal tool call is not a handoff; a settled or superseded run
is never nudged."""

from __future__ import annotations

import pytest

from agent.kanban_stop import build_kanban_stop_nudge


def _call(name: str, cid: str = "1") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}],
    }


@pytest.fixture
def board(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    db = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.delenv("HERMES_KANBAN_STOP_NUDGE", raising=False)
    conn = kbc.connect(db_path=db)
    tid = kb.create_task(conn, title="t", assignee="worker")
    task = kb.claim_task(conn, tid, claimer="host:1")
    assert task is not None
    run_id = kb.get_task(conn, tid).current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    yield kb, conn, tid, run_id
    conn.close()


def test_rejected_terminal_call_still_nudged(board):
    _kb, _conn, tid, _run = board
    messages = [
        _call("kanban_complete"),
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1",
         "content": '{"error": "kanban_complete refused: goal gate"}'},
    ]
    nudge = build_kanban_stop_nudge(messages=messages)
    assert nudge is not None and tid in nudge


def test_terminal_call_with_missing_result_still_nudged(board):
    _kb, _conn, _tid, _run = board
    assert build_kanban_stop_nudge(messages=[_call("kanban_request_review")]) is not None


def test_valid_handoff_recorded_on_board_not_nudged(board):
    kb, conn, tid, run_id = board
    assert kb.request_review(conn, tid, summary="ready", expected_run_id=run_id)
    # Even with no tool call visible in the transcript (e.g. compressed away).
    assert build_kanban_stop_nudge(messages=[]) is None
    assert build_kanban_stop_nudge(messages=[_call("kanban_request_review")]) is None


def test_stale_worker_with_successor_not_nudged(board):
    kb, conn, tid, run_id = board
    kb.reclaim_task(conn, tid, reason="test")
    if kb.get_task(conn, tid).status != "ready":
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        conn.commit()
    assert kb.claim_task(conn, tid, claimer="host:2") is not None
    successor = kb.get_task(conn, tid).current_run_id
    assert successor and successor != run_id
    assert build_kanban_stop_nudge(messages=[]) is None
    # Read-only: the successor's run is untouched.
    assert kb.get_task(conn, tid).current_run_id == successor
    assert kb.get_run(conn, successor).status == "running"


def test_attempts_budget_still_bounds_active_run(board):
    assert build_kanban_stop_nudge(messages=[], attempts=2) is None


def test_blocked_disposition_not_nudged(board):
    kb, conn, tid, run_id = board
    assert kb.block_task(conn, tid, reason="need input", expected_run_id=run_id)
    assert build_kanban_stop_nudge(messages=[]) is None


def test_unbound_worker_falls_back_to_transcript(board, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID")
    assert build_kanban_stop_nudge(messages=[_call("kanban_complete")]) is None
    assert build_kanban_stop_nudge(messages=[]) is not None


def test_guard_never_writes_board(board):
    kb, conn, tid, run_id = board
    before = conn.execute("SELECT count(*) FROM task_events").fetchone()[0]
    build_kanban_stop_nudge(messages=[])
    assert conn.execute("SELECT count(*) FROM task_events").fetchone()[0] == before
    assert kb.get_run(conn, run_id).status == "running"
