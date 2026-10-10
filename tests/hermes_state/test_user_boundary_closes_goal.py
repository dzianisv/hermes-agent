"""A user conversation boundary (session_switch / session_reset / user_exit / new_session) must
close the session's goal; automatic ends (agent_close, compression, orphan reaps) must not —
the conversation continues under another row and the goal migrates with it.

Live 2026-10-09: 5 ``goal:*`` rows stayed ``active`` 6h–3d after their sessions ended, chased by
every goal scan (idle wake, supervisor) although nobody would ever talk to them again.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path: Path):
    handle = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield handle
    finally:
        handle.close()


def _goal(db: SessionDB, sid: str, status: str = "active") -> None:
    db.create_session(session_id=sid, source="telegram")
    db.set_meta(f"goal:{sid}", json.dumps({"goal": "ship it", "status": status, "source": "agent"}))


def _status(db: SessionDB, sid: str) -> str:
    return json.loads(db.get_meta(f"goal:{sid}"))["status"]


@pytest.mark.parametrize("reason", ["session_switch", "session_reset", "user_exit", "new_session"])
def test_user_boundary_clears_the_goal(db, reason):
    _goal(db, "s1")
    db.end_session("s1", reason)
    assert _status(db, "s1") == "cleared"
    assert json.loads(db.get_meta("goal:s1"))["paused_reason"] == f"session ended ({reason})"


@pytest.mark.parametrize("reason", ["agent_close", "compression", "ws_orphan_reap", "tui_shutdown"])
def test_automatic_end_keeps_the_goal(db, reason):
    _goal(db, "s2")
    db.end_session("s2", reason)
    assert _status(db, "s2") == "active"


def test_subagent_agent_close_clears_the_goal(db):
    db.create_session(session_id="parent", source="telegram")
    db.create_session(session_id="sub", source="subagent", parent_session_id="parent")
    db.set_meta("goal:sub", json.dumps({"goal": "child task", "status": "active", "source": "agent"}))
    db.end_session("sub", "agent_close")
    assert _status(db, "sub") == "cleared"


def test_already_done_goal_is_untouched(db):
    _goal(db, "s3", status="done")
    db.end_session("s3", "session_switch")
    assert _status(db, "s3") == "done"


def test_no_goal_row_is_fine(db):
    db.create_session(session_id="s4", source="telegram")
    db.end_session("s4", "session_switch")
    assert db.get_meta("goal:s4") is None
