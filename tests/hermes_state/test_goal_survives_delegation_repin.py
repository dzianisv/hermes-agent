"""An async-delegation re-pin (#57498) ends the parent row as ``session_switch`` while a live child row
carrying ``_delegate_from`` = parent continues the same chat. That is not a user boundary: the goal must
migrate to the child, not be cancelled.

Live 2026-10-13: goals 20260923_125158_801e4460 and 20261012_231801_5f6ab9 both died with
``paused_reason = "session ended (session_switch)"`` the moment their subagent batch finished.
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


def _goal(db: SessionDB, sid: str) -> None:
    db.set_meta(f"goal:{sid}", json.dumps({"goal": "ship it", "status": "active", "source": "agent"}))


def _state(db: SessionDB, sid: str) -> dict:
    raw = db.get_meta(f"goal:{sid}")
    return json.loads(raw) if raw else {}


def _child(db: SessionDB, parent: str, child: str, marker: str) -> None:
    db.create_session(session_id=child, source="telegram", parent_session_id=parent,
                      model_config={marker: parent})


@pytest.mark.parametrize("reason", ["session_switch", "session_reset"])
def test_repin_migrates_goal_to_live_delegate_child(db, reason):
    db.create_session(session_id="parent", source="telegram")
    _goal(db, "parent")
    _child(db, "parent", "child", "_delegate_from")

    db.promote_to_session_reset("parent", reason)

    assert _state(db, "child")["status"] == "active"
    assert _state(db, "child")["goal"] == "ship it"
    assert _state(db, "parent")["status"] == "cancelled"
    assert "continued in child" in _state(db, "parent")["paused_reason"]


def test_repin_with_ended_child_still_closes_goal(db):
    db.create_session(session_id="parent", source="telegram")
    _goal(db, "parent")
    _child(db, "parent", "child", "_delegate_from")
    db.end_session("child", "agent_close")

    db.promote_to_session_reset("parent", "session_switch")

    assert _state(db, "parent")["status"] == "cancelled"
    assert _state(db, "parent")["paused_reason"] == "session ended (session_switch)"
    assert _state(db, "child") == {}


def test_real_user_boundary_without_successor_still_cancels(db):
    db.create_session(session_id="solo", source="telegram")
    _goal(db, "solo")
    db.end_session("solo", "new_session")
    assert _state(db, "solo")["status"] == "cancelled"


def test_child_goal_not_overwritten(db):
    db.create_session(session_id="parent", source="telegram")
    _goal(db, "parent")
    _child(db, "parent", "child", "_delegate_from")
    db.set_meta("goal:child", json.dumps({"goal": "newer", "status": "active", "source": "user"}))

    db.promote_to_session_reset("parent", "session_switch")

    assert _state(db, "child")["goal"] == "newer"
    assert _state(db, "parent")["status"] == "cancelled"
