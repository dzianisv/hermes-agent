"""End to end: the gateway's async-delegation re-pin (``SessionStore.switch_session``, #57498) must
carry the session goal onto the row the chat continues on.

Mirrors the live 2026-10-13 rows: parent ``20261012_231801_5f6ab9`` (Den's /goal) ended as
``session_switch`` when its delegate_task batch finished and the route was re-pinned to the child
``20261013_050401_906311`` (``_delegate_from`` = parent). The goal went ``cancelled`` with
``paused_reason = "session ended (session_switch)"`` and nobody resumed the work overnight.

Drives the real SessionStore + SessionDB + GoalManager (no mocks), so a regression anywhere on the
path (store → _promote_session_reset → end_session → _end_and_bump → goal_controller) fails here.
"""
from __future__ import annotations

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
import hermes_cli.goals as goals_mod
from hermes_cli.goals import GoalManager


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1916982742", chat_type="dm",
                           user_id="1916982742", thread_id="643772")
    entry = s.get_or_create_session(source)
    db = s._db_for_key(entry.session_key)
    assert db is not None, "test needs the real session DB"
    monkeypatch.setattr(goals_mod, "_get_session_db", lambda: db)
    monkeypatch.setattr(goals_mod, "draft_contract", lambda objective, **kw: None)
    return s, entry, db


def _spawn_delegate_child(db, parent_sid: str, child_sid: str) -> None:
    # tools/delegate_tool.py:277 stamps the child row with _delegate_from = parent.
    db.create_session(session_id=child_sid, source="telegram", parent_session_id=parent_sid,
                      model_config={"_delegate_from": parent_sid})


@pytest.mark.parametrize("source", ["user", "agent"])
def test_repin_to_delegate_child_keeps_goal_active(store, source):
    s, entry, db = store
    key, parent = entry.session_key, entry.session_id
    mgr = GoalManager(session_id=parent)
    if source == "user":
        mgr.set("complete agent optimizations; spawn fable-5.1 to review them")
    else:
        assert mgr.set_agent_goal("Hermes harness is healthy: goal-loop probe passes") is not None
    assert GoalManager(session_id=parent).state.status == "active"

    child = parent + "_deleg"
    _spawn_delegate_child(db, parent, child)

    # gateway/run_notifications.py:320 — CAS re-pin of the route onto the owning child.
    switched = s.switch_session(key, child, expected_session_id=parent)
    assert switched is not None and switched.session_id == child

    migrated = GoalManager(session_id=child).state
    assert migrated is not None and migrated.status == "active", \
        "goal must continue on the row the chat now lives on"
    assert migrated.source == source
    assert migrated.goal.startswith(("complete agent", "Hermes harness"))
    old = GoalManager(session_id=parent).state
    assert old.status == "cancelled" and "continued in" in (old.paused_reason or "")
    # exactly one active goal for this conversation
    assert db.get_meta(f"goal:{parent}") and db.get_meta(f"goal:{child}")
    assert sum(1 for sid in (parent, child) if GoalManager(session_id=sid).has_goal()) == 1


def test_user_new_session_still_cancels_goal(store):
    s, entry, db = store
    key, parent = entry.session_key, entry.session_id
    GoalManager(session_id=parent).set("something the user later abandons with /new")
    replacement = s.reset_session(key)
    assert replacement.session_id != parent
    assert GoalManager(session_id=parent).state.status == "cancelled"
    assert not GoalManager(session_id=replacement.session_id).has_goal()


def test_resume_of_unrelated_session_does_not_steal_goal(store):
    s, entry, db = store
    key, parent = entry.session_key, entry.session_id
    GoalManager(session_id=parent).set("goal bound to the departing conversation")
    db.create_session(session_id="archived_other", source="telegram")  # /resume target, no lineage
    assert s.switch_session(key, "archived_other") is not None
    assert GoalManager(session_id=parent).state.status == "cancelled"
    assert not GoalManager(session_id="archived_other").has_goal()
