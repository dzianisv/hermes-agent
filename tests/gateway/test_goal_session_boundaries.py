"""Row 21: caller intent, not newest-child inference, owns goal movement."""
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from hermes_state import AsyncSessionDB
from hermes_cli import goals
from run_agent import AIAgent


@pytest.fixture
def lane(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    db = store._db
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", thread_id="7")
    entry = store.get_or_create_session(source)
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._session_db = AsyncSessionDB(db)
    runner._telegram_topic_profile_name = lambda s: "default"
    runner._sync_telegram_topic_binding = Mock()
    yield runner, store, db, source, entry


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
async def test_delegation_repin_carries_goal_unless_successor_owns_one(lane, existing):
    runner, store, db, source, old = lane
    db.create_session("target", source="telegram")
    goals.GoalManager(old.session_id).set("finish original work")
    if existing:
        goals.GoalManager("target").set("successor work")
    result = await runner._resolve_async_delegation_session(old, "target")
    assert result.session_id == "target"
    assert goals.load_goal("target").goal == ("successor work" if existing else "finish original work")
    assert goals.load_goal(old.session_id).status == ("active" if existing else "cleared")


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
async def test_compression_tip_carries_goal(lane, existing):
    runner, store, db, source, old = lane
    db.end_session(old.session_id, "compression")
    db.create_session("tip", source="telegram", parent_session_id=old.session_id)
    db.bind_telegram_topic(chat_id="42", thread_id="7", user_id="42",
                           session_key=old.session_key, session_id=old.session_id)
    goals.GoalManager(old.session_id).set("original work")
    if existing:
        goals.GoalManager("tip").set("tip work")
    result = await runner._hmwa_heal_telegram_topic_binding(source, old, old.session_key)
    assert result.session_id == "tip"
    assert goals.load_goal("tip").goal == ("tip work" if existing else "original work")
    assert goals.load_goal(old.session_id).status == ("active" if existing else "cleared")


def test_new_clears_without_carrying(lane):
    runner, store, db, source, old = lane
    goals.GoalManager(old.session_id).set("old goal")
    fresh = store.reset_session(old.session_key)
    assert goals.load_goal(old.session_id).status == "cleared"
    assert goals.load_goal(fresh.session_id) is None

@pytest.mark.asyncio
async def test_unrelated_resume_does_not_carry_to_target_or_live_children(lane):
    from gateway.platforms.event import MessageEvent
    runner, store, db, source, old = lane
    db.create_session("resume-target", source="telegram", user_id="42", chat_id="42")
    db.create_session("other-child", source="telegram", parent_session_id=old.session_id)
    goals.GoalManager(old.session_id).set("old goal")
    runner._normalize_source_for_session_key = lambda s: s
    runner._session_key_for_source = lambda s: old.session_key
    runner._resolve_resume_target = AsyncMock(return_value=("resume-target", "Target"))
    runner._resume_access_denied_reply = AsyncMock(return_value=None)
    runner._release_running_agent_state = Mock()
    runner._clear_conversation_scope = Mock()
    runner._evict_cached_agent = Mock()
    await runner._handle_resume_command(MessageEvent(text="/resume Target", source=source))
    assert goals.load_goal(old.session_id).status == "cleared"
    assert goals.load_goal("resume-target") is None
    assert goals.load_goal("other-child") is None

@pytest.mark.parametrize("source,end_on_close", [("subagent", True), ("telegram", True), ("subagent", False)])
def test_terminal_delegate_clears_only_owned_child(lane, source, end_on_close):
    runner, store, db, src, parent = lane
    db.create_session("child", source=source, parent_session_id=parent.session_id)
    goals.GoalManager(parent.session_id).set("parent work")
    goals.GoalManager("child").set("child work")
    agent = object.__new__(AIAgent)
    agent._session_db = db
    agent.session_id = "child"
    agent._owns_session_db = False
    agent._end_session_on_close = end_on_close
    agent._finalize_owned_session_row()
    assert goals.load_goal("child").status == ("cleared" if source == "subagent" and end_on_close else "active")
    assert goals.load_goal(parent.session_id).status == "active"

@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["delegation", "compression"])
async def test_refused_cas_leaves_goals_untouched(lane, path):
    runner, store, db, source, old = lane
    db.create_session("target", source="telegram")
    goals.GoalManager(old.session_id).set("old goal")
    # A real intervening route switch makes the old entry's CAS fail.
    db.create_session("concurrent", source="telegram")
    store.switch_session(old.session_key, "concurrent")
    if path == "delegation":
        result = await runner._resolve_async_delegation_session(old, "target")
        assert result is None
    else:
        db.bind_telegram_topic(chat_id="42", thread_id="7", user_id="42",
                               session_key=old.session_key, session_id="target")
        await runner._hmwa_heal_telegram_topic_binding(source, old, old.session_key)
    assert goals.load_goal(old.session_id).status == "active"
    assert goals.load_goal("target") is None


@pytest.mark.parametrize("status,expected,migrates", [
    ("done", "done", False), ("cancelled", "cleared", False),
    ("waiting", "paused", True), ("cleared", "cleared", False),
])
def test_status_compatibility_and_terminal_migration_guard(lane, status, expected, migrates):
    import json
    runner, store, db, source, old = lane
    db.set_meta("goal:" + old.session_id, json.dumps({"goal": "old", "status": status}))
    assert goals.load_goal(old.session_id).status == expected
    assert goals.migrate_goal_to_session(old.session_id, "target") is migrates
    assert (goals.load_goal("target") is not None) is migrates

@pytest.mark.asyncio
@pytest.mark.parametrize("in_place", [True, False])
async def test_branch_boundary_clears_only_departed_route(lane, in_place):
    from dataclasses import replace
    from gateway.platforms.event import MessageEvent
    runner, store, db, source, old = lane
    db.append_message(old.session_id, "user", "existing conversation")
    goals.GoalManager(old.session_id).set("original goal")
    runner.config = {}
    runner._branch_open_thread = AsyncMock(return_value=None if in_place else replace(source, thread_id="8"))
    runner._clear_session_boundary_security_state = Mock()
    runner._evict_cached_agent = Mock()
    await runner._handle_branch_command(MessageEvent(text="/branch fork", source=source))
    assert goals.load_goal(old.session_id).status == ("cleared" if in_place else "active")
