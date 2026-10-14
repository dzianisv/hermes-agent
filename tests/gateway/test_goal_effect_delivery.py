"""Real post-turn decisions suppress stale notices and FIFO continuation."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import goals
from hermes_state import SessionDB


@pytest.fixture
def lane(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    monkeypatch.setattr(goals, "count_active_delegations", lambda sid: 0)
    monkeypatch.setattr(goals, "gather_background_processes", lambda **kw: [])
    runner = object.__new__(GatewayRunner)
    runner.config = {"goals": {"max_turns": 20}}
    adapter = SimpleNamespace(_pending_messages={})
    runner._delivery_adapter_for = lambda source: adapter
    runner._session_key_for_source = lambda source: "key"
    runner._defer_goal_status_notice_after_delivery = AsyncMock()
    async def execute(fn, *args):
        return fn(*args)
    runner._run_in_executor_with_context = execute
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42")
    yield runner, adapter, source
    db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["clear", "pause", "replace", "none"])
async def test_stale_decision_neither_notifies_nor_enqueues(lane, monkeypatch, mutation):
    runner, adapter, source = lane
    goals.GoalManager("s").set_agent_goal("G1")
    def judge(*args, **kwargs):
        other = goals.GoalManager("s")
        if mutation == "replace":
            other.set_agent_goal("G2", replace=True)
        elif mutation != "none":
            getattr(other, mutation)()
        return "continue", "more work", False, None, False
    monkeypatch.setattr(goals, "judge_goal", judge)
    await runner._post_turn_goal_continuation(
        session_entry=SimpleNamespace(session_id="s"), source=source, final_response="work")
    assert bool(adapter._pending_messages) is (mutation == "none")
    assert runner._defer_goal_status_notice_after_delivery.await_count == int(mutation == "none")
    if mutation == "none":
        assert adapter._pending_messages["key"].internal is False
