"""Known main estop gaps, deliberately not repaired by the goal-row subset.

Run with --runxfail to reproduce the baseline blockers. No live sentinel is touched.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from hermes_cli import goals
from hermes_state import SessionDB
from tests.gateway.test_goal_continuation_drain import _DrainProbeAdapter, _slack_thread_source


@pytest.fixture
def lane(tmp_path, monkeypatch):
    from agent import estop
    monkeypatch.setattr(estop, "_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(estop, "_canonical_root", lambda: tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    monkeypatch.setattr(goals, "gather_background_processes", lambda **kw: [])
    monkeypatch.setattr(goals, "count_active_delegations", lambda sid: 0)
    runner = object.__new__(GatewayRunner)
    runner.config = {"goals": {"max_turns": 20}}
    adapter = _DrainProbeAdapter()
    source = _slack_thread_source()
    runner._session_key_for_source = lambda source: "key"
    runner._delivery_adapter_for = lambda source: adapter
    runner._defer_goal_status_notice_after_delivery = AsyncMock()
    async def execute(fn, *args):
        return fn(*args)
    runner._run_in_executor_with_context = execute
    yield runner, adapter, source, estop
    db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("running", [False, pytest.param(True, marks=pytest.mark.xfail(
    strict=True, reason="main baseline: running-session bypass admits FIFO goal after estop"))])
async def test_estop_after_fifo_admission_before_drain(lane, running):
    runner, adapter, source, estop = lane
    event = runner._synthetic_prompt_event(source, "continue goal")
    assert not event.internal
    runner._enqueue_fifo("key", event, adapter)
    assert adapter._pending_messages["key"] is event
    estop.engage("test-only pause after admission")
    assert estop.is_engaged()
    if running:
        runner._session_state("key").turn.agent = _AGENT_PENDING_SENTINEL
    entered_turn_pipeline = []
    runner._hm_admit_event = AsyncMock(return_value=(event, source, False))
    async def after_gate(*args):
        entered_turn_pipeline.append(event)
        return "entered turn pipeline"
    runner._hm_pending_reply_intercepts = after_gate
    gate = Mock(wraps=runner._hm_estop_gate)
    runner._hm_estop_gate = gate
    adapter.set_message_handler(runner._handle_message)
    adapter._spawn_drain_task(adapter._pending_messages.pop("key"), "key")
    task = adapter._session_tasks["key"]
    await asyncio.wait_for(task, timeout=10)
    assert gate.call_count == 1
    assert entered_turn_pipeline == [], "estop gate admitted the queued continuation"


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="main baseline: post-turn goal hook does not check estop")
async def test_estop_before_post_turn_hook_prevents_judge_and_enqueue(lane, monkeypatch):
    runner, adapter, source, estop = lane
    goals.GoalManager("s").set("work")
    judge = Mock(return_value=("continue", "more work", False, None, False))
    monkeypatch.setattr(goals, "judge_goal", judge)
    estop.engage("test-only pause before hook")
    await runner._post_turn_goal_continuation(
        session_entry=SimpleNamespace(session_id="s"), source=source, final_response="partial")
    assert (judge.call_count, len(adapter._pending_messages)) == (0, 0)
