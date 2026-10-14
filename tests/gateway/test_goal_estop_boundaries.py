"""Goal admission/execution stop boundaries using isolated sentinels and real FIFO drains."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.platforms.event import MessageEvent, MessageType
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
        return await asyncio.to_thread(fn, *args)
    runner._run_in_executor_with_context = execute
    yield runner, adapter, source, estop
    db.close()


@pytest.fixture(params=["normal", "gate_failed"])
def continuation(lane, request):
    mgr = goals.GoalManager("continuation")
    mgr.set("work")
    if request.param == "normal":
        return mgr.next_continuation_prompt()
    mgr.add_gate("exit 1")
    decision = mgr.evaluate_after_turn("partial", user_initiated=True)
    assert decision["verdict"] == "gate_failed" and decision["should_continue"]
    return decision["continuation_prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize("running", [False, True])
async def test_estop_after_fifo_admission_before_drain(lane, continuation, running, paused):
    runner, adapter, source, estop = lane
    event = runner._synthetic_prompt_event(source, continuation)
    assert not event.internal
    runner._enqueue_fifo("key", event, adapter)
    assert adapter._pending_messages["key"] is event
    if paused:
        estop.engage("test-only pause after admission")
    assert estop.is_engaged() is paused
    if running:
        runner._session_state("key").turn.agent = _AGENT_PENDING_SENTINEL
    assert runner._is_session_running("key") is running
    entered_turn_pipeline = await _drain(runner, adapter, source, event)
    assert entered_turn_pipeline == ([] if paused else [event]), "unexpected estop admission"


async def _drain(runner, adapter, source, event):
    # Intercept only auth and the seam AFTER the real execution gate; no model call.
    entered_turn_pipeline = []
    runner._hm_admit_event = AsyncMock(return_value=(event, source, event.internal))
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
    return entered_turn_pipeline


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_at", ["before", "judge", "never"])
async def test_estop_post_turn_hook_admission(lane, monkeypatch, pause_at):
    runner, adapter, source, estop = lane
    goals.GoalManager("s").set("work")
    def verdict(*args, **kwargs):
        if pause_at == "judge":
            estop.engage("test-only pause inside judge")
        return "continue", "more work", False, None, False
    judge = Mock(side_effect=verdict)
    monkeypatch.setattr(goals, "judge_goal", judge)
    manager = AsyncMock(wraps=runner._post_turn_manager)
    runner._post_turn_manager = manager
    enqueue = Mock(wraps=runner._enqueue_fifo)
    runner._enqueue_fifo = enqueue
    if pause_at == "before":
        estop.engage("test-only pause before hook")
    await runner._post_turn_goal_continuation(
        session_entry=SimpleNamespace(session_id="s"), source=source, final_response="partial")
    expected_evaluations = 0 if pause_at == "before" else 1
    assert judge.call_count == expected_evaluations
    assert manager.call_count == expected_evaluations
    assert goals.GoalManager("s").state.turns_used == expected_evaluations
    assert enqueue.call_count == len(adapter._pending_messages) == (1 if pause_at == "never" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["steering", "update", "slash_confirm", "approval", "pause_off", "status"])
async def test_estop_keeps_user_controls_but_not_goal_continuations(lane, continuation, monkeypatch, control):
    from tools import approval, slash_confirm
    from tools.approval_gateway_wait import _ApprovalEntry
    runner, adapter, source, estop = lane
    monkeypatch.setattr(slash_confirm, "_pending", {})
    monkeypatch.setattr(approval, "_gateway_queues", {})
    state = runner._session_state("key")
    text = "yes"
    if control == "steering":
        state.turn.agent = _AGENT_PENDING_SENTINEL
        text = "Use the revised acceptance criteria"
    elif control == "update":
        state.persistent.update_prompt_pending = True
    elif control == "slash_confirm":
        slash_confirm.register("key", "confirm", "reload-mcp", AsyncMock())
        assert slash_confirm.get_pending("key")
    elif control == "approval":
        approval._gateway_queues["key"] = [_ApprovalEntry({"command": "test-only approval"})]
        assert approval.has_blocking_approval("key")
    else:
        text = "/pause off" if control == "pause_off" else "/status"
    estop.engage("test-only control allowance")
    event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source)
    assert not runner._is_goal_continuation_event(event)
    runner._enqueue_fifo("key", event, adapter)
    assert await _drain(runner, adapter, source, event) == [event]
    # The same pending user-control state must not exempt a new goal turn.
    goal_event = runner._synthetic_prompt_event(source, continuation)
    runner._enqueue_fifo("key", goal_event, adapter)
    assert await _drain(runner, adapter, source, goal_event) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("goal_first", [False, True])
async def test_clear_generated_continuations_spares_real_queue_items(lane, continuation, goal_first):
    runner, adapter, source, _ = lane
    event = runner._synthetic_prompt_event(source, continuation)
    if goal_first:
        runner._enqueue_fifo("key", event, adapter)
    for text in ("Write the regression", "Explain [Continuing toward your standing goal]"):
        queued = MessageEvent(text="/queue " + text, message_type=MessageType.TEXT, source=source)
        await runner._busy_queue_command(queued, "key", source)
    if not goal_first:
        runner._enqueue_fifo("key", event, adapter)
    before = [adapter._pending_messages["key"], *runner._overflow_queue("key")]
    real_items = [item for item in before if item is not event]
    assert all(not runner._is_goal_continuation_event(item) for item in real_items)
    assert runner._clear_goal_pending_continuations("key", adapter) == 1
    remaining = list(adapter._pending_messages.values()) + runner._overflow_queue("key")
    assert remaining == real_items


def test_other_internal_events_keep_existing_estop_policy(lane):
    runner, _, source, estop = lane
    estop.engage("test-only internal policy control")
    event = MessageEvent(text="Background process finished", message_type=MessageType.TEXT, source=source)
    assert not runner._is_goal_continuation_event(event)
    assert runner._hm_estop_gate(event, source, True) is None
