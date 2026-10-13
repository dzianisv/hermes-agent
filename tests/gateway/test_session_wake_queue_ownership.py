"""Wake tokens follow exact FIFO events, not the human task occupying their lane."""
import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.session_wake import claim, current, stamp, track
from hermes_cli import goals
from hermes_state import SessionDB
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture
def queue_world(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="456", chat_type="dm")
    key = build_session_key(source)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="local-transport-only"))
    runner = object.__new__(GatewayRunner)
    runner._queued_events = {}
    mgr = goals.GoalManager("queue")
    mgr.set_agent_goal("Verify queued work")
    def wake():
        token = claim("queue", "goal", revision=goals.load_goal("queue").revision)
        assert token
        event = runner._synthetic_prompt_event(source, mgr.next_continuation_prompt())
        stamp(event, "queue", token)
        return event
    yield runner, adapter, source, key, wake
    db.close()


@pytest.mark.asyncio
async def test_clear_and_cancel_settle_all_dropped_events(queue_world):
    runner, adapter, source, key, wake = queue_world
    event = wake()
    runner._enqueue_fifo(key, event, adapter)
    assert runner._clear_goal_pending_continuations(key, adapter) == 1
    replacement = wake()
    human = MessageEvent(text="Keep my queued request", message_type=MessageType.TEXT, source=source)
    runner._enqueue_fifo(key, human, adapter)
    runner._enqueue_fifo(key, replacement, adapter)
    assert runner._clear_goal_pending_continuations(key, adapter) == 1
    assert adapter._pending_messages[key] is human
    event = wake()
    adapter._pending_messages[key] = event
    await adapter.cancel_session_processing(key)
    assert current(wake())


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["goal", "heartbeat"])
async def test_busy_human_task_cannot_settle_queued_wake(queue_world, kind):
    runner, adapter, source, key, wake = queue_world
    human_entered, human_release = asyncio.Event(), asyncio.Event()
    wake_entered, wake_release = asyncio.Event(), asyncio.Event()
    human = MessageEvent(text="Human request", message_type=MessageType.TEXT, source=source)
    human._goal_prompt = True
    async def handler(event):
        if event is human:
            human_entered.set()
            await human_release.wait()
        else:
            wake_entered.set()
            await wake_release.wait()
        return None
    async def busy(event, session_key):
        runner._enqueue_fifo(session_key, event, adapter)
        return True
    adapter._message_handler = handler
    adapter._busy_session_handler = busy
    await adapter.handle_message(human)
    await asyncio.wait_for(human_entered.wait(), 5)
    previous = adapter._session_tasks[key]
    heartbeat_manager = None
    if kind == "goal":
        event = wake()
    else:
        from hermes_cli.heartbeat import HeartbeatManager, HeartbeatState, save_heartbeat
        save_heartbeat("heartbeat-queue", HeartbeatState(prompt="Check result", interval_seconds=60, created_at=1))
        heartbeat_manager = HeartbeatManager("heartbeat-queue")
        token = claim("heartbeat-queue", "heartbeat")
        event = runner._synthetic_prompt_event(source, heartbeat_manager.due_prompt())
        stamp(event, "heartbeat-queue", token)
        event._heartbeat_manager = heartbeat_manager
        event._heartbeat_session_id = "heartbeat-queue"
        event._heartbeat_execution_started = False
    try:
        await adapter.handle_message(event)
        track(adapter, key, event)
        human_release.set()
        await previous
        await asyncio.sleep(0)
        assert current(event), "human completion stole the queued event's ownership"
        if heartbeat_manager:
            assert HeartbeatManager("heartbeat-queue").state.fire_count == 1
        await asyncio.wait_for(wake_entered.wait(), 5)
        revision = goals.load_goal("queue").revision if kind == "goal" else None
        assert claim(event._session_wake_id, kind, revision=revision) is None
    finally:
        human_release.set()
        wake_release.set()
        await adapter.cancel_background_tasks()
    if heartbeat_manager:
        assert HeartbeatManager("heartbeat-queue").state.fire_count == 0


@pytest.mark.asyncio
async def test_cancelled_turn_retains_owner_until_judge_thread_finishes(queue_world, monkeypatch):
    import json
    import threading
    from types import SimpleNamespace
    runner, adapter, source, key, wake = queue_world
    runner.config = {}
    runner._run_in_executor_with_context = asyncio.to_thread
    entered, release = threading.Event(), threading.Event()
    def judge(*args):
        entered.set()
        assert release.wait(15)
        return json.dumps({"verdict": "continue", "reason": "more verification"})
    monkeypatch.setattr(goals, "_call_goal_judge_llm", judge)
    async def handler(event):
        await runner._post_turn_goal_continuation(
            session_entry=SimpleNamespace(session_id="queue"), source=source,
            final_response="First requirement verified.")
    adapter._message_handler = handler
    event = wake()
    await adapter.handle_message(event)
    task = adapter._session_tasks[key]
    assert await asyncio.to_thread(entered.wait, 10)
    try:
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert current(event)
        assert claim("queue", "goal", revision=goals.load_goal("queue").revision) is None
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
    assert goals.load_goal("queue").wake_pending
    assert claim("queue", "goal", revision=goals.load_goal("queue").revision)
