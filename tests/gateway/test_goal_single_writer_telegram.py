"""Deterministic gateway → controller → real Telegram adapter delivery acceptance.

Only the model, clock and Telegram transport are local test doubles. No adapter
method, synthetic-event builder, reducer, FIFO drain or delivery filter is stubbed.
"""
import asyncio
import contextlib
import json
import threading
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner
from gateway.config import GatewayConfig, PlatformConfig, Platform
from gateway.session import SessionSource, SessionEntry, build_session_key
from gateway.run_goal_wake import wake_lifted_goal_barriers
from gateway.run_heartbeat_acceptance import resolve_heartbeat_owner
from gateway.session_wake import claim, stamp, settle
from hermes_cli import goals, heartbeat
from hermes_state import SessionDB
from plugins.platforms.telegram.adapter import TelegramAdapter


def process_claim(path, sid, kind, revision=None):
    """Spawn a fresh interpreter/SQLite connection against only this test's DB."""
    script = """
import json, sys
from hermes_state import SessionDB
from hermes_cli import goals
from gateway.session_wake import claim
from pathlib import Path
db = SessionDB(db_path=Path(sys.argv[1]))
try:
    goals._get_session_db = lambda: db
    print(json.dumps(claim(sys.argv[2], sys.argv[3], revision=json.loads(sys.argv[4]))))
finally:
    db.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path), sid, kind, json.dumps(revision)],
        check=True, capture_output=True, text=True, timeout=15)
    return json.loads(completed.stdout.strip())


class Clock:
    now = 10000.0
    def time(self):
        return self.now
    def advance(self, seconds):
        self.now += seconds


class TelegramTransport:
    """Local recorder at the Telegram API boundary, not a replacement adapter."""
    id = 42
    username = "local_test_bot"
    def __init__(self):
        self.messages, self.actions = [], []
        self.delivery_entered, self.delivery_release = asyncio.Event(), asyncio.Event()
        self.suspend_delivery = False
    async def send_message(self, **kwargs):
        if self.suspend_delivery:
            self.delivery_entered.set()
            await self.delivery_release.wait()
        self.messages.append(kwargs)
        return SimpleNamespace(message_id=len(self.messages))
    async def edit_message_text(self, **kwargs):
        self.actions.append(kwargs)
        return SimpleNamespace(message_id=kwargs["message_id"])
    async def send_chat_action(self, **kwargs):
        self.actions.append(kwargs)


@pytest.fixture
def world(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    clock = Clock()
    # Replace module clock objects, not global time.time (async deadlines stay real).
    from gateway import session_wake
    for mod in [goals, heartbeat, session_wake]:
        monkeypatch.setattr(mod, "time", SimpleNamespace(time=clock.time), raising=False)
    yield db, clock
    db.close()


@pytest.mark.asyncio
async def test_quiet_single_owner_through_real_telegram(world, monkeypatch):
    db, clock = world
    transport = TelegramTransport()
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="local-transport-only"))
    adapter._bot = transport
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="456",
                           chat_type="dm", thread_id="7", message_id="99")
    key = build_session_key(source)
    entry = SessionEntry(session_key=key, session_id="acceptance", created_at=datetime.now(),
                         updated_at=datetime.now(), platform=Platform.TELEGRAM, chat_type="dm", origin=source)
    class FakeGateway(GatewayRunner):
        @property
        def async_session_store(self):
            return SimpleNamespace(get_or_create_session=get_entry)
    runner = object.__new__(FakeGateway)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: adapter.config})
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._running_agents = {}
    runner._queued_events = {}
    runner._voice_mode = {}
    runner._goal_session_db = db
    runner.session_store = SimpleNamespace(list_sessions=lambda: [entry], peek_session_id=lambda _key: entry.session_id,
                                          lookup_by_session_key=lambda _key: entry)
    async def get_entry(*args, **kwargs):
        return entry
    runner._profile_scope_for_source = lambda _: contextlib.nullcontext()
    judge_entered, judge_release = threading.Event(), threading.Event()
    suspend_judge = False
    async def execute(fn, *args):
        return await asyncio.to_thread(fn, *args)
    runner._run_in_executor_with_context = execute
    runner._restored_source = lambda e: e.origin
    runner._delivery_adapter_for = lambda _: adapter
    runner._session_key_for_source = lambda _: key
    runner._thread_metadata_for_source = lambda _: {"thread_id": "7"}
    outputs, turns, verdict = [], [], {"verdict": "continue", "reason": "next requirement"}
    def judge(*args):
        if suspend_judge:
            judge_entered.set()
            assert judge_release.wait(15), "judge release was not signalled"
        return json.dumps(verdict)
    monkeypatch.setattr(goals, "_call_goal_judge_llm", judge)

    async def model_boundary(event):
        assert await resolve_heartbeat_owner(runner, event, entry)
        from gateway.run_heartbeat_acceptance import heartbeat_owner_is_current
        assert heartbeat_owner_is_current(runner, event, key)
        event._heartbeat_execution_started = True  # fake model boundary entered, like _run_agent
        turns.append(event)
        assert event.source.message_id is None
        assert outputs, "unexpected extra autonomous turn"
        text = outputs.pop(0)
        result = {"final_response": text, "messages": [], "api_calls": 1}
        delivered = await runner._hmwa_deliver_turn_response(
            event, event.source, entry, key, None, result, [], text, None,
            runner._is_intentional_silence(result, text))
        await runner._run_post_turn_hooks(agent_result=result, source=event.source,
                                          is_internal=event.internal, event=event)
        return delivered
    adapter._message_handler = model_boundary

    async def drain():
        # Adapter FIFO can replace its task during completion; drain to actual idle.
        for _ in range(30):
            tasks = list(adapter._session_tasks.values())
            if not tasks:
                return
            await asyncio.gather(*tasks)
            await asyncio.sleep(0)
        pytest.fail("adapter did not become idle")

    mgr = goals.GoalManager(entry.session_id)
    mgr.set_agent_goal("Publish the verified build artifact")
    mgr.wait_for_seconds(60, "build finishes later")
    hb = heartbeat.HeartbeatManager(entry.session_id)
    hb.set("Report meaningful deployment changes", 60)
    watch = {key: (source, entry.session_id)}

    for delay in [61, 61]:
        clock.advance(delay)
        outputs.append("[SILENT]")
        before = len(turns)
        await asyncio.gather(wake_lifted_goal_barriers(runner),
                             runner._heartbeat_poll_once(watch), wake_lifted_goal_barriers(runner))
        await drain()
        assert len(turns) == before + 1
        assert goals.load_goal(entry.session_id).status == "waiting"
        assert heartbeat.load_heartbeat(entry.session_id).fire_count == 0
        assert transport.messages == [] and transport.actions == []

    clock.advance(121)
    outputs.append("Build artifact checksum verified: abc123.")
    verdict.update(verdict="wait", wait_for_seconds=60, reason="signature worker running")
    suspend_judge = True
    await wake_lifted_goal_barriers(runner)
    assert await asyncio.to_thread(judge_entered.wait, 10)
    try:
        clock.advance(7200)
        # Another scheduler/process must not enter while the judge is suspended.
        revision = goals.load_goal(entry.session_id).revision
        assert claim(entry.session_id, "goal", revision=revision) is None
        assert await asyncio.to_thread(process_claim, db.db_path, entry.session_id, "goal", revision) is None
    finally:
        suspend_judge = False
        judge_release.set()
        await drain()
    assert len(transport.messages) == 1

    clock.advance(61)
    question = "May I publish this artifact to the public download page?"
    verdict.update(verdict="blocked", blocker="human_only", human_decision=question)
    outputs.extend(["Release signature verified.", "[SILENT]"])
    await wake_lifted_goal_barriers(runner)
    await drain()
    assert sum(question in m["text"] for m in transport.messages) == 1
    assert goals.load_goal(entry.session_id).wake_pending
    await wake_lifted_goal_barriers(runner)
    await drain()
    assert sum(question in m["text"] for m in transport.messages) == 1

    # A queued wake loses its authority if the user stops before adapter execution.
    mgr.resume()
    token = claim(entry.session_id, "goal", revision=mgr.state.revision)
    assert token
    stale = runner._synthetic_prompt_event(source, mgr.next_continuation_prompt())
    stamp(stale, entry.session_id, token)
    mgr.pause()
    before = len(turns)
    await adapter.handle_message(stale)
    await drain()
    assert len(turns) == before

    mgr.resume()
    outputs.append("Published artifact: https://example.test/download/build.zip")
    verdict.update(verdict="done", reason="published artifact verified")
    await runner._post_turn_goal_continuation(session_entry=entry, source=source, final_response="[SILENT]")
    clock.advance(goals.load_goal(entry.session_id).waiting_until - clock.now + 1)
    transport.suspend_delivery = True
    await wake_lifted_goal_barriers(runner)
    await asyncio.wait_for(transport.delivery_entered.wait(), 10)
    try:
        assert goals.load_goal(entry.session_id).status == "done"
        # The completed goal no longer blocks heartbeat; only the delivery owner does.
        assert await asyncio.to_thread(process_claim, db.db_path, entry.session_id, "heartbeat") is None
    finally:
        transport.suspend_delivery = False
        transport.delivery_release.set()
        await drain()
    assert goals.load_goal(entry.session_id).status == "done"
    assert any("https://example.test/download/build.zip" in m["text"].replace("\\", "") for m in transport.messages)

    outputs.append("Deployment health check passed.")
    await runner._heartbeat_poll_once(watch)
    await drain()
    assert heartbeat.load_heartbeat(entry.session_id).fire_count == 1
    assert any("Deployment health check passed" in m["text"] for m in transport.messages)
    assert all(m.get("reply_to_message_id") is None for m in transport.messages)
    assert all(str(m.get("message_thread_id")) == "7" for m in transport.messages)
    assert not any(word in m["text"] for m in transport.messages for word in
                   ["Continuing toward", "remaining", "backing off", "rerouting", "Goal parked", "[SILENT]"])
    assert not outputs


def test_live_owner_never_expires_dead_owner_is_reclaimed(world):
    from gateway.status import get_process_start_time
    db, clock = world
    token = claim("liveness", "heartbeat")
    clock.advance(7200)
    assert claim("liveness", "heartbeat") is None
    settle(SimpleNamespace(_session_wake_id="liveness", _session_wake_token=token))
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        owner = {"token": "dead-owner", "kind": "heartbeat", "revision": None,
                 "pid": child.pid, "start_time": get_process_start_time(child.pid)}
        db._execute_write(lambda conn: conn.execute(
            "INSERT INTO state_meta(key,value) VALUES (?,?)",
            ("session_wake:liveness", json.dumps(owner))))
        assert claim("liveness", "heartbeat") is None
    finally:
        child.terminate()
        child.wait(timeout=5)
    replacement = claim("liveness", "heartbeat")
    assert replacement and replacement != owner["token"]
    assert process_claim(db.db_path, "liveness", "heartbeat") is None
    # A recycled PID is not the recorded owner, even though that PID is alive.
    row = db._execute_write(lambda conn: conn.execute(
        "SELECT value FROM state_meta WHERE key=?", ("session_wake:liveness",)).fetchone()[0])
    recycled = json.loads(row)
    recycled["start_time"] -= 10000
    db._execute_write(lambda conn: conn.execute(
        "UPDATE state_meta SET value=? WHERE key=?", (json.dumps(recycled), "session_wake:liveness")))
    assert claim("liveness", "heartbeat") not in {None, replacement}
    # A late cleanup from the dead process cannot delete the new owner's claim.
    settle(SimpleNamespace(_session_wake_id="liveness", _session_wake_token=owner["token"]))
    assert claim("liveness", "heartbeat") is None
