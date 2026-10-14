"""Gateway wake for goals parked on a wait barrier (``gateway/run_goal_wake.py``).

Contract: on a housekeeping pass, every ACTIVE goal whose barrier (timed / pid / session) has
lifted gets its canonical continuation injected into its gateway session exactly once; a held
barrier, a busy session, a paused goal, or an engaged emergency stop injects nothing. Real
``GoalManager`` state in a temp HERMES_HOME; a fake runner/adapter records the injections.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway import run as gateway_run  # noqa: F401  (import before the per-test home-io guard arms)
from gateway import run_goal_wake
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from hermes_cli import goals


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    goals._DB_CACHE.clear()
    goals._get_session_db()
    yield home
    reset_hermes_home_override(token)
    goals._DB_CACHE.clear()


class _Adapter:
    def __init__(self):
        self._message_handler = object()
        self._active_sessions: set = set()
        self._pending_messages: dict = {}
        self.handled: list = []

    async def handle_message(self, event):
        self.handled.append(event)
        self._active_sessions.add(event.metadata["gateway_session_key"])


class _Store:
    def __init__(self, entries):
        self._entries = entries

    def list_sessions(self):
        return list(self._entries)

    def peek_session_id(self, key):
        return next((e.session_id for e in self._entries if e.session_key == key), None)


class _Runner:
    """Just the seams ``run_goal_wake`` touches on GatewayRunner."""

    def __init__(self, entries):
        self.adapter = _Adapter()
        self.session_store = _Store(entries)
        self.running: set = set()
        self.config = SimpleNamespace(multiplex_profiles=False)

    def _restored_source(self, entry):
        return entry.origin

    def _session_key_for_source(self, source):
        return f"agent:main:{source.platform}:{source.chat_id}"

    def _delivery_adapter_for(self, source):
        return self.adapter

    def _is_session_running(self, key):
        return key in self.running

    def _queue_depth(self, key, *, adapter=None):
        return 1 if key in self.adapter._pending_messages else 0

    def _lane_free(self, key, adapter):
        from gateway.run_busy import GatewayBusySessionMixin
        return GatewayBusySessionMixin._lane_free(self, key, adapter)

    def _profile_scope_for_source(self, source):
        return contextlib.nullcontext()

    async def _run_in_executor_with_context(self, fn, *args):
        return fn(*args)

    async def _warm_goals_session_db(self, label):
        return None

    @staticmethod
    def _synthetic_prompt_event(source, text, *, internal=False):
        # Real builder (not a stub): the system:* -> internal rule lives there and is what we test.
        from gateway.run_goals import GatewayGoalsMixin as GoalsMixin
        return GoalsMixin._synthetic_prompt_event(source, text, internal=internal)

    async def _inject_owned_prompt(self, *args, **kwargs):
        from gateway.run_goals import GatewayGoalsMixin as GoalsMixin
        return await GoalsMixin._inject_owned_prompt(self, *args, **kwargs)


def _entry(sid: str, chat: str = "c1"):
    source = SessionSource(platform="telegram", chat_id=chat, user_id="u1", chat_type="dm")
    return SimpleNamespace(session_id=sid, session_key=f"agent:main:telegram:{chat}", origin=source,
                           suspended=False, transport_profile=None)


def _run(runner):
    return asyncio.run(run_goal_wake.wake_lifted_goal_barriers(runner))


def _parked_agent_goal(sid: str, seconds: int = 60) -> goals.GoalManager:
    mgr = goals.GoalManager(sid)
    assert mgr.set_agent_goal("ship the PR") is not None
    mgr.wait_for_seconds(seconds, reason="waiting on CI")
    return mgr


def _elapse(mgr: goals.GoalManager) -> None:
    mgr.state.waiting_until = time.time() - 1
    mgr._save()


class TestBarrierWake:
    def test_elapsed_timed_barrier_schedules_the_continuation_exactly_once(self, hermes_home):
        mgr = _parked_agent_goal("gw-wake-timed")
        _elapse(mgr)
        runner = _Runner([_entry("gw-wake-timed")])

        assert _run(runner) == 1
        assert len(runner.adapter.handled) == 1
        event = runner.adapter.handled[0]
        assert event.source.platform == "telegram" and event.source.chat_id == "c1"
        assert event.text.startswith("The wait has elapsed; check what you were waiting on and keep going.\n")
        assert "ship the PR" in event.text
        assert event.metadata["gateway_session_key"] == "agent:main:telegram:c1"
        assert goals.GoalManager("gw-wake-timed").state.waiting_until == 0.0   # barrier cleared on disk

        # Next tick: the barrier is gone, so nothing is injected again.
        assert _run(runner) == 0
        assert len(runner.adapter.handled) == 1

    def test_user_goal_gets_its_verbatim_continuation(self, hermes_home):
        mgr = goals.GoalManager("gw-wake-user")
        mgr.set("ship the PR")
        mgr.wait_for_seconds(60, reason="cooldown")
        _elapse(mgr)
        runner = _Runner([_entry("gw-wake-user")])
        assert _run(runner) == 1
        assert runner.adapter.handled[0].text.startswith("[Continuing toward your standing goal]\nGoal:")

    def test_held_barrier_injects_nothing(self, hermes_home):
        _parked_agent_goal("gw-wake-held", seconds=120)
        runner = _Runner([_entry("gw-wake-held")])
        assert _run(runner) == 0
        assert runner.adapter.handled == []
        assert goals.GoalManager("gw-wake-held").state.waiting_until > time.time()

    def test_busy_session_is_deferred_until_idle(self, hermes_home):
        mgr = _parked_agent_goal("gw-wake-busy")
        _elapse(mgr)
        runner = _Runner([_entry("gw-wake-busy")])
        runner.running.add("agent:main:telegram:c1")

        assert _run(runner) == 0
        assert runner.adapter.handled == []
        # Deferred, not dropped: the barrier is still on disk, so the next idle tick wakes it.
        assert goals.GoalManager("gw-wake-busy").state.waiting_until != 0.0

        runner.running.clear()
        assert _run(runner) == 1

    def test_queued_fifo_item_counts_as_busy(self, hermes_home):
        _elapse(_parked_agent_goal("gw-wake-fifo"))
        runner = _Runner([_entry("gw-wake-fifo")])
        runner.adapter._pending_messages["agent:main:telegram:c1"] = object()
        assert _run(runner) == 0

    def test_paused_goal_injects_nothing(self, hermes_home):
        mgr = _parked_agent_goal("gw-wake-paused")
        _elapse(mgr)
        mgr.pause()
        runner = _Runner([_entry("gw-wake-paused")])
        assert _run(runner) == 0
        assert runner.adapter.handled == []

    def test_emergency_stop_leaves_the_barrier_for_later(self, hermes_home):
        _elapse(_parked_agent_goal("gw-wake-estop"))
        runner = _Runner([_entry("gw-wake-estop")])
        with patch("agent.estop.check_paused", return_value=True):
            assert _run(runner) == 0
        assert goals.GoalManager("gw-wake-estop").state.waiting_until != 0.0

    def test_goal_without_a_gateway_session_is_left_to_the_cli_hook(self, hermes_home):
        _elapse(_parked_agent_goal("cli-owned-goal"))
        runner = _Runner([_entry("some-other-session")])
        assert _run(runner) == 0

    def test_pid_barrier_lifts_when_the_process_exits(self, hermes_home):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        mgr = goals.GoalManager("gw-wake-pid")
        assert mgr.set_agent_goal("ship the PR") is not None
        mgr._park("subagent", waiting_on_pid=proc.pid)   # bypass wait_on's liveness precheck
        runner = _Runner([_entry("gw-wake-pid")])
        assert _run(runner) == 1
        assert goals.GoalManager("gw-wake-pid").state.waiting_on_pid is None

    def test_live_pid_barrier_holds(self, hermes_home):
        mgr = goals.GoalManager("gw-wake-pid-live")
        assert mgr.set_agent_goal("ship the PR") is not None
        mgr.wait_on(os.getpid(), reason="subagent")
        runner = _Runner([_entry("gw-wake-pid-live")])
        assert _run(runner) == 0


def test_housekeeping_tick_registers_the_wake_chore():
    """The chore is in the housekeeping chore table whenever a runner + loop exist."""
    import threading

    captured: dict = {}

    def fake_wait(stop_event, interval, queue_watch, chore_fn):
        stop_event.set()

    loop, runner = MagicMock(), MagicMock()
    with patch("gateway.run_delivery_queue_watch.wait_for_next_tick", fake_wait), \
         patch("gateway.run._housekeeping_chore", lambda label, fn: captured.setdefault(label, fn)):
        gateway_run._start_gateway_housekeeping(threading.Event(), adapters={}, loop=loop, runner=runner)
    assert "Goal barrier wake" in captured
    captured["Goal barrier wake"]()
    loop.call_soon_threadsafe.assert_called_once_with(runner._start_heartbeat_poller)


class TestIdleActiveWake:
    """An ACTIVE goal with no barrier whose last turn is older than goals.idle_wake_seconds is
    re-poked exactly once per idle window (live 2026-10-09: 8 such rows sat 6h–1d, invisible to
    the barrier scan because they carried no barrier)."""

    @staticmethod
    def _idle_goal(sid: str, idle_s: int) -> goals.GoalManager:
        mgr = goals.GoalManager(sid)
        assert mgr.set_agent_goal("ship the PR") is not None
        mgr.state.last_turn_at = time.time() - idle_s
        mgr._save()
        return mgr

    def test_idle_active_goal_is_re_poked_once(self, hermes_home):
        self._idle_goal("gw-idle-1", goals.DEFAULT_IDLE_WAKE_SECONDS + 5)
        runner = _Runner([_entry("gw-idle-1")])
        assert _run(runner) == 1
        assert "ship the PR" in runner.adapter.handled[0].text
        # Second pass inside the same idle window: throttled by the last_turn_at stamp.
        assert _run(runner) == 0
        assert len(runner.adapter.handled) == 1

    def test_recent_turn_is_not_poked(self, hermes_home):
        self._idle_goal("gw-idle-2", 30)
        runner = _Runner([_entry("gw-idle-2")])
        assert _run(runner) == 0

    def test_busy_session_defers(self, hermes_home):
        self._idle_goal("gw-idle-3", goals.DEFAULT_IDLE_WAKE_SECONDS + 5)
        runner = _Runner([_entry("gw-idle-3")])
        runner.running.add("agent:main:telegram:c1")
        assert _run(runner) == 0

    def test_paused_goal_is_not_poked(self, hermes_home):
        mgr = self._idle_goal("gw-idle-4", goals.DEFAULT_IDLE_WAKE_SECONDS + 5)
        mgr.pause()
        runner = _Runner([_entry("gw-idle-4")])
        assert _run(runner) == 0

    def test_disabled_by_config(self, hermes_home, monkeypatch):
        self._idle_goal("gw-idle-5", goals.DEFAULT_IDLE_WAKE_SECONDS + 5)
        monkeypatch.setattr(goals, "idle_wake_seconds", lambda: 0)
        runner = _Runner([_entry("gw-idle-5")])
        assert _run(runner) == 0


class TestHandedOffSessionWakesOnDestination:
    """A handed-off session keeps its old routing entry (store entries are never deleted), and the
    source lane's entry stays the most-recently-updated one for hours because the wakes themselves
    touch it. Without consulting ``handoff_platform`` a Telegram-born goal moved to Slack kept
    waking — and posting judge notices — into Telegram."""

    def test_wake_picks_the_handoff_destination_entry_not_the_newest(self, hermes_home):
        sid = "gw-wake-handoff"
        mgr = _parked_agent_goal(sid)
        _elapse(mgr)
        db = goals._get_session_db()
        db.create_session(sid, "telegram")
        assert db.request_handoff(sid, "slack")
        db.complete_handoff(sid)

        telegram = _entry(sid)
        slack_source = SessionSource(platform="slack", chat_id="C1", user_id="system:handoff", chat_type="group",
                                     thread_id="t1", user_name="Handoff")
        slack = SimpleNamespace(session_id=sid, session_key="agent:main:slack:C1:t1", origin=slack_source,
                                suspended=False, transport_profile=None)
        runner = _Runner([telegram, slack])  # telegram first == newest in a real store

        assert _run(runner) == 1
        event = runner.adapter.handled[0]
        assert str(getattr(event.source.platform, "value", event.source.platform)) == "slack"
        assert event.source.chat_id == "C1"
        assert event.internal is True  # system:handoff principal cannot pass the human allowlist

    def test_human_origin_wake_stays_non_internal(self, hermes_home):
        sid = "gw-wake-human"
        mgr = _parked_agent_goal(sid)
        _elapse(mgr)
        runner = _Runner([_entry(sid)])
        assert _run(runner) == 1
        assert runner.adapter.handled[0].internal is False  # authz + emergency stop still apply

    def test_source_lane_entry_still_wakes_when_destination_has_no_entry(self, hermes_home):
        sid = "gw-wake-handoff-orphan"
        mgr = _parked_agent_goal(sid)
        _elapse(mgr)
        db = goals._get_session_db()
        db.create_session(sid, "telegram")
        assert db.request_handoff(sid, "slack")
        db.complete_handoff(sid)
        runner = _Runner([_entry(sid)])

        assert _run(runner) == 0  # wait for the destination; never leak into the source lane
        destination = _entry(sid, "destination")
        destination.origin.platform = "slack"
        destination.session_key = "agent:main:slack:destination"
        runner.session_store._entries.append(destination)
        assert _run(runner) == 1
        assert runner.adapter.handled[0].source.platform == "slack"

    def test_no_handoff_keeps_the_newest_entry(self, hermes_home):
        sid = "gw-wake-nohandoff"
        mgr = _parked_agent_goal(sid)
        _elapse(mgr)
        goals._get_session_db().create_session(sid, "telegram")
        runner = _Runner([_entry(sid, "newest"), _entry(sid, "older")])

        assert _run(runner) == 1
        assert runner.adapter.handled[0].source.chat_id == "newest"


def test_synthetic_prompt_event_marks_system_principal_internal():
    """Every self-injected prompt (wake, continuation, heartbeat) on a handed-off lane must be
    internal: system:handoff can never pass the human allowlist (agentpod 03:34 continuation died)."""
    from gateway.run_goals import GatewayGoalsMixin as GoalsMixin
    handoff = SessionSource(platform="slack", chat_id="C1", user_id="system:handoff", chat_type="group", thread_id="t")
    human = SessionSource(platform="slack", chat_id="C1", user_id="U1", chat_type="group", thread_id="t")
    assert GoalsMixin._synthetic_prompt_event(handoff, "go").internal is True
    assert GoalsMixin._synthetic_prompt_event(human, "go").internal is False


class TestHeartbeatRestoreSkipsStaleHandoffLane:
    """Heartbeat watches are restored from the routing index, which keeps the pre-handoff Telegram
    entry forever. Registering it made every heartbeat tick (and its "⏳ Working" / provider-retry
    bubbles) post into Telegram after the session had moved to Slack."""

    def _restore(self, runner, monkeypatch):
        from gateway import run_heartbeat_restore as mod
        import gateway.run as gw_run
        monkeypatch.setattr(gw_run, "_profile_runtime_scope", lambda home: contextlib.nullcontext(), raising=False)
        monkeypatch.setattr(gw_run, "_multiplex_profile_homes", lambda cfg: [], raising=False)
        import gateway.run_idle_gates as gates
        monkeypatch.setattr(gates, "profile_has_active_heartbeat", lambda home: True)
        from hermes_cli import heartbeat as hb
        monkeypatch.setattr(hb.HeartbeatManager, "is_active", lambda self: True)
        registered = []
        runner._register_heartbeat_watch = lambda key, source, sid: registered.append((key, source))
        runner._profile_scope_key_for_source = lambda source: "default"
        asyncio.run(mod.restore_heartbeat_watches(runner))
        return registered

    def test_only_handoff_lane_registered(self, hermes_home, monkeypatch):
        sid = "gw-hb-handoff"
        db = goals._get_session_db()
        db.create_session(sid, "telegram")
        assert db.request_handoff(sid, "slack")
        db.complete_handoff(sid)
        telegram = _entry(sid)
        slack_source = SessionSource(platform="slack", chat_id="C1", user_id="system:handoff", chat_type="group",
                                     thread_id="t1", user_name="Handoff")
        slack = SimpleNamespace(session_id=sid, session_key="agent:main:slack:C1:t1", origin=slack_source,
                                suspended=False, transport_profile=None)
        registered = self._restore(_Runner([telegram, slack]), monkeypatch)
        assert [k for k, _ in registered] == ["agent:main:slack:C1:t1"]

    def test_non_handed_off_session_unchanged(self, hermes_home, monkeypatch):
        sid = "gw-hb-plain"
        registered = self._restore(_Runner([_entry(sid)]), monkeypatch)
        assert len(registered) == 1
