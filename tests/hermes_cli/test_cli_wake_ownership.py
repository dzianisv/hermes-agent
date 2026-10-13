"""CLI autonomous inputs own one durable session lane through execution/discard."""
import queue
from types import SimpleNamespace

import pytest

from gateway import session_wake
from hermes_cli import goals
from hermes_cli.cli_loops_mixin import CLILoopsMixin


class CLI(CLILoopsMixin):
    def __init__(self, sid):
        self.session_id = sid
        self._pending_input = queue.Queue()
        self._agent_running = False


@pytest.mark.parametrize("invalidate", ["replace", "wait", "clear", "session"])
def test_parked_wake_is_owned_quiet_and_fenced_at_dispatch(monkeypatch, invalidate):
    from hermes_cli.cli_tui_runtime_mixin import CLITuiRuntimeMixin
    clock = SimpleNamespace(now=10000.0)
    monkeypatch.setattr(goals, "time", SimpleNamespace(time=lambda: clock.now))
    cli = CLI("cli-wake")
    mgr = cli._get_goal_manager()
    mgr.set("finish the requested outcome")
    mgr.wait_for_seconds(1, reason="cooldown")
    clock.now += 2
    printed = []
    monkeypatch.setattr("cli._cprint", printed.append)
    cli._maybe_resume_parked_goal()
    event = cli._pending_input.get_nowait()
    assert getattr(event, "_session_wake_id", None) == cli.session_id
    assert session_wake.current(event)
    assert session_wake.claim(cli.session_id, "goal", revision=mgr.state.revision) is None
    assert printed == []
    if invalidate == "replace":
        mgr.set("a different outcome")
    elif invalidate == "wait":
        mgr.wait_for_seconds(60, reason="not ready")
    elif invalidate == "clear":
        mgr.clear()
    else:
        cli.session_id = "another-session"
    # The stale item must be discarded before any UI, credential, or agent work.
    CLITuiRuntimeMixin._tui_process_one_input(cli, event)
    assert not session_wake.current(event)
    assert printed == []


@pytest.mark.parametrize("producer", ["kick", "heartbeat", "continue", "loop"])
@pytest.mark.parametrize("outcome", ["complete", "error", "late_clear", "queue_clear", "thread_clear"])
def test_wakes_share_ownership_and_settle_every_exit(monkeypatch, producer, outcome):
    from cli import HermesCLI
    from hermes_cli.heartbeat import HeartbeatManager
    from hermes_cli.loops import LoopManager
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "cli-owner"
    cli._pending_input = queue.Queue()
    cli._agent_running = False
    mgr = cli._get_goal_manager()
    if producer in {"kick", "continue"}:
        mgr.set("finish the requested outcome")
    if producer == "kick":
        assert cli._kick_goal("finish the requested outcome")
    elif producer == "continue":
        cli.agent = SimpleNamespace(session_id=cli.session_id)
        cli.conversation_history = [{"role": "assistant", "content": "more work remains"}]
        monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: ("continue", "more work", False, None, False))
        cli._maybe_continue_goal_after_turn()
    elif producer == "heartbeat":
        hb = HeartbeatManager(cli.session_id)
        hb.set("check the work", interval_seconds=60)
        monkeypatch.setattr("hermes_cli.heartbeat.time", SimpleNamespace(time=lambda: 10**12))
        cli._maybe_fire_heartbeat_tick()
    else:
        loop = LoopManager(cli.session_id)
        loop.set("check the work", interval_seconds=60)
        monkeypatch.setattr("hermes_cli.loops.time", SimpleNamespace(time=lambda: 10**12))
        cli._maybe_fire_loop_tick()
    event = cli._pending_input.get_nowait()
    assert getattr(event, "_session_wake_id", None) == cli.session_id
    assert session_wake.current(event)
    assert session_wake.claim(cli.session_id, "heartbeat") is None
    if outcome == "queue_clear":
        cli._pending_input.put(event)
        cli._queue_clear("")
    else:
        ran = []
        def execute(message, images=None, voice_input=False):
            assert session_wake.current(event)
            if outcome == "error":
                raise ValueError("turn failed")
            if outcome == "thread_clear":
                mgr.set("changed during initialization")
                cli.conversation_history = []
                turn = SimpleNamespace(cli_wake=event)
                CLIChatTurnMixin._chat_run_agent(cli, turn, message)
                assert turn.result["final_response"] == ""
                assert cli._last_cli_wake_discarded
                return None
            ran.append(message)
            return "response"
        monkeypatch.setattr(cli, "_chat_turn", execute, raising=False)
        if outcome == "late_clear":
            mgr.set("new goal arrived after dequeue")
        if outcome == "error":
            with pytest.raises(ValueError, match="turn failed"):
                CLIChatTurnMixin.chat(cli, event)
        else:
            CLIChatTurnMixin.chat(cli, event)
            assert bool(ran) == (outcome == "complete")
    assert not session_wake.current(event)
    state = mgr.state
    token = session_wake.claim(cli.session_id, "goal", revision=state.revision) if state else session_wake.claim(cli.session_id, "heartbeat")
    assert token is not None
    receipt = SimpleNamespace()
    session_wake.stamp(receipt, cli.session_id, token)
    session_wake.settle(receipt)
