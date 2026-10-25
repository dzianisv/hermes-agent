"""Provider-owned retry categories quiet presentation, never outcomes or operator evidence."""
from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from agent.notification_presentation import notification_policy_snapshot
from agent.status_output import StatusOutputMixin
from gateway.warning_notifications import ProviderRetryText

_SWITCH = "⚠️ Rate limited — switching to fallback provider..."
_KEY = "suppress_provider_retry_notifications"


class Emitter(StatusOutputMixin):
    log_prefix = ""
    _print_fn = None
    suppress_status_output = False
    _executing_tools = False

    def _has_stream_consumers(self):
        return False

    def _touch_activity(self, text):
        self.activity.append(text)


@pytest.mark.parametrize("surface", ["cli", "tui", "telegram", "slack"])
@pytest.mark.parametrize("mode", [None, False, True, "show_override", "hide_override", "null", "invalid", "broad"])
def test_narrow_policy_uses_owner_snapshot_at_real_surface(tmp_path, monkeypatch, caplog, surface, mode):
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from gateway.config import Platform
    from gateway.run import GatewayRunner, _load_gateway_config, _profile_runtime_scope
    from gateway.run_turn_runner import TurnRunner
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext
    from hermes_cli.cli_stream_mixin import CLIStreamMixin
    from tui_gateway import server

    homes = [tmp_path / name for name in ("owner", "launch")]
    display = {} if mode is None else {_KEY: mode}
    if mode in {"show_override", "hide_override", "null"}:
        display = {_KEY: mode != "hide_override", "platforms": {
            surface: {_KEY: {"show_override": False, "hide_override": True, "null": None}[mode]}}}
    if mode == "broad":
        display = {_KEY: False, "suppress_warning_notifications": True}
    for home, config in zip(homes, ({"display": display}, {"display": {_KEY: False}})):
        home.mkdir()
        (home / "config.yaml").write_text(json.dumps(config))
    monkeypatch.setenv("HERMES_HOME", str(homes[1]))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    monkeypatch.setattr("gateway.run.safe_schedule_threadsafe", lambda coro, *a, **k: asyncio.run(coro))
    was_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for home in (homes[0], homes[1], homes[0]):
            with _profile_runtime_scope(home, prepared_secret_scope={}):
                config = _load_gateway_config()
            muted = home == homes[0] and mode in {True, "hide_override", "null", "broad"}
            broad = home == homes[0] and mode == "broad"
            rendered, observed = [], []
            agent = Emitter()
            agent.platform, agent.activity = surface, []
            agent._print_fn = lambda text, **kw: rendered.append(str(text))
            if surface == "cli":
                ui = SimpleNamespace(agent=agent, _invalidate=lambda: rendered.append(ui._spinner_text))
                status_sink = lambda *a: None  # the core print sink owns CLI status lines
                wait_sink = lambda text: CLIStreamMixin._on_thinking(ui, text)
            elif surface == "tui":
                agent.suppress_status_output = True
                monkeypatch.setattr(server, "_emit", lambda ev, sid, payload: rendered.append(payload["text"]))
                monkeypatch.setitem(server._sessions, "retry-owner", {"agent": agent, "profile_home": str(home)})
                callbacks = server._agent_cbs("retry-owner")
                status_sink, wait_sink = callbacks["status_callback"], callbacks["thinking_callback"]
            else:
                agent.suppress_status_output = True
                async def send(chat_id, text, **kwargs):
                    rendered.append(text)
                    return SimpleNamespace(success=True)
                source = SessionSource(platform=Platform(surface), chat_id="test", user_id="test")
                ctx = TurnContext(source=source, user_config=config, _run_still_current=lambda: True,
                                  _status_adapter=SimpleNamespace(send=send), _status_chat_id="test",
                                  _status_thread_metadata={})
                status_sink = TurnRunner(object.__new__(GatewayRunner), ctx)._status_callback_sync
                wait_sink = lambda text: None  # messaging uses activity bookkeeping, not thinking text
            def status(kind, text):
                observed.append((kind, text))
                logging.getLogger("retry-observer").warning("observed %s", text)
                status_sink(kind, text)
            agent.status_callback, agent.thinking_callback = status, wait_sink
            # Invoke outside the owner scope, as callback workers do. The foreground snapshot owns policy.
            with notification_policy_snapshot(agent, surface, None if surface == "tui" else config):
                agent._buffer_diagnostic_status(_SWITCH, provider_retry=True)
                agent._buffer_vprint("provider attempt detail", provider_retry=True)
                agent._flush_status_buffer()
                agent._pending_fallback_notice = "Provider switched successfully"
                agent._emit_pending_fallback_notice()
                agent._emit_diagnostic_wait("Recovery countdown", provider_retry=True)
                assert bool(rendered) is not muted
                assert all(isinstance(text, ProviderRetryText) for _, text in observed)
                assert agent._pending_fallback_notice is None and not agent._retry_status_buffer
                assert agent.activity == ["Recovery countdown"]
                rendered.clear()
                agent._emit_warning("Database incident")
                agent._emit_diagnostic_status("Final provider failure")
                assert bool(rendered) is not broad
                rendered.clear()
                # Identical user/model wording is not classified by its text.
                agent._emit_status(_SWITCH)
                assert rendered and "switching to fallback" in rendered[-1]
                if surface == "tui":
                    from tui_gateway import server_requests
                    controls = []
                    monkeypatch.setattr(server_requests, "send", lambda method, *a, **k: controls.append(method) or {"answer": "yes"})
                    monkeypatch.setattr(server_requests, "send_async", lambda method, *a, **k: controls.append(method))
                    assert callbacks["clarify_callback"]("Continue?", ["yes", "no"]) == "yes"
                    server._emit_approval_request("retry-owner", {"command": "test-command"})
                    assert controls == ["clarify", "approval"]
            assert "observed Provider switched successfully" in caplog.text
    finally:
        set_multiplex_active(was_multiplex)


@pytest.mark.real_retry_backoff
@pytest.mark.parametrize("muted", [False, True])
@pytest.mark.parametrize("status", [429, 503, 529])
@pytest.mark.parametrize("recovers", [False, True])
def test_real_http_retry_and_recovery_producers_keep_failures_and_logs(
    tmp_path, monkeypatch, caplog, status, muted, recovers,
):
    from agent import turn_recovery
    from tests.agent.test_openrouter_bounded_retries import _agent, _virtual_waits, _KEYS
    from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Text

    waits = _virtual_waits(monkeypatch)
    monkeypatch.setattr(turn_recovery, "interruptible_backoff_sleep", lambda agent, seconds, *a, **k: waits.append(seconds))
    error = Error(status, "Rate limit exceeded" if status == 429 else "Provider overloaded", retry_after=120)
    script = [error] * (1 if status == 429 else 2) + [Text("recovered")] if recovers else lambda _: error
    printed, observed, thinking = [], [], []
    with FakeLLMServer(script, api_key=_KEYS) as server:
        agent = _agent(monkeypatch, tmp_path, server, budget=2, fallback=False)
        agent._auto_recovery_cycles = 1
        agent._disable_streaming = True  # isolate logical attempts from the streaming unmask probe
        agent._print_fn = lambda text, **kw: printed.append(str(text))
        agent.status_callback = lambda kind, text: observed.append(text)
        agent.thinking_callback = thinking.append
        with notification_policy_snapshot(agent, "cli", {"display": {_KEY: muted}}):
            result = agent.run_conversation("Answer this request.")
            agent._emit_warning("Unrelated watchdog warning")
    assert waits and "API call failed" in caplog.text
    progress = [text for text in observed if isinstance(text, ProviderRetryText)]
    assert thinking and any(isinstance(text, ProviderRetryText) for text in thinking)
    assert all((str(text) in printed) is not muted for text in progress)
    if not recovers or status != 429:
        assert progress
    assert printed[-1] == "Unrelated watchdog warning"
    if status != 429:
        assert any("automatically" in text for text in progress)
        assert "auto-recovery cycle" in caplog.text
    if recovers:
        assert result["completed"] and result["final_response"] == "recovered"
    else:
        assert result["failed"] and not result["completed"] and result["error"]
        assert result["final_response"]
        assert any(not isinstance(text, ProviderRetryText) for text in observed)
        assert any("Final error:" in text for text in printed)
