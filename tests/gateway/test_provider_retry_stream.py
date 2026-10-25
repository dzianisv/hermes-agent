"""The API's real status-queue sink filters producer categories before serializing them."""
import json
from types import SimpleNamespace

import pytest

from agent.status_output import StatusOutputMixin
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, ThreadSafeAsyncQueue
from gateway.run import _profile_runtime_scope


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [None, False, True])
async def test_stream_status_policy_keeps_unrelated_warnings_and_terminal_outcome(tmp_path, monkeypatch, override):
    homes = [tmp_path / name for name in ("owner", "launch")]
    for home, muted in zip(homes, (True, False)):
        home.mkdir()
        (home / "config.yaml").write_text(json.dumps({"display": {
            "suppress_provider_retry_notifications": muted,
            "platforms": {"api_server": {"suppress_provider_retry_notifications": override}}}}))
    monkeypatch.setenv("HERMES_HOME", str(homes[1]))
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    seen = []
    for home in (homes[0], homes[1], homes[0]):
        async def run(*, status_callback, stream_delta_callback, reasoning_callback, agent_ref, **kwargs):
            with _profile_runtime_scope(home, prepared_secret_scope={}):
                agent_ref[0] = SimpleNamespace(_notification_config=None)
                agent = StatusOutputMixin()
                agent.suppress_status_output = True
                def observe(kind, text):
                    seen.append((kind, text))
                    status_callback(kind, text)
                agent.status_callback = observe
                agent._emit_diagnostic_status("Provider recovery countdown", provider_retry=True)
                agent._emit_warning("Database incident")
                agent._emit_diagnostic_status("Final provider failure")
                stream_delta_callback("Requested result")
                return {"failed": True, "completed": False, "error": "truthful failure"}, None
        monkeypatch.setattr(adapter, "_run_agent", run)
        queue = ThreadSafeAsyncQueue()
        task, _ = adapter._spawn_stream_agent(queue)
        outcome, _ = await task
        payloads = []
        while (payload := await queue.get()) is not None:
            payloads.append(payload)
        texts = [p[1]["text"] for p in payloads if isinstance(p, tuple)]
        muted = override if override is not None else home == homes[0]
        assert texts == ([] if muted else ["Provider recovery countdown"]) + ["Database incident", "Final provider failure"]
        assert payloads[-1] == "Requested result"
        assert outcome == {"failed": True, "completed": False, "error": "truthful failure"}
    assert len(seen) == 9  # callbacks run even when their concrete presentation is muted
