"""Both native status callback shapes reach the real presentation and stream sinks."""
import json
from types import SimpleNamespace

import pytest

from gateway.warning_notifications import ProviderRetryText
from tui_gateway import server


@pytest.mark.parametrize("text_only", [False, True])
@pytest.mark.parametrize("policy", ["show", "retry_only", "all_warnings"])
def test_status_callback_shapes_preserve_retry_policy_and_stream(tmp_path, monkeypatch, text_only, policy):
    config = {"display": {
        "suppress_provider_retry_notifications": policy == "retry_only",
        "suppress_warning_notifications": policy == "all_warnings",
    }}
    (tmp_path / "config.yaml").write_text(json.dumps(config))
    frames = []
    monkeypatch.setattr(server, "write_json", lambda frame: frames.append(frame) or True)
    monkeypatch.setitem(server._sessions, "status-contract", {
        "agent": SimpleNamespace(_notification_config=config),
        "profile_home": str(tmp_path),
    })
    callback = server._agent_cbs("status-contract")["status_callback"]

    def emit(kind, message):
        if text_only:
            callback(message)
        else:
            callback(kind, message)

    retry = ProviderRetryText("Provider recovery countdown")
    emit("warn", retry)
    # The same words without a producer marker are ordinary requested progress.
    emit("info", str(retry))
    callback("warn", "Database incident")
    callback("info", "Requested result")

    events = [frame["params"] for frame in frames]
    assert all(event["type"] == "status.update" for event in events)
    expected = []
    if policy == "show":
        expected.append({"kind": "status" if text_only else "warn", "text": str(retry)})
    expected.append({"kind": "status" if text_only else "info", "text": str(retry)})
    if policy != "all_warnings":
        expected.append({"kind": "warn", "text": "Database incident"})
    expected.append({"kind": "info", "text": "Requested result"})
    assert [event["payload"] for event in events] == expected
