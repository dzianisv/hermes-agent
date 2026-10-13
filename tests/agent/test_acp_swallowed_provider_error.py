"""pi-acp reports a provider failure as a normal turn: after pi's auto-retry gives up it
emits "Retrying (attempt n/m...)" + "Retry finished, resuming." as agent_message_chunk text
and answers session/prompt with stopReason=end_turn (pi-acp dist/index.js:1008/2498 maps
"error" -> "end_turn"). Hermes used to hand that status chatter to the agent loop as the
model's reply (0 tokens, no tool call), so an offline worker burned its stop-nudges and
exited rc=0 without a board call (runs 6690/6696). The ACP client must surface it as a
connection failure instead.
"""

from __future__ import annotations

import pytest

from agent import copilot_acp_client as acp
from agent.copilot_acp_client import CopilotACPClient


def _client() -> CopilotACPClient:
    return CopilotACPClient(api_key="x", base_url="acp://pi", command="pi-acp", args=[])


class _FakeSession:
    def __init__(self, chunks):
        self.chunks = chunks

    def __call__(self, timeout_seconds, **_):
        chunks = self.chunks

        class _Ctx:
            def __enter__(self_inner):
                def _request(method, params, *, text_parts=None, reasoning_parts=None):
                    if method == "session/prompt":
                        text_parts.extend(chunks)
                    return {}
                return {"sessionId": "s1"}, _request

            def __exit__(self_inner, *a):
                return False

        return _Ctx()


FAILED_TURN = [
    "Retrying (attempt 1/3, waiting 2s)...", "Retrying (attempt 2/3, waiting 4s)...",
    "Retrying (attempt 3/3, waiting 8s)...", "Retry finished, resuming.",
]


def test_retry_chatter_only_turn_raises_connection_error(monkeypatch):
    client = _client()
    monkeypatch.setattr(client, "_session", _FakeSession(FAILED_TURN))
    with pytest.raises(ConnectionError, match="provider retries exhausted"):
        client._run_prompt("hi", timeout_seconds=5)


def test_failed_turn_is_classified_as_network_error(monkeypatch):
    from agent.error_classifier import FailoverReason, classify_api_error
    client = _client()
    monkeypatch.setattr(client, "_session", _FakeSession(FAILED_TURN))
    with pytest.raises(ConnectionError) as ei:
        client._run_prompt("hi", timeout_seconds=5)
    assert classify_api_error(ei.value).reason is FailoverReason.timeout


def test_recovered_retry_keeps_real_text_and_drops_status_chatter(monkeypatch):
    client = _client()
    monkeypatch.setattr(client, "_session", _FakeSession(
        ["Retrying (attempt 1/3, waiting 2s)...", "Retry finished, resuming.", "Real answer."]))
    text, _ = client._run_prompt("hi", timeout_seconds=5)
    assert text == "Real answer."


def test_normal_turn_unchanged(monkeypatch):
    client = _client()
    monkeypatch.setattr(client, "_session", _FakeSession(["Hello ", "world"]))
    assert client._run_prompt("hi", timeout_seconds=5)[0] == "Hello world"
