"""OpenRouter retries use the real HTTP client, recovery loop, pool and fallback resolver.

Only waits are virtualized; no real provider or operator credentials are used.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from agent.credential_pool import CredentialPool, PooledCredential
from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Raw, Text

_PRIMARY = "test/primary:free"
_BACKUP = "test/backup:free"
_KEYS = ("fake-account-one", "fake-account-two")


def _agent(monkeypatch, tmp_path, server, *, budget, pool=False, fallback=True):
    from run_agent import AIAgent

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(json.dumps({
        "agent": {"api_max_retries": budget, "auto_recovery_cycles": 0},
        "display": {"suppress_warning_notifications": True},
        "fallback": {"min_switch_reset_seconds": 120},
        "openrouter_free_fallback": {"enabled": False},
        "auxiliary": {"title_generation": {"enabled": False}},
        "updates": {"check": False},
    }))
    # Unknown test model names must not cause catalog network probes.
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **k: 128000)
    credentials = CredentialPool("openrouter", [
        PooledCredential(provider="openrouter", id=f"account-{i}", label=f"account-{i}",
                         auth_type="api_key", priority=i, source="manual",
                         access_token=key, base_url=server.base_url)
        for i, key in enumerate(_KEYS)
    ]) if pool else None
    agent = AIAgent(
        provider="openrouter", model=_PRIMARY, api_key=_KEYS[0], base_url=server.base_url,
        enabled_toolsets=["file"], quiet_mode=True, skip_context_files=True, skip_memory=True,
        skip_background_review=True, save_trajectories=False, credential_pool=credentials,
        fallback_model=[{"provider": "openrouter", "model": _BACKUP,
                         "base_url": server.base_url, "api_key": _KEYS[0]}] if fallback else [],
    )
    assert agent._api_max_retries == budget  # exercise config propagation, not a manually set field
    return agent


def _virtual_waits(monkeypatch, *, cancel=False):
    from agent import turn_api_error

    waits = []
    real_sleep = turn_api_error.interruptible_backoff_sleep

    def wait(agent, seconds, retry, **kwargs):
        waits.append(seconds)
        if cancel:
            # Interrupt during a real wait, not before entry or at a mocked cancellation seam.
            timer = threading.Timer(0.05, lambda: agent.interrupt(hard_cancel=True))
            started = time.monotonic()
            timer.start()
            try:
                result = real_sleep(agent, seconds, retry, **kwargs)
            finally:
                timer.join()
            assert time.monotonic() - started < 5.0
            return result
        return None

    monkeypatch.setattr(turn_api_error, "interruptible_backoff_sleep", wait)
    return waits


def _upstream_error():
    return Raw(json.dumps({"error": {"code": 429, "message": "Provider returned error",
                                    "metadata": {"provider_name": "test upstream",
                                                 "raw": "temporarily rate limited"}}}), status=429)


@pytest.mark.real_retry_backoff
@pytest.mark.parametrize("kind,failures,budget,retry_after,cancel", [
    ("generic", 2, 6, None, False),
    ("upstream", 2, 6, None, False),
    ("overload", 2, 6, None, False),
    ("generic", 3, 3, None, False),
    ("upstream", 3, 3, None, False),
    ("generic", 6, 6, None, False),
    ("generic", 7, 8, None, False),
    ("generic", 2, 6, 0.5, False),
    ("generic", 2, 6, 120, False),
    ("generic", 1, 6, 86400, False),
    ("generic", 1, 6, 120, True),
])
def test_transient_429_spends_budget_before_fallback(
    monkeypatch, tmp_path, caplog, capsys, kind, failures, budget, retry_after, cancel,
):
    failure = (_upstream_error() if kind == "upstream" else
               Error(429, "upstream at capacity" if kind == "overload" else "Rate limit exceeded",
                     retry_after=retry_after))
    waits = _virtual_waits(monkeypatch, cancel=cancel)
    with FakeLLMServer([failure] * failures + [Text("recovered")], api_key=_KEYS) as server:
        agent = _agent(monkeypatch, tmp_path, server, budget=budget)
        result = agent.run_conversation("Please answer this test request.")
        requests = server.main_requests()

    if cancel:
        assert result["interrupted"] and not result["completed"]
        assert len(requests) == 1 and len(waits) == 1
        assert not agent._fallback_activated
    else:
        assert result["completed"] and result["final_response"] == "recovered"
        switched = failures >= budget
        assert [r["model"] for r in requests] == [_PRIMARY] * failures + [
            _BACKUP if switched else _PRIMARY]
        assert agent._fallback_activated == switched
        assert len(waits) == min(failures, budget - 1)
        for attempt, wait in enumerate(waits, 1):
            exponential = min(2.0 * 2 ** (attempt - 1), 60.0)
            minimum = max(exponential, min(retry_after or 0, 600))
            assert minimum <= wait <= max(exponential * 1.5, min(retry_after or 0, 600))
        # Same-model retries must preserve the complete request/cache prefix.
        assert all(r["messages"] == requests[0]["messages"] for r in requests[:failures])
    assert "API call failed" in caplog.text  # suppression is presentation-only
    assert "switching to fallback" not in capsys.readouterr().out.lower()


@pytest.mark.parametrize("status,message", [
    (429, "Rate limit exceeded: free-models-per-day. Add credits to increase the daily limit."),
    (429, "Daily request limit exceeded. Try again tomorrow."),
    (402, "Insufficient credits"),
    (401, "Invalid API key"),
    (403, "Request blocked by content policy"),
])
def test_terminal_limits_rotate_accounts_without_transient_waits(
    monkeypatch, tmp_path, caplog, status, message,
):
    waits = _virtual_waits(monkeypatch)
    with FakeLLMServer(lambda _: Error(status, message, retry_after=86400), api_key=_KEYS) as server:
        agent = _agent(monkeypatch, tmp_path, server, budget=6, pool=True, fallback=False)
        result = agent.run_conversation("Please answer this test request.")
        requests = [r for r in server.requests if r["kind"] == "main"]

    assert result["failed"] and not result["completed"]
    assert message.lower() in result["error"].lower()
    assert waits == []
    if status == 429:
        assert result["failure_reason"] == "rate_limit" and result["failure_retryable"] is False
        assert "daily" in result["final_response"].lower() or "free-models-per-day" in result["final_response"]
        assert all(e.last_error_reset_at > time.time() + 86000 for e in agent._credential_pool.entries())
    assert all(r["body"]["model"] == _PRIMARY for r in requests)
    if status != 403:  # a content-policy refusal says nothing about the other account
        assert [r["auth"] for r in requests] == [f"Bearer {key}" for key in _KEYS]
        assert not agent._credential_pool.has_available(model=_PRIMARY)
    assert "API call failed" in caplog.text
