"""An explicit non-OpenRouter primary must never be replaced by the free-first policy."""
from types import SimpleNamespace

import pytest

from agent import smart_provider_fallback as smart


@pytest.mark.parametrize("primary_runtime", [
    {"provider": "opencode-go", "model": "deepseek-v4.1-flash"},
    None,  # no runtime snapshot: the active provider decides
])
def test_non_openrouter_primary_is_never_hijacked(primary_runtime, monkeypatch):
    import agent.openrouter_free_fallback as free

    monkeypatch.setattr(smart, "enabled", lambda: True)

    def fail(**kwargs):
        raise AssertionError("free-model catalog must not be fetched for an opencode-go primary")

    monkeypatch.setattr(free, "fetch_free_openrouter_models", fail)
    sentinel = [{"provider": "openrouter", "model": "configured:free"}]
    agent = SimpleNamespace(
        provider="opencode-go",
        model="deepseek-v4.1-flash",
        _primary_runtime=primary_runtime,
        _fallback_chain=sentinel,
        _fallback_index=0,
        _smart_free_models={"cached:free"},
        _smart_free_available=True,
        _smart_codex_models={"gpt-6-sol"},
        _rate_limited_until=0,
    )

    smart.prepare_turn(agent)

    assert agent._fallback_chain is sentinel
    assert agent._fallback_index == 0
    assert agent._smart_free_models == {"cached:free"}
    assert agent._smart_free_available is True
    assert agent._smart_codex_models == {"gpt-6-sol"}
    assert agent._rate_limited_until == 0


def test_openrouter_primary_still_takes_free_first_path(monkeypatch):
    import agent.openrouter_free_fallback as free

    monkeypatch.setattr(smart, "enabled", lambda: True)
    monkeypatch.setattr(smart, "_key", lambda agent=None: "account")
    monkeypatch.setattr(smart, "free_available", lambda key, models: True)
    monkeypatch.setattr(smart, "codex_models", lambda: [])
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return [{"provider": "openrouter", "model": "a:free"}]

    monkeypatch.setattr(free, "fetch_free_openrouter_models", fetch)
    agent = SimpleNamespace(
        provider="openrouter",
        model="a:free",
        _primary_runtime={"provider": "openrouter", "model": "a:free"},
    )

    smart.prepare_turn(agent)

    assert calls
    assert agent._fallback_chain == [{"provider": "openrouter", "model": "a:free"}]
    assert agent._smart_free_models == {"a:free"}
    assert agent._smart_free_available is True
    assert agent._rate_limited_until == 0
