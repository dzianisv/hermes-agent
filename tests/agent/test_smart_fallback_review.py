"""Recovery must not substitute an untested primary or truncate cached routes."""
from types import SimpleNamespace


def test_recovery_does_not_restore_different_catalog_primary(monkeypatch):
    from agent import smart_provider_fallback as smart
    from agent.agent_runtime_helpers import restore_primary_runtime
    monkeypatch.setattr(smart, "enabled", lambda: True)
    agent = SimpleNamespace(
        provider="openrouter", model="verified:free",
        _primary_runtime={"provider": "openrouter", "model": "untested:free"},
        _smart_free_models={"verified:free", "untested:free"},
        _smart_free_available=True, _fallback_activated=True,
    )
    assert restore_primary_runtime(agent) is False
    assert agent.model == "verified:free"


def test_cached_catalog_keeps_spares_and_releases_suppression(monkeypatch):
    from agent import openrouter_free_fallback as free
    from hermes_cli import models
    monkeypatch.setattr(free, "_read_config", lambda: {"enabled": True})
    monkeypatch.setattr(free, "_free_model_cache", (0, []))
    monkeypatch.setattr(free, "_model_health", {})
    catalog = {f"route-{i}:free": {"id": f"route-{i}:free",
        "pricing": {"prompt": "0", "completion": "0"},
        "supported_parameters": ["tools"]} for i in range(3)}
    calls = []
    def fetch(*args, **kwargs):
        calls.append(True)
        return [], catalog
    monkeypatch.setattr(models, "_fetch_live_catalog_index", fetch)
    first = free.fetch_free_openrouter_models(max_entries=1)[0]["model"]
    for _ in range(free._config_suppress_after_failures()):
        free.record_model_failure(first)
    assert len(free.fetch_free_openrouter_models(max_entries=3)) == 2
    free._model_health[first]["last_failure_ts"] = 0
    assert len(free.fetch_free_openrouter_models(max_entries=3)) == 3
    assert len(calls) == 1
