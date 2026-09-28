"""The /model picker must keep serving a provider's live catalog across routine auth-store writes.

Production (2026-09-27): the Telegram picker intermittently showed Copilot's static fallback while
``provider_models_cache.json`` held the live catalog, because the credential pool's usage-counter
writes to ``auth.json`` re-keyed every provider's cache row and the cache-only read then treated
the row as belonging to other credentials.
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import patch

import pytest


def _write_auth(home, *, copilot_token: str, request_count: int, mtime_ns: int) -> None:
    path = home / "auth.json"
    path.write_text(json.dumps({
        "version": 1,
        "credential_pool": {
            "copilot": [{
                "id": "gh-cli", "label": "gh", "auth_type": "api_key", "priority": 0,
                "source": "manual", "access_token": copilot_token,
                "request_count": request_count, "last_status": "ok" if request_count else None,
                "last_status_at": time.time() if request_count else None,
            }],
            "openrouter": [{
                "id": "or", "label": "or", "auth_type": "api_key", "priority": 0,
                "source": "env:OPENROUTER_API_KEY", "access_token": "sk-or-x",
                "request_count": request_count * 3,
            }],
        },
    }), encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def test_copilot_fingerprint_survives_routine_auth_writes_but_tracks_credential_changes(
        tmp_path, monkeypatch):
    import hermes_cli.models as mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ("GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_account_a")

    _write_auth(tmp_path, copilot_token="gho_pool_a", request_count=0, mtime_ns=1_000_000_000)
    baseline = mod._credential_fingerprint("copilot")

    # Usage counters / status stamps rewrite auth.json with a new mtime: same credentials.
    _write_auth(tmp_path, copilot_token="gho_pool_a", request_count=7, mtime_ns=2_000_000_000)
    assert mod._credential_fingerprint("copilot") == baseline

    # A different token in the environment is a real account switch.
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_account_b")
    assert mod._credential_fingerprint("copilot") != baseline

    # So is a different pooled Copilot credential (the catalog resolver falls back to the pool).
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_account_a")
    _write_auth(tmp_path, copilot_token="gho_pool_b", request_count=7, mtime_ns=3_000_000_000)
    assert mod._credential_fingerprint("copilot") != baseline


@pytest.fixture
def _no_swr_inflight():
    import hermes_cli.models as mod
    with mod._swr_refresh_lock:
        mod._swr_refresh_inflight.clear()
    yield
    with mod._swr_refresh_lock:
        mod._swr_refresh_inflight.clear()


def test_cache_only_read_serves_live_row_stored_under_other_credentials(
        tmp_path, monkeypatch, _no_swr_inflight):
    import hermes_cli.models as mod
    from hermes_cli.models_catalog_static import _PROVIDER_MODELS

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    live = ["gpt-5.9-copilot-live-only", *_PROVIDER_MODELS["copilot"][:3]]
    assert live[0] not in _PROVIDER_MODELS["copilot"]
    (tmp_path / "provider_models_cache.json").write_text(json.dumps({
        "copilot": {"fp": "written-under-earlier-credentials", "at": time.time() - 30, "models": live},
    }), encoding="utf-8")

    with patch.object(mod, "_spawn_swr_refresh") as spawn, \
         patch.object(mod, "provider_model_ids") as blocking_fetch:
        served = mod.cached_provider_model_ids("copilot", non_blocking=True)

    assert served == live
    spawn.assert_called_once_with("copilot")  # the background refresh re-keys the row
    blocking_fetch.assert_not_called()
