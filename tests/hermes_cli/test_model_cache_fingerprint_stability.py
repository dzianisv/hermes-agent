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

