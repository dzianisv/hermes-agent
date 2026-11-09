"""Regression tests for the cron model catalog guard.

Why: a stored cron pin can outlive the provider catalog that once served it. The measured incident
spent a full cron turn on an OpenRouter slug whose retired ``:free`` variant only failed after the
provider request, so the guard must stop bad pins at write time and before dispatch.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cron import scheduler
from cron.model_guard import guard_dispatch, unserved_model_diagnosis


class _JsonResponse:
    """Minimal ``urlopen`` response wrapper for hermetic JSON probe tests."""

    def __init__(self, payload: dict):
        self._body = __import__("json").dumps(payload).encode("utf-8")

    def __enter__(self) -> "_JsonResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _patch_openrouter_probe(monkeypatch: pytest.MonkeyPatch, payload: dict | None = None) -> None:
    """Keep tests offline while controlling the auth/key probe result."""
    import cron.model_guard as model_guard

    if payload is None:
        monkeypatch.setattr(
            model_guard.urllib.request,
            "urlopen",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
        )
        return
    monkeypatch.setattr(
        model_guard.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _JsonResponse(payload),
    )


def _runtime(provider: str = "openrouter") -> dict[str, str]:
    return {
        "api_key": "test-key",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": provider,
        "api_mode": "chat_completions",
    }


def _write_config(tmp_path: Path, *, model: str, provider: str = "openrouter") -> None:
    (tmp_path / "config.yaml").write_text(
        f"model:\n  default: {model}\n  provider: {provider}\n",
        encoding="utf-8",
    )


def _run_job_until_construct(
    tmp_path: Path,
    job: dict,
    *,
    construct_side_effect: Exception,
) -> tuple[bool, str | None, int]:
    """Drive ``run_job`` until the agent-construction seam and report its outcome."""

    def _resolve(**_kwargs):
        return _runtime()

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._get_hermes_home", return_value=tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=MagicMock()), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_resolve), \
         patch("tools.mcp_tool_discovery.discover_mcp_tools", return_value=[]), \
         patch("cron.scheduler._construct_cron_agent", side_effect=construct_side_effect) as construct:
        success, _output, _final, error = scheduler.run_job(dict(job))
    return success, error, construct.call_count


def _jobs_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    main_model: str = "main-model",
    main_provider: str = "openrouter",
):
    """In-memory cron job store wired like the real create/update chokepoints."""
    import cron.jobs as jobs

    (tmp_path / "config.yaml").write_text(
        f"model:\n  default: {main_model}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(jobs, "get_hermes_home", lambda: tmp_path, raising=True)
    state: dict[str, list[dict]] = {"jobs": []}
    monkeypatch.setattr(jobs, "load_jobs", lambda: list(state["jobs"]), raising=True)
    monkeypatch.setattr(
        jobs,
        "save_jobs",
        lambda rows, **_kwargs: state.__setitem__("jobs", list(rows)),
        raising=True,
    )
    resolver = MagicMock(return_value={"provider": main_provider})
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolver)
    return jobs, state


def test_unserved_model_diagnosis_passes_provider_base_url_to_validator(monkeypatch: pytest.MonkeyPatch):
    # Regression: the guard must pass the provider base URL into the validator so it can consult
    # the live /models listing instead of soft-accepting against the curated fallback.
    seen_kwargs: list[dict[str, object]] = []

    def _validator(_model: str, _provider: str, **kwargs):
        seen_kwargs.append(dict(kwargs))
        if kwargs.get("base_url"):
            return {
                "accepted": False,
                "persist": False,
                "recognized": True,
                "message": "Model was absent from the live catalog.",
            }
        return {
            "accepted": True,
            "persist": False,
            "recognized": False,
            "message": "soft-accepted because /models was unreachable",
        }

    monkeypatch.setattr("cron.model_guard.validate_requested_model", _validator)
    monkeypatch.setattr(
        "cron.model_guard.get_provider",
        lambda *_args, **_kwargs: SimpleNamespace(base_url="https://openrouter.ai/api/v1"),
    )
    monkeypatch.setattr(
        "cron.model_guard.fetch_api_models",
        lambda *_args, **_kwargs: ["minimax/minimax-m3.1-flash", "minimax/minimax-text-01"],
    )
    _patch_openrouter_probe(monkeypatch)

    diagnosis = unserved_model_diagnosis(
        "minimax/minimax-m3.1-flash:free",
        "openrouter",
        job_name="nightly brief",
    )

    assert diagnosis is not None
    assert seen_kwargs == [{"base_url": "https://openrouter.ai/api/v1"}]
    assert "Model was absent from the live catalog." in diagnosis


def test_unserved_model_diagnosis_lists_catalog_alternatives_and_key_limit(monkeypatch: pytest.MonkeyPatch):
    # Regression: a retired OpenRouter ``:free`` slug must name a serving sibling and warn that
    # the key cannot reach paid fallbacks when the account limit is $0.00.
    monkeypatch.setattr(
        "cron.model_guard.validate_requested_model",
        lambda *_args, **_kwargs: {
            "accepted": False,
            "persist": False,
            "recognized": True,
            "message": "The free launch period has ended.",
        },
    )
    monkeypatch.setattr(
        "cron.model_guard.get_provider",
        lambda *_args, **_kwargs: SimpleNamespace(base_url="https://openrouter.ai/api/v1"),
    )
    monkeypatch.setattr(
        "cron.model_guard.fetch_api_models",
        lambda *_args, **_kwargs: [
            "minimax/minimax-m3.1-flash",
            "minimax/minimax-text-01",
            "google/gemini-2.5-flash",
        ],
    )
    _patch_openrouter_probe(monkeypatch, {"data": {"limit": 0}})

    diagnosis = unserved_model_diagnosis(
        "minimax/minimax-m3.1-flash:free",
        "openrouter",
        job_name="nightly brief",
    )

    assert diagnosis is not None
    assert "`minimax/minimax-m3.1-flash:free`" in diagnosis
    assert "`minimax/minimax-m3.1-flash`" in diagnosis
    assert "$0.00" in diagnosis


def test_pre_dispatch_guard_blocks_before_agent_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Regression: the scheduler must reject an unserved slug before it opens state or constructs
    # an agent, so a doomed provider request never burns a cron turn.
    _write_config(tmp_path, model="minimax/minimax-m3.1-flash:free")
    monkeypatch.setattr(
        "cron.model_guard.validate_requested_model",
        lambda *_args, **_kwargs: {
            "accepted": False,
            "persist": False,
            "recognized": True,
            "message": "The free launch period has ended.",
        },
    )
    monkeypatch.setattr(
        "cron.model_guard.get_provider",
        lambda *_args, **_kwargs: SimpleNamespace(base_url="https://openrouter.ai/api/v1"),
    )
    monkeypatch.setattr(
        "cron.model_guard.fetch_api_models",
        lambda *_args, **_kwargs: ["minimax/minimax-m3.1-flash", "minimax/minimax-text-01"],
    )
    _patch_openrouter_probe(monkeypatch, {"data": {"limit": 0}})

    job = {"id": "guard-job", "name": "guard job", "prompt": "hello", "model": None, "provider": None}
    with patch("cron.scheduler._hermes_home", tmp_path), patch(
        "cron.scheduler._get_hermes_home", return_value=tmp_path
    ):
        with pytest.raises(RuntimeError) as excinfo:
            scheduler._load_cron_job_config(job, "guard-job", "guard job")
    assert "`minimax/minimax-m3.1-flash:free`" in str(excinfo.value)
    assert "`minimax/minimax-m3.1-flash`" in str(excinfo.value)

    success, error, construct_calls = _run_job_until_construct(
        tmp_path,
        job,
        construct_side_effect=AssertionError("agent construction must stay unreachable"),
    )
    assert success is False
    assert construct_calls == 0
    assert error is not None and "minimax/minimax-m3.1-flash:free" in error


def test_served_model_reaches_agent_construction_seam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Regression: a catalog-present slug must not trip the guard; the run should continue until
    # the normal agent-construction seam.
    _write_config(tmp_path, model="minimax/minimax-m3.1-flash")
    monkeypatch.setattr(
        "cron.model_guard.validate_requested_model",
        lambda *_args, **_kwargs: {"accepted": True, "persist": True, "recognized": True, "message": None},
    )
    _patch_openrouter_probe(monkeypatch)

    job = {"id": "ok-job", "name": "ok job", "prompt": "hello", "model": None, "provider": None}
    success, error, construct_calls = _run_job_until_construct(
        tmp_path,
        job,
        construct_side_effect=RuntimeError("agent constructed"),
    )

    assert success is False
    assert construct_calls == 1
    assert error == "RuntimeError: agent constructed"


def test_create_and_update_reject_unserved_pins_without_mutating_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    # Regression: bad pins must be refused at the storage chokepoint, including ``pinned=True``
    # copying the current main provider/model onto a job.
    jobs, state = _jobs_store(monkeypatch, tmp_path)

    def _verdict(model: str, provider: str, **_kwargs):
        accepted = model == "good-model"
        return {"accepted": accepted, "persist": accepted, "recognized": True, "message": "not served"}

    monkeypatch.setattr("cron.model_guard.validate_requested_model", _verdict)
    monkeypatch.setattr(
        "cron.model_guard.fetch_api_models",
        lambda *_args, **_kwargs: ["good-model", "minimax/minimax-m3.1-flash"],
    )
    _patch_openrouter_probe(monkeypatch)

    with pytest.raises(ValueError, match="bad-model"):
        jobs.create_job(prompt="bad", schedule="every 1 hour", model="bad-model")
    assert state["jobs"] == []

    good = jobs.create_job(prompt="good", schedule="every 1 hour", model="good-model")
    assert state["jobs"][0]["model"] == "good-model"

    with pytest.raises(ValueError, match="bad-model"):
        jobs.update_job(good["id"], {"model": "bad-model"})
    assert state["jobs"][0]["model"] == "good-model"

    unpinned = jobs.create_job(prompt="copy main", schedule="every 1 hour")
    assert state["jobs"][1]["model"] is None

    with pytest.raises(ValueError, match="main-model"):
        jobs.update_job(unpinned["id"], {"pinned": True})
    assert state["jobs"][1]["model"] is None
    assert state["jobs"][1]["provider"] is None


def test_guard_fail_open_on_validator_error_allows_job_to_proceed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    # Regression: a transient validator/catalog outage must not stop a runnable cron job.
    _write_config(tmp_path, model="minimax/minimax-m3.1-flash:free")
    calls: list[tuple[str, str]] = []

    def _boom(model: str, provider: str, **_kwargs):
        calls.append((model, provider))
        raise RuntimeError("provider unreachable")

    monkeypatch.setattr("cron.model_guard.validate_requested_model", _boom)
    _patch_openrouter_probe(monkeypatch)

    job = {"id": "fail-open", "name": "fail open", "prompt": "hello", "model": None, "provider": None}
    guard_dispatch("minimax/minimax-m3.1-flash:free", "openrouter")
    success, error, construct_calls = _run_job_until_construct(
        tmp_path,
        job,
        construct_side_effect=RuntimeError("agent constructed"),
    )

    assert calls
    assert success is False
    assert construct_calls == 1
    assert error == "RuntimeError: agent constructed"


def test_blank_provider_skips_validation_without_touching_validator(monkeypatch: pytest.MonkeyPatch):
    # Regression: no authoritative provider means no catalog check; this must stay fail-open and
    # must not ping the validator at all.
    called = False

    def _validator(*_args, **_kwargs):
        nonlocal called
        called = True
        return {"accepted": False, "persist": False, "recognized": True, "message": "boom"}

    monkeypatch.setattr("cron.model_guard.validate_requested_model", _validator)
    _patch_openrouter_probe(monkeypatch)

    assert unserved_model_diagnosis("bad-model", "") is None
    assert called is False


def test_env_kill_switch_disables_guard_without_touching_validator(monkeypatch: pytest.MonkeyPatch):
    # Regression: operators need a global off switch for the guard, and disabling it must bypass
    # validation entirely rather than reporting a false pass.
    called = False

    def _validator(*_args, **_kwargs):
        nonlocal called
        called = True
        return {"accepted": False, "persist": False, "recognized": True, "message": "boom"}

    monkeypatch.setattr("cron.model_guard.validate_requested_model", _validator)
    monkeypatch.setenv("HERMES_CRON_MODEL_GUARD", "off")
    _patch_openrouter_probe(monkeypatch)

    assert unserved_model_diagnosis("bad-model", "openrouter") is None
    guard_dispatch("bad-model", "openrouter")
    assert called is False
