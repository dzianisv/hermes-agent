"""facts_hash gate: unchanged script facts skip the agent run and delivery entirely."""

from __future__ import annotations

import logging

from tests.cron.test_monitor_kind import _install_agent_stubs, _write_script, hermes_env  # noqa: F401


def _make_job(home, body: str, **extra):
    from cron.jobs import create_job, update_job

    _write_script(home, "facts.sh", body)
    (home / "config.yaml").write_text("cron:\n  preflight: false\n", encoding="utf-8")
    job = create_job(prompt="Report anomalies", schedule="every 60m", script="facts.sh",
                     deliver="local")
    return update_job(job["id"], {"facts_hash": True, **extra})


def _fire(job_id, monkeypatch, delivered: list):
    import cron.scheduler as sched
    from cron.jobs import get_job

    monkeypatch.setattr(sched, "_deliver_result",
                        lambda job, content, **_kw: delivered.append(content) or None)
    job = get_job(job_id)
    job["deliver"] = "telegram:1"  # force the delivery path; _deliver_result is stubbed
    assert sched.run_one_job(job) is True
    return get_job(job_id)


def test_same_facts_twice_second_is_skipped(hermes_env, monkeypatch, caplog):
    job = _make_job(hermes_env, "echo 'tenant 8788 failed'\n")
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)
    delivered: list = []

    after = _fire(job["id"], monkeypatch, delivered)
    assert observed["agent_runs"] == 1
    assert len(delivered) == 1
    assert after["facts_state"]["last_hash"]

    with caplog.at_level(logging.INFO, logger="cron.scheduler"):
        _fire(job["id"], monkeypatch, delivered)
    assert observed["agent_runs"] == 1, "unchanged facts must not run the agent"
    assert len(delivered) == 1, "unchanged facts must not deliver"
    assert "skipped: facts unchanged" in caplog.text


def test_changed_facts_run_again(hermes_env, monkeypatch):
    job = _make_job(hermes_env, "echo 'state A'\n")
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)
    delivered: list = []
    first = _fire(job["id"], monkeypatch, delivered)
    _write_script(hermes_env, "facts.sh", "echo 'state B'\n")
    second = _fire(job["id"], monkeypatch, delivered)
    assert observed["agent_runs"] == 2
    assert len(delivered) == 2
    assert first["facts_state"]["last_hash"] != second["facts_state"]["last_hash"]


def test_ignore_patterns_strip_volatile_text(hermes_env, monkeypatch):
    job = _make_job(hermes_env, "date +'failed for %N'\n",
                    facts_ignore=[r"failed for \d+"])
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)
    delivered: list = []
    _fire(job["id"], monkeypatch, delivered)
    _fire(job["id"], monkeypatch, delivered)
    assert observed["agent_runs"] == 1


def test_manual_run_bypasses_gate(hermes_env, monkeypatch):
    from cron.jobs import trigger_job

    job = _make_job(hermes_env, "echo same\n")
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)
    delivered: list = []
    _fire(job["id"], monkeypatch, delivered)
    trigger_job(job["id"])
    _fire(job["id"], monkeypatch, delivered)
    assert observed["agent_runs"] == 2


def test_failed_delivery_does_not_store_hash(hermes_env, monkeypatch):
    import cron.scheduler as sched
    from cron.jobs import get_job

    job = _make_job(hermes_env, "echo same\n")
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)
    monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: "telegram down")
    j = get_job(job["id"])
    j["deliver"] = "telegram:1"
    sched.run_one_job(j)
    assert not (get_job(job["id"]).get("facts_state") or {}).get("last_hash")
    _fire(job["id"], monkeypatch, [])
    assert observed["agent_runs"] == 2


def test_job_without_flag_unchanged(hermes_env, monkeypatch):
    from cron.jobs import create_job

    _write_script(hermes_env, "facts.sh", "echo same\n")
    (hermes_env / "config.yaml").write_text("cron:\n  preflight: false\n", encoding="utf-8")
    job = create_job(prompt="p", schedule="every 60m", script="facts.sh", deliver="local")
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)
    _fire(job["id"], monkeypatch, [])
    after = _fire(job["id"], monkeypatch, [])
    assert observed["agent_runs"] == 2
    assert "facts_state" not in after
