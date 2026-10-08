"""Workflow stages: ``kanban.stages`` + ``hermes kanban step`` / ``--step``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_stages as kst

STAGES = [
    {"key": "todo", "owner": "product-lead-agentpod", "status": "todo"},
    {"key": "design", "owner": "architect-critic-fable", "status": "ready"},
    {"key": "critic", "owner": "architect-critic-astra", "status": "ready"},
    {"key": "development", "owner": "software-engineer", "status": "ready"},
    {"key": "review", "owner": "reviewer", "status": "review"},
    {"key": "ready-for-merge", "owner": "software-engineer", "status": "ready"},
    {"key": "ready-for-deploy", "owner": "product-lead-agentpod", "status": "ready"},
    {"key": "done", "owner": None, "status": "done"},
]


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kst, "load_stages", lambda raw=None: kst._parse(STAGES if raw is None else raw))
    kb.init_db()
    return home


def _run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    kc.build_parser(sub)
    return kc.kanban_command(parser.parse_args(["kanban", *argv]))


def _show(tid: str, capsys) -> dict:
    capsys.readouterr()
    assert _run(["show", tid, "--json"]) == 0
    return json.loads(capsys.readouterr().out)


def _create(capsys, *extra) -> str:
    capsys.readouterr()
    assert _run(["create", "stage test", "--triage", "--json", *extra]) == 0
    return json.loads(capsys.readouterr().out)["id"]


def test_create_with_step_records_stage(kanban_home, capsys):
    tid = _create(capsys, "--step", "design")
    task = _show(tid, capsys)["task"]
    assert task["current_step_key"] == "design"
    assert task["status"] == "triage"  # --step records the stage only; no handoff


def test_create_with_unknown_step_rejected(kanban_home, capsys):
    assert _run(["create", "x", "--triage", "--step", "nope"]) != 0
    assert "unknown stage" in capsys.readouterr().err


def test_edit_step(kanban_home, capsys):
    tid = _create(capsys)
    assert _run(["edit", tid, "--step", "critic"]) == 0
    assert _show(tid, capsys)["task"]["current_step_key"] == "critic"
    assert _run(["edit", tid, "--step", "bogus"]) != 0


def test_step_reassigns_owner_and_status(kanban_home, capsys):
    tid = _create(capsys)
    assert _run(["step", tid, "design", "--note", "spec it"]) == 0
    out = _show(tid, capsys)
    t = out["task"]
    assert (t["current_step_key"], t["assignee"], t["status"]) == ("design", "architect-critic-fable", "ready")
    assert not any(c["body"].startswith("STAGE:") for c in out["comments"])
    assert any(e["kind"] == "stage_changed" and e["payload"]["to"] == "design"
               and e["payload"].get("note") == "spec it" for e in out["events"])


def test_step_review_stage_uses_review_lane(kanban_home, capsys):
    tid = _create(capsys)
    assert _run(["step", tid, "review"]) == 0
    t = _show(tid, capsys)["task"]
    assert (t["assignee"], t["status"]) == ("reviewer", "review")


def test_step_next_advances(kanban_home, capsys):
    tid = _create(capsys)
    assert _run(["step", tid, "design"]) == 0
    assert _run(["step", tid, "--next"]) == 0
    t = _show(tid, capsys)["task"]
    assert (t["current_step_key"], t["assignee"], t["status"]) == ("critic", "architect-critic-astra", "ready")


def test_step_next_from_no_stage_starts_at_first(kanban_home, capsys):
    tid = _create(capsys)
    assert _run(["step", tid, "--next"]) == 0
    t = _show(tid, capsys)["task"]
    assert (t["current_step_key"], t["status"]) == ("todo", "todo")


def test_step_unknown_key_rejected(kanban_home, capsys):
    tid = _create(capsys)
    capsys.readouterr()
    assert _run(["step", tid, "nope"]) != 0
    assert "unknown stage" in capsys.readouterr().err
    assert _show(tid, capsys)["task"]["current_step_key"] is None


def test_step_keep_status_leaves_parked_card_parked(kanban_home, capsys):
    tid = _create(capsys)
    assert _run(["step", tid, "design", "--keep-status"]) == 0
    t = _show(tid, capsys)["task"]
    assert (t["current_step_key"], t["assignee"], t["status"]) == ("design", "architect-critic-fable", "triage")


def test_step_refuses_claimed_running_task(kanban_home, capsys):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="r", assignee="software-engineer")
        assert kb.claim_task(conn, tid) is not None
    assert _run(["step", tid, "review"]) != 0


def test_step_respects_parent_gate(kanban_home, capsys):
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="p", triage=True)
        child = kb.create_task(conn, title="c", parents=(parent,), triage=True)
    assert _run(["step", child, "development"]) == 0
    t = _show(child, capsys)["task"]
    assert (t["assignee"], t["status"]) == ("software-engineer", "todo")


def test_list_step_key_filter_and_column(kanban_home, capsys):
    a = _create(capsys, "--step", "design")
    b = _create(capsys, "--step", "critic")
    capsys.readouterr()
    assert _run(["list", "--step-key", "design", "--json"]) == 0
    ids = [t["id"] for t in json.loads(capsys.readouterr().out)]
    assert ids == [a] and b not in ids
    assert _run(["list"]) == 0
    out = capsys.readouterr().out
    assert "design" in out and "critic" in out


def test_show_text_has_stage(kanban_home, capsys):
    tid = _create(capsys, "--step", "design")
    capsys.readouterr()
    assert _run(["show", tid]) == 0
    assert "stage:" in capsys.readouterr().out


def test_parse_rejects_bad_config():
    with pytest.raises(ValueError):
        kst._parse([{"key": "a", "status": "running"}])
    with pytest.raises(ValueError):
        kst._parse([{"key": "a"}, {"key": "a"}])
