"""Harness fix #20: change requests carry a PASS/FAIL finding per R/ACCEPTANCE line;
round 2+ review briefs carry prior findings and only the diff since the last reviewed sha.

Real kanban DB under a temp HERMES_HOME and a real git repo for the incremental diff.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_review_brief as krb

BODY = "Do the thing.\nR1: record rc\nR2: classify exits\nACCEPTANCE: tests green\n"


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _in_review(conn, body=BODY):
    tid = kb.create_task(conn, title="t", body=body, assignee="dev")
    claimed = kb.claim_task(conn, tid)
    assert kb.request_review(conn, tid, summary="ready", reviewer="reviewer",
                             expected_run_id=claimed.current_run_id)
    return tid, kb.claim_review_task(conn, tid)


def _f(i, v="FAIL", e="see log line 3"):
    return {"id": i, "verdict": v, "evidence": e}


def test_requirement_ids_from_card():
    assert krb.requirement_ids([BODY]) == ["R1", "R2", "ACCEPTANCE"]
    two = "ACCEPTANCE: a\nACCEPTANCE: b\nSCOPE: x\nPROOF: y"
    assert krb.requirement_ids([two]) == ["ACCEPTANCE-1", "ACCEPTANCE-2"]
    assert krb.requirement_ids(["SCOPE: only\nPROOF: p"]) == []


def test_request_changes_refused_without_findings_lists_missing_ids(kanban_home):
    with kbc.connect() as conn:
        tid, rc = _in_review(conn)
        ok, detail = kb.request_changes(conn, tid, reason="fix", expected_run_id=rc.current_run_id)
        assert ok is False
        assert "R1, R2, ACCEPTANCE" in detail
        ok, detail = kb.request_changes(
            conn, tid, reason="fix", expected_run_id=rc.current_run_id,
            metadata={"findings": [_f("R1", "PASS"), _f("r2")]})
        assert ok is False and "missing ids: ACCEPTANCE" in detail
        ok, detail = kb.request_changes(
            conn, tid, reason="fix", expected_run_id=rc.current_run_id,
            metadata={"findings": [_f("R1", "MAYBE"), _f("R2", e=""), _f("ACCEPTANCE")]})
        assert ok is False
        assert "R1: verdict must be PASS or FAIL" in detail and "R2: evidence is empty" in detail
        # Refusal leaves the review run untouched.
        t = kb.get_task(conn, tid)
        assert t.status == "running" and t.current_run_id == rc.current_run_id


def test_request_changes_accepts_complete_findings_and_records_them(kanban_home):
    with kbc.connect() as conn:
        tid, rc = _in_review(conn)
        findings = [_f("R1", "PASS"), _f("R2"), _f("ACCEPTANCE")]
        ok, impl = kb.request_changes(
            conn, tid, reason="fix R2", expected_run_id=rc.current_run_id,
            metadata={"findings": findings, "reviewed_sha": "abc1234"})
        assert (ok, impl) == (True, "dev")
        prior = krb.prior_review(conn, tid)
        assert prior["findings"] == findings and prior["reviewed_sha"] == "abc1234"


def test_card_without_requirement_lines_needs_no_findings(kanban_home):
    with kbc.connect() as conn:
        tid, rc = _in_review(conn, body="SCOPE: small\nno numbered requirements")
        ok, impl = kb.request_changes(conn, tid, reason="fix", expected_run_id=rc.current_run_id)
        assert (ok, impl) == (True, "dev")


def test_tool_handler_passes_metadata_and_refuses(kanban_home, monkeypatch):
    from tools import kanban_tools as kt
    with kbc.connect() as conn:
        tid, rc = _in_review(conn)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(rc.current_run_id))
    out = json.loads(kt._handle_request_changes({"reason": "fix"}))
    assert "missing" in out.get("error", "").lower() and "R1" in out["error"]
    out = json.loads(kt._handle_request_changes({
        "reason": "fix", "metadata": {"findings": [_f("R1"), _f("R2"), _f("ACCEPTANCE")]}}))
    assert out.get("ok") is True, out


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True,
                          text=True).stdout.strip()


def test_round2_brief_has_prior_findings_and_incremental_diff(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("old-content\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "one")
    reviewed = _git(repo, "rev-parse", "HEAD")
    with kbc.connect() as conn:
        tid, rc = _in_review(conn)
        first = krb.build_review_brief(conn, kb.get_task(conn, tid), fetch=lambda l: (None, "x"))
        assert "RE-REVIEW" not in first and "R1, R2, ACCEPTANCE" in first
        ok, _ = kb.request_changes(
            conn, tid, reason="fix R2", expected_run_id=rc.current_run_id,
            metadata={"findings": [_f("R1", "PASS", "rc logged"), _f("R2", "FAIL", "no classifier"),
                                   _f("ACCEPTANCE", "FAIL", "2 tests red")],
                      "reviewed_sha": reviewed})
        assert ok
        (repo / "a.txt").write_text("old-content\nnew-classifier\n")
        _git(repo, "commit", "-qam", "two")
        brief = krb.build_review_brief(conn, kb.get_task(conn, tid), fetch=lambda l: (None, "x"),
                                       workspace=str(repo))
    assert "RE-REVIEW (round 2)" in brief
    assert "R2: FAIL — no classifier" in brief and "R1: PASS — rc logged" in brief
    assert f"since last reviewed sha {reviewed}" in brief
    assert "-old-content" not in brief  # only the delta, not the whole file


def test_round2_brief_without_sha_says_so_and_falls_back(kanban_home):
    with kbc.connect() as conn:
        tid, rc = _in_review(conn)
        ok, _ = kb.request_changes(
            conn, tid, reason="fix", expected_run_id=rc.current_run_id,
            metadata={"findings": [_f("R1"), _f("R2"), _f("ACCEPTANCE")]})
        assert ok
        assert "reviewed_sha" not in krb.prior_review(conn, tid)
        brief = krb.build_review_brief(conn, kb.get_task(conn, tid), fetch=lambda l: (None, "x"))
    assert "No reviewed sha was recorded" in brief and "fall back to a full review" in brief


def test_rereview_model_config(monkeypatch):
    monkeypatch.setattr(kb, "_kanban_cfg", lambda: {})
    assert krb.rereview_model() == (None, None)
    monkeypatch.setattr(kb, "_kanban_cfg",
                        lambda: {"rereview_model": "gpt-5-mini", "rereview_provider": "openai"})
    assert krb.rereview_model() == ("gpt-5-mini", "openai")
