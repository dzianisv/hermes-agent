"""Issue #11/#20: reviews must be one full pass with a structured record.

Real data: on the live board t_763828ca got 17 ``changes_requested`` rounds and
t_19bbeefb / t_8f8d7102 11 each; PRs #5390 and #5420 took 5 rounds each at 1-2
findings per round, each round re-reading the full diff, because the reviewer
verdict was free text only (no head SHA, no per-acceptance-item verdict, no
stable finding ids) and the next reviewer had no prior record to diff against.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.hermes_cli._kanban_modules import KanbanModules

kb = KanbanModules()

HEAD1 = "8c6f8ca1776b40a41c4bdfeff2760420d74aa5fd"
HEAD2 = "1b2c3d4e5f60718293a4b5c6d7e8f90123456789"
BASE = "82cf9b8f91"


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kb, "_active_pr_guard_applies", lambda _url: (True, None))
    kb.init_db()
    return home


def _to_review_claimed(conn, monkeypatch, tid=None):
    if tid is None:
        tid = kb.create_task(conn, title="ship it", assignee="software-engineer")
    c = kb.claim_task(conn, tid)
    kb.add_comment(conn, tid, author="software-engineer",
                   body="Opened https://github.com/example/repo/pull/5420")
    assert kb.request_review(conn, tid, summary="PR ready", reviewer="reviewer",
                             expected_run_id=c.current_run_id)
    rv = kb.claim_review_task(conn, tid, claimer="reviewer:1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_PROFILE", "reviewer")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(rv.current_run_id))
    return tid


def _record(head=HEAD1, items=None, findings=None):
    return {
        "base_sha": BASE, "head_sha": head,
        "items": items if items is not None else [
            {"id": "AC1", "verdict": "PASS", "evidence": "tests/x.py::t passes"},
            {"id": "AC2", "verdict": "FAIL", "evidence": "gateway.py:812 logs raw token"},
        ],
        "findings": findings if findings is not None else [
            {"id": "F1", "summary": "redact before slicing", "evidence": "gateway.py:812"},
        ],
    }


def _call(fn, args):
    return json.loads(fn(args))


def test_request_changes_requires_review_record(kanban_home, monkeypatch):
    from tools import kanban_tools as tools
    with kb.connect() as conn:
        _to_review_claimed(conn, monkeypatch)
    out = _call(tools._handle_request_changes, {"reason": "fix redaction"})
    assert "error" in out and "review_record" in out["error"]


def test_unreviewed_item_or_missing_head_is_rejected(kanban_home, monkeypatch):
    from tools import kanban_tools as tools
    with kb.connect() as conn:
        _to_review_claimed(conn, monkeypatch)
    rec = _record(items=[{"id": "AC1", "verdict": "UNREVIEWED", "evidence": ""}])
    out = _call(tools._handle_request_changes, {"reason": "x", "review_record": rec})
    assert "error" in out and "UNREVIEWED" in out["error"]
    rec = _record(); rec.pop("head_sha")
    out = _call(tools._handle_approve, {"summary": "LGTM", "review_record": rec})
    assert "error" in out and "head_sha" in out["error"]
    with kb.connect() as conn:  # nothing landed
        assert kb.get_task(conn, kb.get_task(conn, __import__("os").environ["HERMES_KANBAN_TASK"]).id).status == "running"


def test_approve_rejects_fail_items(kanban_home, monkeypatch):
    from tools import kanban_tools as tools
    with kb.connect() as conn:
        _to_review_claimed(conn, monkeypatch)
    out = _call(tools._handle_approve, {"summary": "LGTM", "review_record": _record()})
    assert "error" in out and "FAIL" in out["error"]


def test_record_stored_untruncated_and_shown_on_rereview(kanban_home, monkeypatch):
    from tools import kanban_tools as tools
    long_ev = "line evidence " * 600  # > any comment/summary cap
    with kb.connect() as conn:
        tid = _to_review_claimed(conn, monkeypatch)
    rec = _record(findings=[{"id": "F1", "summary": "redact", "evidence": long_ev}])
    out = _call(tools._handle_request_changes, {"reason": "fix F1", "review_record": rec})
    assert out.get("ok") is True, out
    with kb.connect() as conn:
        stored = kb.latest_review_record(conn, tid)
        assert stored["head_sha"] == HEAD1
        assert stored["findings"][0]["evidence"] == long_ev.strip()
        _to_review_claimed(conn, monkeypatch, tid)
        ctx = kb.build_worker_context(conn, tid)
    assert "Prior review record" in ctx and "F1" in ctx
    assert f"git diff {HEAD1}..HEAD" in ctx
    # Re-review: new findings must be classed; F1 keeps its id.
    rec2 = _record(head=HEAD2,
                   items=[{"id": "AC1", "verdict": "PASS", "evidence": "ok"},
                          {"id": "AC2", "verdict": "FAIL", "evidence": "still"}],
                   findings=[{"id": "F2", "summary": "new issue", "evidence": "a.py:1"}])
    out = _call(tools._handle_request_changes, {"reason": "F2", "review_record": rec2})
    assert "error" in out and "class" in out["error"]
    rec2["findings"][0]["class"] = "regression"
    out = _call(tools._handle_request_changes, {"reason": "F2", "review_record": rec2})
    assert out.get("ok") is True, out


def test_approve_with_full_pass_record(kanban_home, monkeypatch):
    from tools import kanban_tools as tools
    with kb.connect() as conn:
        tid = _to_review_claimed(conn, monkeypatch)
    rec = _record(items=[{"id": "AC1", "verdict": "PASS", "evidence": "t passes"}], findings=[])
    out = _call(tools._handle_approve, {"summary": "LGTM", "review_record": rec})
    assert out.get("ok") is True, out
    with kb.connect() as conn:
        assert kb.latest_review_record(conn, tid)["verdict"] == "approved"
