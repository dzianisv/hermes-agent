"""Round 2+ re-review brief: the dispatcher no longer runs git diff itself (it ran in a
scratch workspace under ~/.hermes, hit the wrong repo, and fell back to full reviews).
The brief now names the reviewed sha and the handed-in head and tells the reviewer to run
``git diff`` in the code repo itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_review_brief as krb
from tests.hermes_cli._kanban_modules import KanbanModules

kb = KanbanModules()


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kb, "_active_pr_guard_applies", lambda _url: (True, None))
    kb.init_db()
    c = kb.connect()
    yield c
    c.close()


def _review_round(conn, tid, head):
    claimed = kb.claim_task(conn, tid)
    kb.add_comment(conn, tid, author="impl", body="Opened https://github.com/e/r/pull/1 for review.")
    md = {"head_sha": head} if head else None
    assert kb.request_review(conn, tid, summary="PR ready", reviewer="reviewer",
                             metadata=md, expected_run_id=claimed.current_run_id)


def test_request_review_records_handed_in_head(conn):
    tid = kb.create_task(conn, title="t", assignee="impl")
    _review_round(conn, tid, "c" * 40)
    payload = json.loads(kb._latest_event(conn, tid, "review_requested")["payload"])
    assert payload.get("head_sha") == "c" * 40


def test_rereview_brief_names_both_shas_and_instructs_reviewer(conn):
    tid = kb.create_task(conn, title="t", assignee="impl")
    _review_round(conn, tid, "a" * 40)
    kb.claim_review_task(conn, tid)
    ok, why = kb.request_changes(conn, tid, reason="fix it", metadata={"reviewed_sha": "a" * 40})
    assert ok, why
    _review_round(conn, tid, "b" * 40)
    out = krb.build_review_brief(conn, kb.get_task(conn, tid), fetch=lambda _l: (None, "x"))
    assert "a" * 40 in out and "b" * 40 in out
    assert f"git diff {'a' * 40}..{'b' * 40}" in out
    assert "verify each prior finding" in out and "no new findings on unchanged code unless critical" in out
    assert "fall back" not in out and "unavailable (" not in out


def test_rereview_brief_without_handed_in_head_uses_HEAD(conn):
    tid = kb.create_task(conn, title="t", assignee="impl")
    _review_round(conn, tid, None)
    kb.claim_review_task(conn, tid)
    ok, why = kb.request_changes(conn, tid, reason="fix it", metadata={"reviewed_sha": "a" * 40})
    assert ok, why
    _review_round(conn, tid, None)
    out = krb.build_review_brief(conn, kb.get_task(conn, tid), fetch=lambda _l: (None, "x"))
    assert f"git diff {'a' * 40}..HEAD" in out


def test_diff_helpers_removed():
    assert not hasattr(krb, "_git_diff_since")
    assert not hasattr(krb, "_repos_containing")
