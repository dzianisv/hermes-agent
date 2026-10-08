"""Issues-to-resolve #4: a reviewed card must not loop forever.

Live board: cards bounced 4-6 change-request rounds because the brief never
named the proof. The Nth change request (kanban.review_round_limit, default 3)
parks the card in triage for re-spec instead of respawning the implementer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.hermes_cli._kanban_modules import KanbanModules

kb = KanbanModules()

PR_COMMENT = "Opened https://github.com/example/repo/pull/123 for review."


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kb, "_active_pr_guard_applies", lambda _url: (True, None))
    kb.init_db()
    return home


def _round(conn, tid, first=False):
    if first:
        claimed = kb.claim_task(conn, tid)
        kb.add_comment(conn, tid, author="software-engineer", body=PR_COMMENT)
    else:
        claimed = kb.claim_task(conn, tid)
    assert kb.request_review(conn, tid, summary="PR ready", reviewer="reviewer",
                             expected_run_id=claimed.current_run_id)
    kb.claim_review_task(conn, tid)
    ok, who = kb.request_changes(conn, tid, reason="proof missing")
    assert ok, who
    return kb.get_task(conn, tid)


def _events(conn, tid, kind):
    return conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?", (tid, kind)).fetchall()


def test_third_change_request_parks_in_triage(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="software-engineer")
        assert _round(conn, tid, first=True).status == "ready"
        assert _round(conn, tid).status == "ready"
        t = _round(conn, tid)
        assert t.status == "triage"
        assert t.assignee == "software-engineer"
        assert len(_events(conn, tid, "review_loop_detected")) == 1
        # The dispatcher never claims triage: no fourth implementer spawn.
        assert kb.claim_task(conn, tid) is None
        # Re-spec returns it to the pool with a fresh brief.
        assert kb.specify_triage_task(conn, tid, body="PROOF: live check X")
        assert kb.get_task(conn, tid).status in ("todo", "ready")


def test_limit_zero_disables(kanban_home):
    (kanban_home / "config.yaml").write_text("kanban:\n  review_round_limit: 0\n")
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="software-engineer")
        _round(conn, tid, first=True)
        _round(conn, tid)
        assert _round(conn, tid).status == "ready"
        assert not _events(conn, tid, "review_loop_detected")
