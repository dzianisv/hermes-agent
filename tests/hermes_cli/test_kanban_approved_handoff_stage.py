"""Issue #26 / #6: an approved review must hand off to merge, not loop as rework,
and every handoff must record the delivery stage (``current_step_key``).

Real card t_c76abc52: the reviewer approved PR #5446 but the only verdicts were
``kanban_complete`` (premature done, merge still pending) or
``kanban_request_changes``, so the approval was recorded as ``changes_requested``
and the engineer was respawned as if rework was needed.
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


def _to_review(conn) -> str:
    tid = kb.create_task(conn, title="ship it", assignee="software-engineer")
    claimed = kb.claim_task(conn, tid)
    kb.add_comment(conn, tid, author="software-engineer", body=PR_COMMENT)
    assert kb.request_review(conn, tid, summary="PR ready", reviewer="reviewer",
                             expected_run_id=claimed.current_run_id)
    return tid


def _events(conn, tid, kind):
    return conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (tid, kind)).fetchall()


def test_approve_for_merge_hands_to_implementer_at_merge_stage(kanban_home):
    with kb.connect() as conn:
        tid = _to_review(conn)
        assert kb.get_task(conn, tid).current_step_key == "review"
        rv = kb.claim_review_task(conn, tid)
        ok, who = kb.approve_for_merge(conn, tid, summary="LGTM at abc123",
                                       expected_run_id=rv.current_run_id)
        assert ok, who
        assert who == "software-engineer"
        t = kb.get_task(conn, tid)
        assert t.status == "ready"
        assert t.assignee == "software-engineer"
        assert t.current_step_key == "merge"
        latest = kb.latest_run(conn, tid)
        assert latest.outcome == "approved"
        assert latest.profile == "reviewer"  # reviewer provenance kept on the run
        assert not _events(conn, tid, "changes_requested"), "approval is not a changes request"
        assert len(_events(conn, tid, "review_approved")) == 1
        # Merge needs the same PR: the open-PR respawn guard must not hold it.
        assert kb.check_respawn_guard(conn, tid) is None
        assert kb.goal_run_status(conn, tid, rv.current_run_id) == "approved"
        # Merger finishes -> done stage.
        m = kb.claim_task(conn, tid)
        assert kb.complete_task(conn, tid, summary="merged as deadbeef",
                                expected_run_id=m.current_run_id)
        assert kb.get_task(conn, tid).current_step_key == "done"


def test_changes_requested_records_development_stage(kanban_home):
    with kb.connect() as conn:
        tid = _to_review(conn)
        kb.claim_review_task(conn, tid)
        ok, _ = kb.request_changes(conn, tid, reason="add tests")
        assert ok
        assert kb.get_task(conn, tid).current_step_key == "development"


def test_approve_refused_outside_review_run(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="software-engineer")
        kb.claim_task(conn, tid)
        ok, reason = kb.approve_for_merge(conn, tid, summary="LGTM")
        assert not ok and "review" in reason
