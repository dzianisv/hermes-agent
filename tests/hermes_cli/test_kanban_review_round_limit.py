"""Issues-to-resolve #4: a reviewed card must not loop forever.

Live board: cards bounced 4-6 change-request rounds because the brief never
named the proof. The Nth change request (kanban.review_round_limit, default 3)
blocks the card needs_input with a comment instead of respawning the implementer.
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


def test_third_change_request_blocks_with_comment(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="software-engineer")
        assert _round(conn, tid, first=True).status == "ready"
        assert _round(conn, tid).status == "ready"
        t = _round(conn, tid)
        assert (t.status, t.block_kind, t.assignee) == ("blocked", "needs_input", "software-engineer")
        assert len(_events(conn, tid, "review_loop_detected")) == 1
        body = conn.execute("SELECT body FROM task_comments WHERE task_id=? AND author='kanban-harness'",
                            (tid,)).fetchone()[0]
        assert "review loop: 3" in body
        # No fourth implementer spawn until someone unblocks it.
        assert kb.claim_task(conn, tid) is None
        assert kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status in ("todo", "ready")


def test_limit_zero_disables(kanban_home):
    (kanban_home / "config.yaml").write_text("kanban:\n  review_round_limit: 0\n")
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="software-engineer")
        _round(conn, tid, first=True)
        _round(conn, tid)
        assert _round(conn, tid).status == "ready"
        assert not _events(conn, tid, "review_loop_detected")


def _add_closed_run(conn, tid, seconds):
    conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
        "VALUES (?, 'software-engineer', 'crashed', 'crashed', 1000, ?)", (tid, 1000 + seconds))
    conn.commit()


def test_claim_parks_card_past_24h_active_time(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="loops", assignee="software-engineer")
        _add_closed_run(conn, tid, 86400)
        assert kb.claim_task(conn, tid) is None
        assert kb.get_task(conn, tid).status == "triage"
        assert len(_events(conn, tid, "active_time_exceeded")) == 1


def test_claim_under_24h_still_respawns(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="normal retry", assignee="software-engineer")
        _add_closed_run(conn, tid, 86399)
        assert kb.claim_task(conn, tid) is not None


def test_active_limit_zero_disables(kanban_home):
    (kanban_home / "config.yaml").write_text("kanban:\n  active_seconds_limit: 0\n")
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="software-engineer")
        _add_closed_run(conn, tid, 10 * 86400)
        assert kb.claim_task(conn, tid) is not None


CFG = "kanban:\n  review_round_limit: 3\n"


def _rereview(conn, tid, reason):
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    assert kb.request_review(conn, tid, summary="PR ready", reviewer="reviewer",
                             expected_run_id=claimed.current_run_id)
    kb.claim_review_task(conn, tid)
    ok, who = kb.request_changes(conn, tid, reason=reason)
    assert ok, who
    return who, kb.get_task(conn, tid)


def test_under_cap_returns_to_implementer(kanban_home):
    (kanban_home / "config.yaml").write_text(CFG)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="software-engineer")
        t = _round(conn, tid, first=True)
        assert (t.status, t.assignee) == ("ready", "software-engineer")
        who, t = _rereview(conn, tid, "finding 2")
        assert (who, t.status, t.assignee) == ("software-engineer", "ready", "software-engineer")
        assert not _events(conn, tid, "review_loop_detected")


def test_rounds_reset_after_unblock(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="software-engineer")
        _round(conn, tid, first=True); _round(conn, tid)
        assert _round(conn, tid).status == "blocked"
        assert kb.unblock_task(conn, tid)
        assert _round(conn, tid).status == "ready"
        assert _round(conn, tid).status == "ready"
        assert _round(conn, tid).status == "blocked"


def _approve(conn, tid):
    claimed = kb.claim_task(conn, tid)
    assert kb.request_review(conn, tid, summary="PR ready", reviewer="reviewer",
                             expected_run_id=claimed.current_run_id)
    kb.claim_review_task(conn, tid)
    ok, who = kb.approve_for_merge(conn, tid, summary="approved slice 1")
    assert ok, who


def test_rounds_reset_after_approval_of_earlier_slice(kanban_home):
    """t_9aa956b9: slice 1 got one change request then approval+merge; slice 2's
    second change request must not be counted as round 3."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="software-engineer")
        _round(conn, tid, first=True)
        _approve(conn, tid)
        assert _round(conn, tid).status == "ready"
        assert _round(conn, tid).status == "ready"
        assert not _events(conn, tid, "review_loop_detected")
        assert _round(conn, tid).status == "blocked"
