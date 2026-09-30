"""EM harness gates: marker-required worker blocks, and approval returns to the implementer.

Worker blocks (``expected_run_id`` set) from ``kanban.block_marker_required_profiles``
are not blockers unless the reason names BLOCKER:HUMAN, BLOCKER:EXTERNAL, or
BLOCKER:DEP. A reviewer completion that says APPROVED at a head sha hands the
card back for the sanctioned merge instead of closing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_harness import BLOCK_REJECTED_NOTE, approval_return_comment


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _claim(conn, title="t", assignee="software-engineer"):
    tid = kb.create_task(conn, title=title, assignee=assignee)
    claimed = kb.claim_task(conn, tid, claimer=assignee)
    assert claimed is not None
    return tid, claimed.current_run_id


def _events(conn, tid, kind):
    return [event for event in kb.list_events(conn, tid) if event.kind == kind]


def _review_claimed(conn, *, assignee="builder", reviewer="reviewer"):
    tid, impl_run = _claim(conn, assignee=assignee)
    assert kb.request_review(
        conn, tid, summary="ready for review", reviewer=reviewer, expected_run_id=impl_run,
    )
    review = kb.claim_review_task(conn, tid, claimer=reviewer)
    assert review is not None
    return tid, review.current_run_id


# ---------------------------------------------------------------------------
# Gate 1 — block marker required
# ---------------------------------------------------------------------------


def test_markerless_worker_block_returns_ready(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid, run_id = _claim(conn)
        before = kb.get_task(conn, tid)
        assert kb.block_task(
            conn, tid, reason="waiting on review and CI", expected_run_id=run_id,
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        assert task.current_run_id is None
        assert task.claim_lock is None
        assert task.worker_pid is None
        assert (task.block_kind, task.block_recurrences) == (
            before.block_kind, before.block_recurrences,
        )
        rejected = _events(conn, tid, "block_rejected")
        assert len(rejected) == 1
        assert rejected[0].payload == {
            "reason": "waiting on review and CI", "rejections": 1,
        }
        comments = kb.list_comments(conn, tid)
        assert comments[-1].author == "em-harness"
        assert comments[-1].body == BLOCK_REJECTED_NOTE
        run = kb.latest_run(conn, tid)
        assert run is not None
        assert run.outcome == "block_rejected"


@pytest.mark.parametrize("marker", [
    "BLOCKER:HUMAN ship-or-not",
    "BLOCKER:EXTERNAL github",
    "  BLOCKER:DEP t_abc1234",
])
def test_each_blocker_marker_still_blocks(kanban_home: Path, marker: str) -> None:
    with kbc.connect_closing() as conn:
        tid, run_id = _claim(conn)
        assert kb.block_task(
            conn, tid, reason=f"cannot proceed\n{marker}", expected_run_id=run_id,
        )
        assert kb.get_task(conn, tid).status == "blocked"
        assert _events(conn, tid, "block_rejected") == []


def test_human_block_without_run_id_is_unaffected(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid, _run_id = _claim(conn)
        assert kb.block_task(conn, tid, reason="waiting on CI")
        assert kb.get_task(conn, tid).status == "blocked"
        assert _events(conn, tid, "block_rejected") == []


def test_fourth_markerless_block_lands_blocked(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid, run_id = _claim(conn)
        for n in range(1, 4):
            assert kb.block_task(
                conn, tid, reason="still waiting on CI", expected_run_id=run_id,
            )
            assert kb.get_task(conn, tid).status == "ready"
            assert _events(conn, tid, "block_rejected")[-1].payload["rejections"] == n
            claimed = kb.claim_task(conn, tid, claimer="software-engineer")
            assert claimed is not None
            run_id = claimed.current_run_id
        assert kb.block_task(
            conn, tid, reason="still waiting on CI", expected_run_id=run_id,
        )
        assert kb.get_task(conn, tid).status == "blocked"
        assert len(_events(conn, tid, "block_rejected")) == 3


def test_non_matching_assignee_block_is_unaffected(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid, run_id = _claim(conn, assignee="worker")
        assert kb.block_task(
            conn, tid, reason="waiting on CI", expected_run_id=run_id,
        )
        assert kb.get_task(conn, tid).status == "blocked"
        assert _events(conn, tid, "block_rejected") == []


def test_dependency_with_open_parent_is_a_blocker(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="software-engineer")
        tid, run_id = _claim(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=tid, expected_child_run_id=run_id)
        assert kb.block_task(
            conn, tid, reason="waiting on parent", kind="dependency", expected_run_id=run_id,
        )
        assert kb.get_task(conn, tid).status == "todo"
        assert _events(conn, tid, "block_rejected") == []


def test_handle_block_surfaces_rejection(kanban_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools import kanban_tools as kt

    with kbc.connect_closing() as conn:
        tid, run_id = _claim(conn)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    out = json.loads(kt._handle_block({"reason": "waiting on the review tag"}))
    assert out["ok"] is True
    assert out["status"] == "ready"
    assert out["block_rejected"] is True
    assert "BLOCKER:HUMAN" in out["note"]


# ---------------------------------------------------------------------------
# Gate 2 — approval returns to the implementer
# ---------------------------------------------------------------------------


def test_reviewer_approved_complete_returns_to_implementer(kanban_home: Path) -> None:
    sha = "abc1234"
    with kbc.connect_closing() as conn:
        tid, run_id = _review_claimed(conn, assignee="builder")
        assert kb.complete_task(
            conn, tid, summary=f"APPROVED at {sha}", expected_run_id=run_id,
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        assert task.assignee == "builder"
        assert task.completed_at is None
        approved = _events(conn, tid, "review_approved")
        assert len(approved) == 1
        assert approved[0].payload["step"] == "merge"
        assert approved[0].payload["head_sha"] == sha
        assert approved[0].payload["implementer"] == "builder"
        run = kb.latest_run(conn, tid)
        assert run is not None
        assert run.outcome == "review_approved"
        assert kb.list_comments(conn, tid)[-1].body == approval_return_comment(sha)


def test_review_complete_without_approved_is_done(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid, run_id = _review_claimed(conn)
        assert kb.complete_task(
            conn, tid, summary="LGTM — merged", expected_run_id=run_id,
        )
        assert kb.get_task(conn, tid).status == "done"
        assert _events(conn, tid, "review_approved") == []


def test_review_complete_with_merged_metadata_is_done(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid, run_id = _review_claimed(conn)
        assert kb.complete_task(
            conn, tid, summary="APPROVED at abc1234", metadata={"merged": True},
            expected_run_id=run_id,
        )
        assert kb.get_task(conn, tid).status == "done"
        assert _events(conn, tid, "review_approved") == []


def test_request_changes_approved_records_merge_step(kanban_home: Path) -> None:
    sha = "deadbee"
    with kbc.connect_closing() as conn:
        tid, run_id = _review_claimed(conn, assignee="builder")
        ok, implementer = kb.request_changes(
            conn, tid, reason=f"APPROVED at {sha}", expected_run_id=run_id,
        )
        assert (ok, implementer) == (True, "builder")
        changes = _events(conn, tid, "changes_requested")[-1].payload
        assert changes["step"] == "merge"
        assert changes["head_sha"] == sha
        assert kb.get_task(conn, tid).status == "ready"
