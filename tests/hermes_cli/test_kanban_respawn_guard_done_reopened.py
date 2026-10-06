"""``done_reopened`` must lift the ``active_pr`` respawn guard (t_cf53508c).

An operator reopening a done card after its PR comment is a deliberate
re-queue (same class as ``unblocked``). Without it in the handoff-kind set
the reopened card sits ``respawn_guarded {"reason":"active_pr"}`` forever.

* :func:`test_done_reopened_after_pr_comment_releases_guard` — the defect.
* :func:`test_pr_comment_without_later_handoff_still_guards` — negative control.
* :func:`test_same_second_done_reopened_stays_guarded` — strictly-after rule:
  a same-second tie stays guarded (fail closed).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import pytest


PR_URL = "https://github.com/dzianisv/hermes-agent/pull/25"


@pytest.fixture()
def kb(monkeypatch):
    """Fresh HERMES_HOME + kanban DB, with an 'a' profile that can spawn."""
    test_home = tempfile.mkdtemp(prefix="kanban_respawn_guard_test_")
    for prof in ("a", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    # Strip HERMES_KANBAN_* pins so a dispatched worker running this file
    # never writes onto the real board.
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME",
                "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_LOGS_ROOT"):
        monkeypatch.delenv(var, raising=False)
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("hermes_cli")
            or mod.startswith("hermes_state")
            or mod == "hermes_constants"
        ):
            del sys.modules[mod]
    from tests.hermes_cli._kanban_modules import KanbanModules
    from hermes_cli import profiles
    # profile_exists resolves from HOME, not HERMES_HOME: treat assignees as real.
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    yield KanbanModules()


@pytest.fixture(autouse=True)
def _clear_pr_caches(kb):
    names = ("_PR_GUARD_CACHE", "_PR_STATE_CACHE")
    for name in names:
        getattr(kb, name, {}).clear()
    yield
    for name in names:
        getattr(kb, name, {}).clear()


def _stub_pr_status(kb, monkeypatch, status) -> None:
    """Stub every PR-metadata seam so no test shells out to ``gh``."""
    monkeypatch.setattr(
        kb, "_fetch_pr_status",
        lambda owner, repo, number: dict(status) if status else None,
        raising=False,
    )
    monkeypatch.setattr(
        kb, "_pr_url_is_open",
        lambda _url: bool(status) and status.get("state") == "OPEN",
        raising=False,
    )


_GREEN_NO_DECISION = {
    "state": "OPEN",
    "reviewDecision": "",
    "statusCheckRollup": [{"name": "test", "conclusion": "SUCCESS"}],
}


def test_done_reopened_after_pr_comment_releases_guard(kb, monkeypatch):
    _stub_pr_status(kb, monkeypatch, _GREEN_NO_DECISION)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="ship the thing", assignee="a")
        assert kb.claim_task(conn, task_id) is not None
        kb.add_comment(conn, task_id, "worker", f"Opened PR: {PR_URL}")
        conn.commit()
        assert kb.complete_task(
            conn, task_id, summary="opened PR", force=True,
        )
        # Strictly-after rule: the reopen must land in a later second.
        time.sleep(1.1)
        ok, err = kb.reopen_done_task(conn, task_id, actor="lead")
        assert ok, err

        assert conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'done_reopened'",
            (task_id,),
        ).fetchone() is not None

        assert kb.check_respawn_guard(conn, task_id) is None, (
            "an operator reopen-done after the PR comment is a deliberate "
            "re-queue and must lift active_pr"
        )
        assert kb.claim_task(conn, task_id) is not None


def test_pr_comment_without_later_handoff_still_guards(kb, monkeypatch):
    _stub_pr_status(kb, monkeypatch, {
        "state": "OPEN",
        "reviewDecision": "REVIEW_REQUIRED",
        "statusCheckRollup": [{"name": "test", "conclusion": "SUCCESS"}],
    })
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="ship the thing", assignee="a")
        kb.add_comment(conn, task_id, "worker", f"Opened PR: {PR_URL}")
        conn.commit()

        assert kb.check_respawn_guard(conn, task_id) == "active_pr"


def test_same_second_done_reopened_stays_guarded(kb, monkeypatch):
    _stub_pr_status(kb, monkeypatch, _GREEN_NO_DECISION)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="ship the thing", assignee="a")
        comment_id = kb.add_comment(conn, task_id, "worker", f"Opened PR: {PR_URL}")
        conn.commit()
        created_at = conn.execute(
            "SELECT created_at FROM task_comments WHERE id = ?", (comment_id,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'done_reopened', NULL, ?)",
            (task_id, created_at),
        )
        conn.commit()

        assert kb.check_respawn_guard(conn, task_id) == "active_pr", (
            "a same-second tie must stay guarded (fail closed)"
        )
