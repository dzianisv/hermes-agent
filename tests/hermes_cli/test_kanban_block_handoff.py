"""Mechanism B: block-for-handoff must reach the named next assignee.

Measured defect (2026-11-09): a worker's shell subprocess carries
``HERMES_DELEGATED_CHILD_CONTEXT``, so ``hermes kanban assign`` from it is refused
("kanban: delegate_task child contexts cannot mutate Kanban tasks via the CLI") and
a card blocked with a handoff reason dead-ended as ``blocked`` on the architect.

The fence stays. ``kanban_block(next_assignee=…)`` records the target and
``kanban_db.apply_block_handoffs`` — run by the dispatcher at the top of every pass —
reassigns and resumes the card.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from tests.hermes_cli._kanban_modules import KanbanModules

kb = KanbanModules()


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    c = kb.connect()
    yield c
    c.close()


def _parser():
    from hermes_cli import kanban as kanban_cli

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    kanban_cli.build_parser(sub)
    return kanban_cli, parser


def _kinds(conn, tid):
    return [r[0] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ?", (tid,))]


def _comments(conn, tid):
    return [r["body"] for r in conn.execute(
        "SELECT body FROM task_comments WHERE task_id = ?", (tid,))]


def _blocked(conn, tid, *, next_assignee, kind="needs_input"):
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    ok = kb.block_task(conn, tid, reason="HANDOFF, not a defect: re-spec is final",
                       kind=kind, expected_run_id=claimed.current_run_id,
                       next_assignee=next_assignee)
    assert ok
    return claimed


def test_block_for_handoff_is_applied_by_the_dispatcher(conn, monkeypatch, tmp_path):
    tid = kb.create_task(conn, title="rework", assignee="swe-architect")
    _blocked(conn, tid, next_assignee="software-engineer")
    task = kb.get_task(conn, tid)
    assert task.status == "blocked"
    assert task.assignee == "swe-architect"
    assert task.block_next_assignee == "software-engineer"

    # The measured failing condition: the worker's own shell is still fenced.
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", str(tmp_path / ".hermes"))
    kanban_cli, parser = _parser()
    args = parser.parse_args(["kanban", "assign", tid, "software-engineer"])
    assert kanban_cli._is_delegated_child_cli_mutation(args) is True
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT")

    # ... and the dispatcher applies the handoff with no board mutation by the worker.
    assert kb.apply_block_handoffs(conn) == [(tid, "swe-architect", "software-engineer")]
    task = kb.get_task(conn, tid)
    assert task.status == "ready"
    assert task.assignee == "software-engineer"
    assert task.block_next_assignee is None
    assert "handoff_applied" in _kinds(conn, tid)
    assert kb.apply_block_handoffs(conn) == []


def test_dispatch_once_applies_handoff_in_one_pass(conn, monkeypatch):
    monkeypatch.setattr(kb, "_profile_exists_fn", lambda: None)
    tid = kb.create_task(conn, title="rework", assignee="swe-architect")
    _blocked(conn, tid, next_assignee="software-engineer")
    spawned = []
    res = kb.dispatch_once(
        conn, spawn_fn=lambda task, ws, b=None: spawned.append((task.id, task.assignee)) or None)
    assert (tid, "swe-architect", "software-engineer") in res.handoffs_applied
    assert (tid, "software-engineer") in spawned
    assert kb.get_task(conn, tid).assignee == "software-engineer"


def test_handoff_to_unknown_profile_stays_blocked_and_is_named(conn):
    tid = kb.create_task(conn, title="rework", assignee="swe-architect")
    _blocked(conn, tid, next_assignee="ghost-profile")
    assert kb.apply_block_handoffs(conn, profile_exists=lambda _p: False) == []
    task = kb.get_task(conn, tid)
    assert task.status == "blocked"
    assert task.block_next_assignee == "ghost-profile"
    assert "handoff_rejected" in _kinds(conn, tid)
    assert any("BLOCKER:HARNESS handoff" in body for body in _comments(conn, tid))
    # Deduped: a second pass does not spam the operator.
    kb.apply_block_handoffs(conn, profile_exists=lambda _p: False)
    assert _kinds(conn, tid).count("handoff_rejected") == 1


def test_worker_shell_reassign_still_fenced(conn, monkeypatch, tmp_path):
    """The fence is NOT widened: a worker/delegated shell may not reassign."""
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", str(tmp_path / ".hermes"))
    kanban_cli, parser = _parser()
    args = parser.parse_args(["kanban", "reassign", "t_deadbeef", "software-engineer"])
    assert kanban_cli._is_delegated_child_cli_mutation(args) is True
    assert kanban_cli.kanban_command(args) == 1
