"""Runnable-assignee enforcement (card t_e36115cf).

Hermes must not represent non-runnable placeholder ownership as active
engineering work. A card assigned to a reservation label with no installed
profile (``copilot-external``, ``builder``, ``a``) looks owned but can never
execute.

These tests pin the behaviour contract of the single authoritative validator
and of the surfaces that must use it:

* ``kanban_db.ensure_runnable_assignee`` — canonicalize + validate against the
  profiles directory ON DISK (derived, never a hand-maintained list).
* CLI ``create`` / ``assign`` / ``reassign``.
* Dispatcher admission for existing legacy cards (fail closed + durable event).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

# These suites ARE the guard's tests: they install a real profile on a temp
# HERMES root and must see the real profile_exists predicate, not the autouse
# stub that makes every assignee resolve.
pytestmark = pytest.mark.real_profile_gate


from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def installed_profile(kanban_home):
    """Create a real profile directory on disk and return its name."""
    name = "software-engineer"
    (kanban_home / "profiles" / name).mkdir(parents=True)
    return name


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


# ---------------------------------------------------------------------------
# The validator itself
# ---------------------------------------------------------------------------


def test_validator_accepts_profile_installed_on_disk(installed_profile):
    assert kb.ensure_runnable_assignee(installed_profile) == installed_profile


def test_validator_normalizes_case_of_installed_profile(installed_profile):
    assert kb.ensure_runnable_assignee("Software-Engineer") == installed_profile


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_validator_allows_unassignment(blank, installed_profile):
    assert kb.ensure_runnable_assignee(blank) is None


def test_validator_rejects_name_absent_from_profiles_dir(installed_profile):
    with pytest.raises(kb.UnknownAssigneeError) as exc:
        kb.ensure_runnable_assignee("copilot-external")
    msg = str(exc.value)
    # Actionable: names the offender AND what a runnable owner looks like.
    assert "copilot-external" in msg
    assert installed_profile in msg


def test_validator_derives_candidates_from_disk_not_a_literal_list(kanban_home):
    """A profile created after import time must immediately be accepted.

    This is the P-GUARD property: the validator reads the source of truth
    (the profiles directory) on every call, so it cannot go stale.
    """
    with pytest.raises(kb.UnknownAssigneeError):
        kb.ensure_runnable_assignee("late-arrival")
    (kanban_home / "profiles" / "late-arrival").mkdir(parents=True)
    assert kb.ensure_runnable_assignee("late-arrival") == "late-arrival"


# ---------------------------------------------------------------------------
# DB layer
# ---------------------------------------------------------------------------


def test_create_task_rejects_unknown_assignee(conn, installed_profile):
    with pytest.raises(kb.UnknownAssigneeError):
        kb.create_task(conn, title="t", assignee="copilot-external")
    assert kb.list_tasks(conn) == []


def test_assign_task_rejects_unknown_assignee(conn, installed_profile):
    tid = kb.create_task(conn, title="t", assignee=installed_profile)
    with pytest.raises(kb.UnknownAssigneeError):
        kb.assign_task(conn, tid, "builder")
    task = kb.get_task(conn, tid)
    assert task is not None and task.assignee == installed_profile


# ---------------------------------------------------------------------------
# CLI surfaces
# ---------------------------------------------------------------------------


def _create_args(**over):
    base = dict(
        title="t", body=None, assignee=None, created_by="tester",
        workspace="scratch", branch=None, tenant=None, priority=0,
        parent=[], triage=False, idempotency_key=None, max_runtime=None,
        skills=[], max_retries=None, model_override=None,
        provider_override=None, goal_mode=False, goal_max_turns=None,
        initial_status="running", project=None, json=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_cli_create_rejects_unknown_assignee(conn, installed_profile, capsys):
    rc = kb_cli._cmd_create(_create_args(assignee="copilot-external"))
    assert rc == 2
    assert "copilot-external" in capsys.readouterr().err
    assert kb.list_tasks(conn) == []


def test_cli_create_accepts_installed_profile(conn, installed_profile):
    rc = kb_cli._cmd_create(_create_args(assignee=installed_profile))
    assert rc == 0
    assert [t.assignee for t in kb.list_tasks(conn)] == [installed_profile]


@pytest.mark.parametrize("sentinel", ["none", "None", "-", "null"])
def test_cli_create_honours_unassign_sentinels(sentinel, conn, installed_profile):
    """``create --assignee none`` must leave the card unowned, not be rejected.

    ``assign``/``reassign`` already translate these sentinels to ``None``
    before validating, and the validator's own error message tells the user
    to "pass 'none' to leave the task unassigned" — so ``create`` rejecting
    them contradicts the advice it prints.
    """
    rc = kb_cli._cmd_create(_create_args(assignee=sentinel))
    assert rc == 0
    assert [t.assignee for t in kb.list_tasks(conn)] == [None]


def test_cli_assign_rejects_unknown_assignee(conn, installed_profile, capsys):
    tid = kb.create_task(conn, title="t", assignee=installed_profile)
    rc = kb_cli._cmd_assign(
        argparse.Namespace(task_id=tid, profile="copilot-external")
    )
    assert rc == 2
    assert "copilot-external" in capsys.readouterr().err
    # Existing ownership is preserved, never silently rewritten.
    task = kb.get_task(conn, tid)
    assert task is not None and task.assignee == installed_profile


def test_cli_assign_still_allows_unassignment(conn, installed_profile):
    tid = kb.create_task(conn, title="t", assignee=installed_profile)
    rc = kb_cli._cmd_assign(argparse.Namespace(task_id=tid, profile="none"))
    assert rc == 0
    task = kb.get_task(conn, tid)
    assert task is not None and task.assignee is None


def test_cli_reassign_rejects_unknown_assignee(conn, installed_profile, capsys):
    tid = kb.create_task(conn, title="t", assignee=installed_profile)
    rc = kb_cli._cmd_reassign(
        argparse.Namespace(
            task_id=tid, profile="builder", reclaim=False, reason=None,
        )
    )
    assert rc == 2
    assert "builder" in capsys.readouterr().err
    task = kb.get_task(conn, tid)
    assert task is not None and task.assignee == installed_profile


# ---------------------------------------------------------------------------
# Dispatcher admission for legacy cards
# ---------------------------------------------------------------------------


def _legacy_ready(conn, installed_profile, assignee):
    """A row that predates validation: written straight to SQL."""
    tid = kb.create_task(conn, title="legacy", assignee=installed_profile)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET assignee = ?, status = 'ready' WHERE id = ?",
            (assignee, tid),
        )
    return tid


def test_dispatcher_fails_closed_on_legacy_unknown_assignee(
    conn, installed_profile
):
    tid = _legacy_ready(conn, installed_profile, "copilot-external")
    spawns = []
    res = kb.dispatch_once(
        conn, spawn_fn=lambda *a, **k: (spawns.append(a), 1234)[1],
    )
    assert spawns == []
    assert tid in res.skipped_nonspawnable
    task = kb.get_task(conn, tid)
    assert task is not None
    # Fields preserved: not reassigned, not archived, not claimed.
    assert task.assignee == "copilot-external"
    assert task.status == "ready"
    assert task.claim_lock is None
    # Durable diagnostic explaining WHY.
    events = kb.list_events(conn, tid)
    diag = [e for e in events if e.kind == "assignee_not_runnable"]
    assert len(diag) == 1
    assert diag[0].payload["assignee"] == "copilot-external"


def test_dispatcher_diagnostic_is_not_re_emitted_every_tick(
    conn, installed_profile
):
    tid = _legacy_ready(conn, installed_profile, "copilot-external")
    for _ in range(3):
        kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)
    kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert kinds.count("assignee_not_runnable") == 1


def test_dispatcher_still_spawns_a_real_installed_profile(
    conn, installed_profile
):
    tid = _legacy_ready(conn, installed_profile, installed_profile)
    spawns = []
    kb.dispatch_once(
        conn, spawn_fn=lambda *a, **k: (spawns.append(a), 4321)[1],
    )
    assert len(spawns) == 1
    task = kb.get_task(conn, tid)
    assert task is not None and task.status == "running"
