"""Hermetic tests for the audited workspace-metadata correction path.

``hermes kanban set-workspace`` / :func:`hermes_cli.kanban_db.set_task_workspace`
is the ONLY supported way to fix a card whose recorded workspace is wrong (the
dispatcher's own ``set_workspace_path`` is an internal claim-time write with no
kind, no audit event and no claim guard). The contract under test:

* it rewrites ``workspace_kind`` + ``workspace_path``;
* it clears ``last_failure_error`` (new workspace is new evidence);
* it clears ``branch_name`` when moving away from worktree kind;
* it moves NO files and starts NO worker;
* identity, status, assignee, claim, links/dependencies and prior events are
  preserved;
* every invalid request is refused, and a refusal writes nothing;
* the change is auditable (``workspace_updated`` event with old/new + actor
  and any cleared fields) and reads back through the normal API.

Every test runs against an isolated board in a temp ``HERMES_HOME``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.hermes_cli._kanban_modules import KanbanModules

kb = KanbanModules()


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _snapshot(conn, task_id):
    t = kb.get_task(conn, task_id)
    return {
        "id": t.id,
        "title": t.title,
        "status": t.status,
        "assignee": t.assignee,
        "priority": t.priority,
        "created_at": t.created_at,
        "claim_lock": t.claim_lock,
        "branch_name": t.branch_name,
        "project_id": t.project_id,
    }


def test_records_an_existing_worktree_and_is_auditable(kanban_home, tmp_path):
    """The real correction shape: shared repo root -> per-card worktree."""
    wt = tmp_path / "worktrees" / "t_card-a"
    wt.mkdir(parents=True)
    before_entries = sorted(p.name for p in wt.iterdir())

    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="card a", assignee="agent-a",
            workspace_kind="dir", workspace_path=str(tmp_path / "repo-root"),
            initial_status="blocked",
        )
        child = kb.create_task(conn, title="child", initial_status="blocked")
        kb.link_tasks(conn, t, child)
        before = _snapshot(conn, t)
        events_before = len(kb.list_events(conn, t))

        applied = kb.set_task_workspace(
            conn, t, workspace_kind="worktree", workspace_path=str(wt),
            actor="operator-x",
        )

        # Readback through the normal API, not the raw row.
        task = kb.get_task(conn, t)
        assert task.workspace_kind == "worktree"
        assert task.workspace_path == str(wt)

        # Identity / status / claim / links preserved.
        assert _snapshot(conn, t) == before
        assert kb.child_ids(conn, t) == [child]

        # Audit event appended (history preserved, nothing rewritten).
        events = kb.list_events(conn, t)
        assert len(events) == events_before + 1
        ev = events[-1]
        assert ev.kind == "workspace_updated"
        assert ev.payload["actor"] == "operator-x"
        assert ev.payload["old"]["workspace_kind"] == "dir"
        assert ev.payload["new"] == {
            "workspace_kind": "worktree", "workspace_path": str(wt)}
        assert applied["new"] == ev.payload["new"]

    # No files moved, created or removed.
    assert sorted(p.name for p in wt.iterdir()) == before_entries


def test_refusals_write_nothing(kanban_home, tmp_path):
    existing = tmp_path / "ws-ok"
    existing.mkdir()
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="card b",
            workspace_kind="dir", workspace_path=str(existing),
            initial_status="blocked",
        )
        before = _snapshot(conn, t)
        events_before = len(kb.list_events(conn, t))

        bad = [
            # unknown kind
            dict(workspace_kind="container", workspace_path=str(existing)),
            dict(workspace_kind="", workspace_path=str(existing)),
            # missing path for a path-bearing kind
            dict(workspace_kind="dir", workspace_path=None),
            dict(workspace_kind="worktree", workspace_path="   "),
            # relative path
            dict(workspace_kind="dir", workspace_path="relative/ws"),
            # path that does not exist (this command records, never creates)
            dict(workspace_kind="worktree",
                 workspace_path=str(tmp_path / "nope")),
            # scratch + explicit path is the #28818 foot-gun
            dict(workspace_kind="scratch", workspace_path=str(existing)),
            # unknown task
            dict(workspace_kind="dir", workspace_path=str(existing)),
        ]
        for i, kwargs in enumerate(bad):
            target = t if i < len(bad) - 1 else "t_does_not_exist"
            with pytest.raises(kb.WorkspaceUpdateRefused):
                kb.set_task_workspace(conn, target, **kwargs)

        # Untouched: kind, path, identity, and the event log.
        task = kb.get_task(conn, t)
        assert (task.workspace_kind, task.workspace_path) == ("dir", str(existing))
        assert _snapshot(conn, t) == before
        assert len(kb.list_events(conn, t)) == events_before
        assert not (tmp_path / "nope").exists(), "refusal must not create the dir"


def test_running_task_is_refused(kanban_home, tmp_path):
    ws = tmp_path / "ws-run"
    ws.mkdir()
    with kb.connect() as conn:
        t = kb.create_task(conn, title="card c", initial_status="running")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='running' WHERE id = ?", (t,))
        assert kb.get_task(conn, t).status == "running"
        with pytest.raises(kb.WorkspaceUpdateRefused) as exc:
            kb.set_task_workspace(
                conn, t, workspace_kind="dir", workspace_path=str(ws))
        assert "running" in str(exc.value)
        assert kb.get_task(conn, t).workspace_kind == "scratch"


def test_active_claim_is_refused(kanban_home, tmp_path):
    import time as _time

    ws = tmp_path / "ws-claim"
    ws.mkdir()
    with kb.connect() as conn:
        t = kb.create_task(conn, title="card d", initial_status="blocked")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET claim_lock = ?, claim_expires = ? WHERE id = ?",
                ("worker-1", int(_time.time()) + 600, t),
            )
        with pytest.raises(kb.WorkspaceUpdateRefused) as exc:
            kb.set_task_workspace(
                conn, t, workspace_kind="dir", workspace_path=str(ws))
        assert "claim" in str(exc.value)

        # An EXPIRED claim is not an active one — the correction goes through.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET claim_expires = ? WHERE id = ?",
                (int(_time.time()) - 1, t),
            )
        kb.set_task_workspace(
            conn, t, workspace_kind="dir", workspace_path=str(ws))
        assert kb.get_task(conn, t).workspace_path == str(ws)
        # The claim itself is untouched by the workspace write.
        assert kb.get_task(conn, t).claim_lock == "worker-1"


def test_worktree_can_move_to_another_kind_clears_branch_name(kanban_home, tmp_path):
    """A worktree card carrying branch_name CAN move to scratch/dir and branch_name is cleared."""
    ws = tmp_path / "ws-branch"
    ws.mkdir()
    with kb.connect() as conn:
        t = kb.create_task(conn, title="card e", initial_status="blocked")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_kind='worktree', branch_name=? "
                "WHERE id = ?", ("wt/card-e", t),
            )

        # Moving from worktree to dir should succeed now and clear branch_name
        applied = kb.set_task_workspace(
            conn, t, workspace_kind="dir", workspace_path=str(ws))

        task = kb.get_task(conn, t)
        assert task.workspace_kind == "dir"
        assert task.workspace_path == str(ws)
        assert task.branch_name is None  # Cleared

        # The cleared branch_name should be in the result and event
        assert "cleared" in applied
        assert applied["cleared"]["branch_name"] == "wt/card-e"

        # Verify the event recorded the cleared field
        events = kb.list_events(conn, t)
        ws_event = [e for e in events if e.kind == "workspace_updated"][-1]
        assert ws_event.payload.get("cleared", {}).get("branch_name") == "wt/card-e"


def test_cli_verb_applies_and_refuses(kanban_home, tmp_path, capsys):
    """Drive the real argparse surface (`hermes kanban set-workspace`)."""
    from hermes_cli import kanban as kcli

    ws = tmp_path / "cli-ws"
    ws.mkdir()
    with kb.connect() as conn:
        t = kb.create_task(conn, title="card f", initial_status="blocked")

    args = _parse_kanban(
        ["set-workspace", t, "--kind", "worktree", "--path", str(ws), "--json"])
    assert kcli.kanban_command(args) == 0
    out = capsys.readouterr().out
    assert str(ws) in out

    with kb.connect() as conn:
        assert kb.get_task(conn, t).workspace_path == str(ws)

    # Refusal exits non-zero and changes nothing.
    args = _parse_kanban(
        ["set-workspace", t, "--kind", "dir", "--path", "relative"])
    assert kcli.kanban_command(args) == 1
    with kb.connect() as conn:
        assert kb.get_task(conn, t).workspace_kind == "worktree"


def test_delegated_child_cannot_set_workspace():
    """Workers must not rewrite their own (or another card's) workspace."""
    from hermes_cli import kanban as kcli

    assert "set-workspace" in kcli._DELEGATED_CHILD_DENIED_ACTIONS


def test_set_workspace_clears_stale_failure_error(kanban_home, tmp_path):
    """set-workspace clears last_failure_error and resets the respawn guard."""
    from hermes_cli import kanban_db_dispatch as kbd

    ws = tmp_path / "ws-ok"
    ws.mkdir()

    with kb.connect() as conn:
        t = kb.create_task(conn, title="card with stale error", initial_status="blocked")
        # Stamp a workspace EACCES error (the measured shape)
        error_text = "workspace: [Errno 13] Permission denied: '/root/x'"
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
                (error_text, t),
            )

        # Verify the guard parks it (infrastructure_cooldown, not blocker_auth)
        guard_reason = kbd.check_respawn_guard(conn, t)
        assert guard_reason != "blocker_auth"

        # Now fix the workspace
        applied = kb.set_task_workspace(
            conn, t, workspace_kind="dir", workspace_path=str(ws))

        # last_failure_error should be NULL now
        row = conn.execute(
            "SELECT last_failure_error FROM tasks WHERE id = ?", (t,)
        ).fetchone()
        assert row[0] is None

        # The cleared error should be recorded
        assert "cleared" in applied
        assert applied["cleared"]["last_failure_error"] == error_text

        # The event should record the cleared field
        events = kb.list_events(conn, t)
        ws_event = [e for e in events if e.kind == "workspace_updated"][-1]
        assert ws_event.payload.get("cleared", {}).get("last_failure_error") == error_text

        # check_respawn_guard should no longer park it
        guard_reason = kbd.check_respawn_guard(conn, t)
        assert guard_reason is None


def _parse_kanban(argv):
    import argparse

    from hermes_cli import kanban as kcli

    top = argparse.ArgumentParser()
    subs = top.add_subparsers(dest="command")
    kcli.build_parser(subs)
    return top.parse_args(["kanban"] + list(argv))
