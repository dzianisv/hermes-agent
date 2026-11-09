"""AC4: the manual DB write used to clear a mis-targeted re-spec is now a CLI command.

Live: card t_bde1784e needed `UPDATE task_events ... 'no_progress_respec'` by hand
because the only way to reset the streak was to insert the event directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban_wait_on as kwo
from tests.hermes_cli._kanban_modules import KanbanModules

kb = KanbanModules()


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    c = kb.connect()
    yield c
    c.close()


def _no_progress_run(conn, tid):
    claimed = kb.claim_task(conn, tid)
    run_id = claimed.current_run_id
    kb.merge_run_metadata(conn, tid, run_id,
                          {"head_start": "c1", "worker_session_id": f"sess-{run_id}"})
    assert kb.schedule_task(conn, tid, reason="CI still red", expected_run_id=run_id)
    kb.unblock_task(conn, tid)


def _parse(argv):
    from hermes_cli import kanban as kanban_cli

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    kanban_cli.build_parser(sub)
    return kanban_cli, parser.parse_args(argv)


def test_respec_reset_cli_resets_the_streak(conn):
    tid = kb.create_task(conn, title="impl", assignee="software-engineer")
    _no_progress_run(conn, tid)
    _no_progress_run(conn, tid)
    assert kwo.no_progress_streak(conn, tid, "c1") == 2

    kanban_cli, args = _parse(["kanban", "respec-reset", tid, "--reason", "infra crash"])
    assert kanban_cli.kanban_command(args) == 0
    assert kwo.no_progress_streak(conn, tid, "c1") == 0
    payloads = [r["payload"] for r in conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'no_progress_respec'",
        (tid,))]
    assert any('"reset": true' in (p or "") for p in payloads)


def test_respec_reset_cli_unknown_task_fails(conn):
    kanban_cli, args = _parse(["kanban", "respec-reset", "t_doesnotexist"])
    assert kanban_cli.kanban_command(args) == 1
