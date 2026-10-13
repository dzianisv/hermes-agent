"""Profile-level quota-wall cooldown: a wall on one card parks its siblings."""
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _run(conn, tid, outcome, ended_at):
    kb.claim_task(conn, tid)
    run_id = kb.get_task(conn, tid).current_run_id
    conn.execute("UPDATE task_runs SET outcome=?, status=?, ended_at=? WHERE id=?",
                 (outcome, outcome, ended_at, run_id))
    conn.execute("UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, "
                 "claim_expires=NULL, worker_pid=NULL WHERE id=?", (tid,))
    conn.commit()


@pytest.fixture
def clock(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    state = {"now": 5_000_000}
    monkeypatch.setattr(kb.time, "time", lambda: state["now"])
    return state


def test_quota_wall_on_task_a_parks_sibling_same_assignee(kanban_home, clock):
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="a", assignee="swe")
        b = kb.create_task(conn, title="b", assignee="swe")
        _run(conn, a, "rate_limited", clock["now"])
        clock["now"] += 100
        assert kbd.check_respawn_guard(conn, b) == "profile_rate_limit_cooldown"
        clock["now"] += 300
        assert kbd.check_respawn_guard(conn, b) is None


def test_quota_wall_does_not_park_other_assignee(kanban_home, clock):
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="a", assignee="swe")
        c = kb.create_task(conn, title="c", assignee="reviewer")
        _run(conn, a, "rate_limited", clock["now"])
        clock["now"] += 100
        assert kbd.check_respawn_guard(conn, c) is None
