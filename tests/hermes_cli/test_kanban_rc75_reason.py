"""rc 75 is shared by quota walls, timeouts, stale-provider aborts and network loss.
Live board: 50 of 52 rc=75 "rate_limited" runs were pi-acp stale-call aborts, yet #28's
profile cooldown parked every sibling card as if the profile's quota were exhausted.
The worker now writes its classified failure_reason; only quota reasons park siblings."""
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import quiet_single_query as qsq


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    kb.init_db()
    return home


def _rl_run(conn, tid, reason):
    kb.claim_task(conn, tid)
    run_id = kb.get_task(conn, tid).current_run_id
    md = json.dumps({"provider_reason": reason}) if reason else None
    conn.execute("UPDATE task_runs SET outcome='rate_limited', status='rate_limited', "
                 "ended_at=?, metadata=? WHERE id=?", (int(kb.time.time()), md, run_id))
    conn.execute("UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, "
                 "claim_expires=NULL, worker_pid=NULL WHERE id=?", (tid,))
    conn.commit()


@pytest.mark.parametrize("reason", ["timeout", "network_unavailable", "server_error"])
def test_non_quota_rc75_does_not_park_siblings(kanban_home, reason):
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="a", assignee="swe")
        b = kb.create_task(conn, title="b", assignee="swe")
        _rl_run(conn, a, reason)
        assert kbd.check_respawn_guard(conn, b) != "profile_rate_limit_cooldown"


@pytest.mark.parametrize("reason", ["rate_limit", None])
def test_quota_rc75_still_parks_siblings(kanban_home, reason):
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="a", assignee="swe")
        b = kb.create_task(conn, title="b", assignee="swe")
        _rl_run(conn, a, reason)
        assert kbd.check_respawn_guard(conn, b) == "profile_rate_limit_cooldown"


def test_worker_writes_reason_trailer_and_dispatcher_reads_it(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    with pytest.raises(SystemExit):
        qsq.exit_single_query(75, reason="timeout")
    err = capsys.readouterr().err
    monkeypatch.setattr(kbd._kb, "read_worker_log", lambda *a, **k: err)
    assert kbd._worker_log_exit_reason("t_x") == "timeout"
