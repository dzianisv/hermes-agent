"""`hermes kanban complete` must enforce the PROOF-CMD contract (real temp sqlite DB).

Regression: a card with ``PROOF-CMD: false`` was completed via the CLI and went to done.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID"):
        monkeypatch.delenv(var, raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _card(body):
    with kbc.connect_closing() as conn:
        return kb.create_task(conn, title="widget", body=body, assignee="alice")


def _state(tid):
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        events = [e for e in kb.list_events(conn, tid) if e.kind == "proof_run"]
        runs = conn.execute("SELECT metadata FROM task_runs WHERE task_id = ?", (tid,)).fetchall()
    return task.status, events, [r[0] for r in runs]


def test_cli_complete_refuses_failing_proof_cmd(kanban_home):
    tid = _card("Fix it.\n\nPROOF-CMD: false\n")
    out = kc.run_slash(f"complete {tid} --summary probe")
    status, events, _ = _state(tid)
    assert status != "done", out
    assert "PROOF-CMD" in out and "exited 1" in out
    assert events and events[-1].payload["passed"] is False
    assert events[-1].payload["proof_run"][0]["exit_code"] == 1


def test_cli_complete_records_passing_proof_run(kanban_home):
    tid = _card("Fix it.\n\nPROOF-CMD: true\n")
    kc.run_slash(f"complete {tid} --summary probe")
    status, events, runs = _state(tid)
    assert status == "done"
    assert events and events[-1].payload["passed"] is True
    assert any(m and '"proof_run"' in m and '"exit_code": 0' in m for m in runs)


def test_proof_cmd_timeout_refuses(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_PROOF_TIMEOUT", "1")
    tid = _card("PROOF-CMD: sleep 5\n")
    with kbc.connect_closing() as conn:
        with pytest.raises(kb.ProofFailedError) as ei:
            kb.complete_task(conn, tid, summary="probe")
    assert ei.value.failed["timed_out"] is True
    assert _state(tid)[0] != "done"
