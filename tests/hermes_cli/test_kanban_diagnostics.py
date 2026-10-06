"""Tests for hermes_cli.kanban_diagnostics — rule-engine that produces
structured distress signals (diagnostics) for kanban tasks.

These tests exercise each rule in isolation using minimal in-memory
task/event/run fixtures (no DB) plus a few integration-style cases
that round-trip through the real kanban_db to make sure the rule
engine works on sqlite3.Row objects as well as dataclasses.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_diagnostics as kd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task(**overrides):
    base = {
        "id": "t_demo00",
        "title": "demo task",
        "assignee": "demo",
        "status": "ready",
        "consecutive_failures": 0,
        "last_failure_error": None,
    }
    base.update(overrides)
    return base


def _event(kind, ts=None, **payload):
    return {
        "kind": kind,
        "created_at": int(ts if ts is not None else time.time()),
        "payload": payload or None,
    }


def _run(outcome="completed", run_id=1, error=None):
    return {
        "id": run_id,
        "outcome": outcome,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Each rule — positive + negative + clearing
# ---------------------------------------------------------------------------
















def test_running_with_open_parents_fires_only_while_running():
    """A running card whose parent is not terminal is flagged; the same graph
    on a ready/todo card (the gate is holding it) and a done parent are not."""
    graph = {"parents": [{"id": "t_parent", "title": "p", "status": "todo"}], "children": []}
    diags = kd.compute_task_diagnostics(_task(status="running", started_at=100), [], [], graph=graph)
    assert [d.kind for d in diags] == ["running_with_open_parents"]
    assert diags[0].data["open_parents"] == [{"id": "t_parent", "status": "todo"}]
    assert "hermes kanban unlink t_parent t_demo00" in diags[0].actions[0].payload["command"]
    assert kd.compute_task_diagnostics(_task(status="todo"), [], [], graph=graph) == []
    done_graph = {"parents": [{"id": "t_parent", "title": "p", "status": "done"}], "children": []}
    assert kd.compute_task_diagnostics(_task(status="running"), [], [], graph=done_graph) == []


def test_stuck_in_blocked_fires_past_threshold():
    now = int(time.time())
    task = _task(status="blocked")
    events = [
        _event("blocked", ts=now - 3600 * 48, reason="needs approval"),
    ]
    diags = kd.compute_task_diagnostics(
        task, events, [], now=now,
    )
    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "stuck_in_blocked"
    assert d.severity == "warning"
    assert d.data["age_hours"] >= 48








# ---------------------------------------------------------------------------
# Severity sorting
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Integration — runs through real kanban_db so sqlite.Row fields work
# ---------------------------------------------------------------------------


def test_engine_works_on_sqlite_row_objects(kanban_home):
    """Regression: the rule functions must handle sqlite3.Row (which
    supports mapping access but not attribute access and isn't a dict)
    as well as dataclass Task / plain dict. The API layer passes Row
    objects directly.
    """
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="p", assignee="w")
        real = kb.create_task(conn, title="r", assignee="x", created_by="w")
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent,
                summary="with phantom", created_cards=[real, "t_deadbeef1"],
            )
        # Pull Row objects the way the API helper does.
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (parent,),
        ).fetchone()
        events = list(conn.execute(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        runs = list(conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        diags = kd.compute_task_diagnostics(row, events, runs)
        assert len(diags) == 1
        assert diags[0].kind == "hallucinated_cards"
        assert "t_deadbeef1" in diags[0].data["phantom_ids"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Error-tolerance: a broken rule shouldn't 500 the whole compute call
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# stranded_in_ready
#
# Surfaces ready tasks that nobody has claimed within the threshold.
# Identity-agnostic by design: catches typo'd assignees, deleted profiles,
# down external worker pools, and misconfigured dispatchers in one rule.
# ---------------------------------------------------------------------------


def test_stranded_in_ready_fires_when_age_exceeds_threshold():
    """Default threshold = 30 min. A ready task promoted 45 min ago
    with no claim should fire as a warning."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    # 45 min = 2700s, threshold = 1800s.
    events = [_event("created", ts=now - 45 * 60)]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert len(stranded) == 1
    assert stranded[0].severity == "warning"
    assert stranded[0].data["age_seconds"] == 45 * 60
    assert stranded[0].data["assignee"] == "demo"




# ---------------------------------------------------------------------------
# triage_aux_unavailable rule — auto-decompose aware
# ---------------------------------------------------------------------------


def _triage_task():
    return _task(id="t_triage1", status="triage")








def test_severity_at_or_above_uses_threshold_semantics():
    assert kd.severity_at_or_above("warning", "warning") is True
    assert kd.severity_at_or_above("error", "warning") is True
    assert kd.severity_at_or_above("critical", "warning") is True
    assert kd.severity_at_or_above("critical", "error") is True
    assert kd.severity_at_or_above("warning", "error") is False
    assert kd.severity_at_or_above("error", "critical") is False
    assert kd.severity_at_or_above("mystery", "warning") is False
    assert kd.severity_at_or_above("warning", None) is True


# ---------------------------------------------------------------------------
# repeated_failures after a handoff cleared last_failure_error
#
# request_review / request_changes clear the task-scoped error text but keep
# consecutive_failures; the diagnostic must fall back (read-only) to the run
# history instead of claiming no error was recorded.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["review", "ready"])
def test_repeated_failures_falls_back_to_run_history_error(kanban_home, status):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="streak", assignee="worker")
        assert kb.claim_task(conn, tid) is not None
        with kb.write_txn(conn):
            kb._end_run(conn, tid, outcome="crashed", status="crashed", error="401 auth failed")
            conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, consecutive_failures = 5, "
                "last_failure_error = NULL WHERE id = ?", (status, tid),
            )
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
        runs = list(conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (tid,),
        ).fetchall())
        run_id = runs[-1]["id"]
        diags = [d for d in kd.compute_task_diagnostics(row, [], runs) if d.kind == "repeated_failures"]
        assert len(diags) == 1
        diag = diags[0]
        assert "(no error recorded)" not in diag.title
        assert f"(historical, run {run_id})" in diag.title
        assert "401 auth failed" in diag.title and "401 auth failed" in diag.detail
        assert "run history" in diag.detail
        assert diag.data["last_error"] is None
        assert diag.data["historical_run_id"] == run_id
        assert "historical_error" not in diag.data
        # Read-only: neither the error text nor the streak is written back.
        after = conn.execute(
            "SELECT last_failure_error, consecutive_failures FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert after["last_failure_error"] is None
        assert after["consecutive_failures"] == 5
    finally:
        conn.close()


def test_repeated_failures_prefers_task_error_over_history():
    task = _task(consecutive_failures=5, last_failure_error="current boom")
    runs = [_run("crashed", run_id=1, error="old boom")]
    diags = kd._rule_repeated_failures(task, [], runs, int(time.time()), {"failure_threshold": 3})
    assert len(diags) == 1
    assert "current boom" in diags[0].title and "historical" not in diags[0].title
    assert diags[0].data["historical_run_id"] is None


_FAKE_BEARER = "sk-test-FAKE0123456789abcdefABCDEF"
_FAKE_ENV_KEY = "sk-proj-FAKE0123456789abcdefABCDEFGHIJ"


def test_repeated_failures_historical_error_is_redacted():
    """Historical run text is raw worker output shown board-wide; credentials never leak."""
    task = _task(consecutive_failures=5, last_failure_error=None)
    error = (
        f'curl -H "Authorization: Bearer {_FAKE_BEARER}" https://api.example.com\n'
        f"OPENAI_API_KEY={_FAKE_ENV_KEY} 401 auth failed"
    )
    runs = [_run("crashed", run_id=7, error=error)]
    diags = kd._rule_repeated_failures(task, [], runs, int(time.time()), {"failure_threshold": 3})
    assert len(diags) == 1
    diag = diags[0]
    assert "(historical, run 7)" in diag.title
    assert "run history (run 7)" in diag.detail
    assert diag.data["historical_run_id"] == 7
    payload = json.dumps(diag.data)
    for secret in (_FAKE_BEARER, _FAKE_ENV_KEY):
        assert secret not in diag.title
        assert secret not in diag.detail
        assert secret not in payload
