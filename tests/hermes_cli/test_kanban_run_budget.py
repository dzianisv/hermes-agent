"""Issue #4: a per-task run budget enforced at the claim boundary.

Fixtures replay real run histories from ~/.hermes/kanban.db:
- t_763828ca: 18 review round-trips (17 ``changes_requested``) before parking.
- t_18fa57ca: 32 consecutive ``blocked`` runs ("Unchanged (26th check)...").
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect

# (outcome, duration_seconds) taken verbatim from t_763828ca's task_runs.
T_763828CA_RUNS = [
    ("reclaimed", 53), ("scheduled", 0), ("blocked", 958), ("scheduled", 1274),
    ("blocked", 3091), ("blocked", 949), ("blocked", 142), ("review_requested", 4205),
    ("changes_requested", 176), ("review_requested", 300), ("changes_requested", 127),
    ("review_requested", 1315), ("changes_requested", 170), ("review_requested", 2090),
    ("changes_requested", 169), ("review_requested", 5450), ("changes_requested", 177),
]
T_18FA57CA_RUNS = [("blocked", d) for d in (57, 278, 29, 32, 26, 25, 24, 25, 24, 27)]


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _replay(conn, tid, runs, *, start=None):
    t = start if start is not None else int(time.time()) - sum(d for _, d in runs) - 10 * len(runs) - 100
    for outcome, dur in runs:
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (?, 'dev', ?, ?, ?, ?)", (tid, outcome, outcome, t, t + dur),
        )
        if outcome == "changes_requested":
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'changes_requested', NULL, ?)",
                (tid, t + dur),
            )
        t += dur + 10
    conn.commit()


def _ready(conn, title="t"):
    tid = kb.create_task(conn, title=title, assignee="dev")
    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    conn.commit()
    return tid


def test_repeated_rejections_park_in_triage(kanban_home):
    with connect() as conn:
        tid = _ready(conn)
        _replay(conn, tid, T_763828CA_RUNS)
        assert kb.claim_task(conn, tid) is None
        task = kb.get_task(conn, tid)
        assert task.status == "triage"
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        assert "run_budget_exhausted" in kinds
        bodies = [c.body for c in kb.list_comments(conn, tid)]
        assert any("changes_requested" in b and "--reset-budget" in b for b in bodies)


def test_approved_handoffs_do_not_count(kanban_home):
    with connect() as conn:
        tid = _ready(conn)
        _replay(conn, tid, [("review_requested", 60), ("approved", 30)] * 5 + [("changes_requested", 30)] * 2)
        assert kb.claim_task(conn, tid) is not None


def test_repeat_hold_parks(kanban_home):
    with connect() as conn:
        tid = _ready(conn)
        _replay(conn, tid, T_18FA57CA_RUNS)
        assert kb.claim_task(conn, tid) is None
        assert kb.get_task(conn, tid).status == "triage"


def test_active_time_budget(kanban_home):
    with connect() as conn:
        tid = _ready(conn)
        _replay(conn, tid, [("review_requested", 13 * 3600), ("completed", 12 * 3600)],
                start=int(time.time()) - 30 * 3600)
        assert kb.claim_task(conn, tid) is None


def test_unblock_does_not_reset_but_owner_reset_does(kanban_home):
    with connect() as conn:
        tid = _ready(conn)
        _replay(conn, tid, T_18FA57CA_RUNS)
        assert kb.claim_task(conn, tid) is None
        # Ordinary unblock / reassign can't reopen it.
        assert kb.unblock_task(conn, tid) is False
        kb.assign_task(conn, tid, "other")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        conn.commit()
        assert kb.claim_task(conn, tid) is None
        assert kb.get_task(conn, tid).status == "triage"
        # Explicit owner reset re-arms the budget.
        assert kb.reset_run_budget(conn, tid, author="owner") is True
        assert kb.get_task(conn, tid).status in ("todo", "ready")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        conn.commit()
        assert kb.claim_task(conn, tid) is not None


def test_thresholds_from_config(kanban_home, monkeypatch):
    import hermes_cli.config as cfg
    monkeypatch.setattr(cfg, "load_config_readonly",
                        lambda: {"kanban": {"run_budget": {"max_rejections": 0, "max_active_hours": 0,
                                                           "max_repeat_holds": 0}}})
    with connect() as conn:
        tid = _ready(conn)
        _replay(conn, tid, T_763828CA_RUNS)
        assert kb.claim_task(conn, tid) is not None
