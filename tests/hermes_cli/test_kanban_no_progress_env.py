"""Mechanism A: host/infrastructure failures must not count as "no progress".

Live defect (card t_bde1784e, 2026-11-09): the no-progress guard re-spec'd a healthy
card to the architect twice (streak 6, then streak 3) because its trailing runs had
died on the HOST — a missing pinned interpreter, a broken dependency environment, an
unreadable workspace — and every non-progress outcome counted toward the streak.

Real temp DB, real functions; only observation (``head_fn``) is injected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_wait_on as kwo
from tests.hermes_cli._kanban_modules import KanbanModules

kb = KanbanModules()

LAUNCHER = ("pid 3115 not alive Worker's last output: 'Failed to find executable "
            "/home/engineer/.hermes/tools/python-3.14.7-linux-x64/bin/python3: "
            "No such file or directory'")
NO_DEPS = ("pid 3142 not alive Worker's last output: 'hermes: no dependency environment "
           "is committed for this install; run `hermes pm repair`'")
WS_FAULT = "workspace: [Errno 13] Permission denied: '/root/.hermes/kanban/boards/vibebrowser'"
CARD_CRASH = "pid 4000 not alive Worker's last output: 'AssertionError: expected 200 but got 404'"


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


def _events(conn, tid, kind):
    return conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?", (tid, kind)).fetchall()


def _comments(conn, tid):
    return [r["body"] for r in conn.execute(
        "SELECT body FROM task_comments WHERE task_id = ?", (tid,))]


def _stamp_crash(conn, tid, *, error, session=True, head="c1"):
    """Book a closed run the way the reaper does: restore the phase, close the run."""
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    run_id = claimed.current_run_id
    meta = {"head_start": head}
    if session:
        meta["worker_session_id"] = f"sess-{run_id}"
    kb.merge_run_metadata(conn, tid, run_id, meta)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, claim_expires = NULL, "
            "worker_pid = NULL, worker_started_at = NULL WHERE id = ?", (tid,))
        kb._end_run(conn, tid, outcome="crashed", status="crashed", error=error)
    return run_id


def _stamp_no_progress(conn, tid, *, head="c1"):
    """A real worker run that ended with no commit, no status change, no green check."""
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    run_id = claimed.current_run_id
    kb.merge_run_metadata(conn, tid, run_id,
                          {"head_start": head, "worker_session_id": f"sess-{run_id}"})
    assert kb.schedule_task(conn, tid, reason="CI still red", expected_run_id=run_id)
    kb.unblock_task(conn, tid)
    return run_id


def _head(_ws):
    return "c1"


def _guard(conn, tid, **kw):
    kw.setdefault("architect", "swe-architect")
    kw.setdefault("limit", 2)
    return kwo.no_progress_guard(conn, tid, "software-engineer", None, head_fn=_head, **kw)


def test_environmental_crashes_never_respec(conn):
    tid = kb.create_task(conn, title="impl", assignee="software-engineer")
    for _ in range(3):
        _stamp_crash(conn, tid, error=LAUNCHER, session=False)
    assert kwo.no_progress_streak(conn, tid, "c1") == 0
    assert _guard(conn, tid) is None
    assert kb.get_task(conn, tid).assignee == "software-engineer"
    assert len(_events(conn, tid, "host_incident")) == 1
    assert any("BLOCKER:HARNESS infra" in body for body in _comments(conn, tid))
    # Repeated calls on the same broken host must not re-announce the incident.
    assert _guard(conn, tid) is None
    assert len(_events(conn, tid, "host_incident")) == 1


def test_dependency_environment_error_is_environmental(conn):
    tid = kb.create_task(conn, title="impl", assignee="software-engineer")
    for _ in range(2):
        _stamp_crash(conn, tid, error=NO_DEPS, session=False)
    assert kwo.no_progress_streak(conn, tid, "c1") == 0
    assert _guard(conn, tid) is None
    assert kb.get_task(conn, tid).assignee == "software-engineer"


def test_workspace_permission_fault_is_environmental(conn):
    tid = kb.create_task(conn, title="impl", assignee="software-engineer")
    for _ in range(2):
        _stamp_crash(conn, tid, error=WS_FAULT, session=False)
    assert kwo.no_progress_streak(conn, tid, "c1") == 0
    assert _guard(conn, tid) is None
    assert kb.get_task(conn, tid).assignee == "software-engineer"


def test_no_agent_start_is_neutral(conn):
    """No session stamp and no heartbeat: the worker never reached the card, so the
    run is neutral even when its error text is not a known host signature."""
    tid = kb.create_task(conn, title="impl", assignee="software-engineer")
    for _ in range(3):
        _stamp_crash(conn, tid, error=CARD_CRASH, session=False)
    assert kwo.no_progress_streak(conn, tid, "c1") == 0
    assert _guard(conn, tid) is None
    assert kb.get_task(conn, tid).assignee == "software-engineer"


def test_genuine_no_progress_still_respecs(conn):
    tid = kb.create_task(conn, title="impl", assignee="software-engineer")
    _stamp_no_progress(conn, tid)
    _stamp_no_progress(conn, tid)
    assert kwo.no_progress_streak(conn, tid, "c1") == 2
    assert _guard(conn, tid) == "no_progress_respec"
    assert kb.get_task(conn, tid).assignee == "swe-architect"
    assert _events(conn, tid, "no_progress_respec")


def test_environmental_run_does_not_reset_a_genuine_streak(conn):
    tid = kb.create_task(conn, title="impl", assignee="software-engineer")
    _stamp_no_progress(conn, tid)
    _stamp_crash(conn, tid, error=LAUNCHER, session=False)
    assert kwo.no_progress_streak(conn, tid, "c1") == 1
    assert _guard(conn, tid) is None
    _stamp_no_progress(conn, tid)
    assert kwo.no_progress_streak(conn, tid, "c1") == 2
    assert _guard(conn, tid) == "no_progress_respec"


def test_respec_reset_clears_the_streak(conn):
    tid = kb.create_task(conn, title="impl", assignee="software-engineer")
    _stamp_no_progress(conn, tid)
    _stamp_no_progress(conn, tid)
    assert _guard(conn, tid) == "no_progress_respec"
    kb.assign_task(conn, tid, "software-engineer")
    assert kwo.reset_no_progress(conn, tid, reason="infra crash", author="operator") is True
    assert kwo.no_progress_streak(conn, tid, "c1") == 0
    assert _guard(conn, tid) is None
    assert kwo.reset_no_progress(conn, "t_doesnotexist") is False
