"""An interrupted Kanban worker is re-spawned into its own session, not a blank one.

Contract: after a run ends crashed / timed_out / reclaimed and that worker had recorded its
session (which it does as soon as its agent is built), the next spawn for the same assignee
is ``chat --resume <session> -q "You were interrupted ..."``. Everything else — a normal
handoff, an unknown or missing session, a different assignee, a resumed worker that never
got its session loaded, or the consecutive-resume cap — keeps the fresh ``-q`` start.
Real kanban DB and real state.db under a temp HERMES_HOME; only the process spawn is faked.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_worker_session import stamp_worker_session_on_run
from hermes_state import SessionDB

FRESH_TAIL = ["chat", "-q"]


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kbd, "_resolve_worker_cli_toolsets", lambda _home: None)
    kb.init_db()
    return h


def _make_session(home: Path, session_id: str) -> None:
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(session_id=session_id, source="kanban")
        db.append_message(session_id=session_id, role="user", content="work kanban task")
    finally:
        db.close()


class _Spawner:
    """Records the argv the real ``_worker_argv`` would launch; reports a PID."""

    def __init__(self):
        self.argvs: list[list[str]] = []
        self.tasks: list[kb.Task] = []

    def __call__(self, task, workspace, board=None):
        self.tasks.append(task)
        self.argvs.append(kbd._worker_argv(task, task.assignee, None))
        return 999_999


def _dispatch(conn, spawner):
    kbd.dispatch_once(conn, spawn_fn=spawner, failure_limit=50)
    return spawner.argvs[-1]


def _worker_records_session(monkeypatch, conn, task_id, session_id):
    """What a real worker does once its agent is built: stamp its session on its run."""
    run_id = kb.get_task(conn, task_id).current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    try:
        assert stamp_worker_session_on_run(session_id)
    finally:
        monkeypatch.delenv("HERMES_KANBAN_TASK")
        monkeypatch.delenv("HERMES_KANBAN_RUN_ID")


def _end_run(conn, task_id, outcome, monkeypatch):
    """Drive the run to ``outcome`` through the real dispatcher/operator paths."""
    if outcome == "reclaimed":
        assert kb.reclaim_task(conn, task_id, reason="manual", signal_fn=lambda *_a, **_k: None)
    elif outcome == "crashed":
        with kb.write_txn(conn):  # past the launch-window grace
            conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (int(time.time()) - 3600, task_id))
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        monkeypatch.setattr(kbd, "_worker_alive", lambda *_a: False)
        assert task_id in kbd.detect_crashed_workers(conn)
    elif outcome == "timed_out":
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET max_runtime_seconds = 1, started_at = ? WHERE id = ?",
                         (int(time.time()) - 3600, task_id))
            conn.execute("UPDATE task_runs SET max_runtime_seconds = 1, started_at = ? "
                         "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                         (int(time.time()) - 3600, task_id))
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        assert task_id in kbd.enforce_max_runtime(conn, signal_fn=lambda *_a, **_k: None)
    elif outcome == "completed":
        assert kb.complete_task(conn, task_id, result="done", summary="done")
        # Re-open so a later dispatch exists to observe.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
    else:  # pragma: no cover
        raise AssertionError(outcome)
    last = kb.list_runs(conn, task_id)[-1]
    assert last.outcome == outcome


def _new_task(conn, assignee="default"):
    return kb.create_task(conn, title="ship it", assignee=assignee)


def _assert_fresh(argv, task_id):
    assert "--resume" not in argv
    assert argv[-3:] == [*FRESH_TAIL, f"work kanban task {task_id}"]


@pytest.mark.parametrize("outcome", ["crashed", "timed_out", "reclaimed"])
def test_interrupted_run_with_known_session_is_resumed(home, monkeypatch, outcome):
    _make_session(home, "20261007_084126_b25abd")
    spawner = _Spawner()
    with kbc.connect_closing() as conn:
        tid = _new_task(conn)
        _assert_fresh(_dispatch(conn, spawner), tid)
        _worker_records_session(monkeypatch, conn, tid, "20261007_084126_b25abd")
        first_run = kb.get_task(conn, tid).current_run_id
        _end_run(conn, tid, outcome, monkeypatch)
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)

        argv = _dispatch(conn, spawner)

        i = argv.index("chat")
        assert argv[i:i + 4] == ["chat", "--resume", "20261007_084126_b25abd", "-q"]
        assert argv[i + 4] == (
            f"You were interrupted (previous run {outcome}). Continue kanban task {tid} "
            "from where you stopped; re-read the card first.")
        # Visible on the run and as an event (what `hermes kanban show` prints).
        run = kb.list_runs(conn, tid)[-1]
        assert run.metadata["resumed_from_session"] == "20261007_084126_b25abd"
        assert run.metadata["resume_of_run"] == first_run
        assert run.metadata["resume_outcome"] == outcome
        kinds = [(e.kind, e.payload) for e in kb.list_events(conn, tid) if e.kind == "worker_resumed"]
        assert kinds and kinds[-1][1]["session_id"] == "20261007_084126_b25abd"


def test_first_dispatch_and_normal_handoff_stay_fresh(home, monkeypatch):
    _make_session(home, "s_done")
    spawner = _Spawner()
    with kbc.connect_closing() as conn:
        tid = _new_task(conn)
        _assert_fresh(_dispatch(conn, spawner), tid)
        _worker_records_session(monkeypatch, conn, tid, "s_done")
        _end_run(conn, tid, "completed", monkeypatch)
        _assert_fresh(_dispatch(conn, spawner), tid)
        assert spawner.tasks[-1].resume_session_id is None


def test_fresh_argv_is_unchanged_by_the_feature(home):
    task = kb.Task(
        id="t_x", title="t", body=None, assignee="default", status="running", priority=0,
        created_by=None, created_at=1, started_at=None, completed_at=None,
        workspace_kind="scratch", workspace_path=None, claim_lock=None, claim_expires=None, tenant=None,
    )
    assert kbd._worker_argv(task, "default", None) == [
        "hermes", "-p", "default", "--cli", "--accept-hooks", "chat", "-q", "work kanban task t_x"]


def test_crash_without_recorded_session_starts_fresh(home, monkeypatch):
    spawner = _Spawner()
    with kbc.connect_closing() as conn:
        tid = _new_task(conn)
        _dispatch(conn, spawner)
        _end_run(conn, tid, "crashed", monkeypatch)
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
        _assert_fresh(_dispatch(conn, spawner), tid)


def test_session_missing_from_state_db_starts_fresh(home, monkeypatch):
    _make_session(home, "someone_else")  # state.db exists, but not this session
    spawner = _Spawner()
    with kbc.connect_closing() as conn:
        tid = _new_task(conn)
        _dispatch(conn, spawner)
        _worker_records_session(monkeypatch, conn, tid, "s_gone")
        _end_run(conn, tid, "reclaimed", monkeypatch)
        _assert_fresh(_dispatch(conn, spawner), tid)


def test_reassigned_task_does_not_resume_another_profiles_session(home, monkeypatch):
    _make_session(home, "s_prev")
    spawner = _Spawner()
    with kbc.connect_closing() as conn:
        tid = _new_task(conn)
        _dispatch(conn, spawner)
        _worker_records_session(monkeypatch, conn, tid, "s_prev")
        _end_run(conn, tid, "reclaimed", monkeypatch)
        # The previous run belonged to someone else (as if reassigned since).
        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET profile = 'other' WHERE task_id = ?", (tid,))
        _assert_fresh(_dispatch(conn, spawner), tid)


def test_resume_that_never_loaded_falls_back_to_fresh(home, monkeypatch):
    """The resumed worker died before its agent built (no stamp): next spawn is fresh."""
    _make_session(home, "s_bad")
    spawner = _Spawner()
    with kbc.connect_closing() as conn:
        tid = _new_task(conn)
        _dispatch(conn, spawner)
        _worker_records_session(monkeypatch, conn, tid, "s_bad")
        _end_run(conn, tid, "crashed", monkeypatch)
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
        assert "--resume" in _dispatch(conn, spawner)
        # Resumed worker exits immediately without stamping its session.
        _end_run(conn, tid, "crashed", monkeypatch)
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
        _assert_fresh(_dispatch(conn, spawner), tid)


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_consecutive_resumes_are_capped(home, monkeypatch, cap):
    (home / "config.yaml").write_text(f"kanban:\n  resume_interrupted_max: {cap}\n")
    assert kbd.resume_interrupted_max() == cap
    _make_session(home, "s_loop")
    spawner = _Spawner()
    with kbc.connect_closing() as conn:
        tid = _new_task(conn)
        _dispatch(conn, spawner)
        resumed = 0
        for _ in range(cap + 1):
            _worker_records_session(monkeypatch, conn, tid, "s_loop")
            _end_run(conn, tid, "crashed", monkeypatch)
            monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
            argv = _dispatch(conn, spawner)
            if "--resume" not in argv:
                _assert_fresh(argv, tid)
                break
            resumed += 1
        assert resumed == cap


def test_cap_zero_disables_resume(home, monkeypatch):
    (home / "config.yaml").write_text("kanban:\n  resume_interrupted_max: 0\n")
    _make_session(home, "s_off")
    spawner = _Spawner()
    with kbc.connect_closing() as conn:
        tid = _new_task(conn)
        _dispatch(conn, spawner)
        _worker_records_session(monkeypatch, conn, tid, "s_off")
        _end_run(conn, tid, "reclaimed", monkeypatch)
        _assert_fresh(_dispatch(conn, spawner), tid)


def test_default_cap_is_registered_and_two(home):
    from hermes_cli.config import load_config

    assert load_config()["kanban"]["resume_interrupted_max"] == 2
    assert kbd.resume_interrupted_max() == 2
