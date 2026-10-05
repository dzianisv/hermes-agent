"""Spawn fence: do not start a worker while a live process already carries the task.

The dispatcher matches ``worker_task_argv_token(task_id)`` as a whole argv
element (never a substring) anywhere on the host. A hit is a respawn-guard
deferral — no claim, no run row, no failure increment — so a lost claim cannot
double-spawn a worker that is still alive.
"""

from __future__ import annotations

import subprocess
import sys

import psutil
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


class _Proc:
    """psutil-like process: the scanner reads ``.info`` and nothing else."""

    def __init__(self, pid, ppid, cmdline, status="running"):
        self.info = {
            "pid": pid,
            "ppid": ppid,
            "cmdline": cmdline,
            "status": status,
        }


class _Denied:
    """Raises when the scanner touches ``.info``."""

    @property
    def info(self):
        raise psutil.AccessDenied(pid=1)


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _spawn_recorder():
    calls = []

    def spawn_fn(task, workspace, board=None):
        calls.append(task.id)
        return 1

    return calls, spawn_fn


def _dispatch(conn, tid, spawn_fn, *, lane="ready"):
    row = conn.execute(
        "SELECT id, assignee FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    result = kbd.DispatchResult()
    consumed = kbd._dispatch_lane_task(
        conn, row, row["assignee"], result,
        lane=lane, dry_run=False, ttl_seconds=None, board=None,
        failure_limit=kbd.DEFAULT_FAILURE_LIMIT, spawn_fn=spawn_fn,
        per_profile_cap=None, per_profile_running={},
    )
    return consumed, result


def _force_task_id(conn, task_id: str) -> str:
    """Create a ready assigned task and pin its id (prefix-sharing cases)."""
    created = kb.create_task(conn, title=task_id, assignee="alice")
    if created == task_id:
        return task_id
    with kb.write_txn(conn):
        for table in (
            "task_events", "task_comments", "task_runs",
            "task_attachments", "kanban_notify_subs",
        ):
            conn.execute(
                f"UPDATE {table} SET task_id = ? WHERE task_id = ?",
                (task_id, created),
            )
        conn.execute("UPDATE tasks SET id = ? WHERE id = ?", (task_id, created))
    return task_id


def _guard_events(conn, tid):
    return [e for e in kb.list_events(conn, tid) if e.kind == "respawn_guarded"]


def test_worker_argv_uses_exact_token(monkeypatch):
    """``_worker_argv`` must pass the token function's string as ``-q``."""
    from types import SimpleNamespace

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    task = SimpleNamespace(
        id="t_007efb64", skills=None, model_override=None,
        provider_override=None, reasoning_effort=None,
    )
    argv = kbd._worker_argv(task, "alice", None)
    token = kbd.worker_task_argv_token("t_007efb64")
    assert token == "work kanban task t_007efb64"
    assert argv[argv.index("-q") + 1] == token


def test_exact_token_refuses_spawn_without_claim(
    board, all_assignees_spawnable, monkeypatch,
):
    """A live process with the exact token blocks the lane: no spawn, no claim."""
    conn = board
    tid = kb.create_task(conn, title="live", assignee="alice")
    token = kbd.worker_task_argv_token(tid)
    pid, ppid = 424242, 7
    monkeypatch.setattr(
        kbd, "_live_worker_proc_iter",
        lambda: [_Proc(pid, ppid, ["hermes", "chat", "-q", token])],
    )
    calls, spawn_fn = _spawn_recorder()
    consumed, result = _dispatch(conn, tid, spawn_fn)
    assert consumed is False
    assert calls == []
    assert result.respawn_guarded == [(tid, "live_worker_process")]
    task = kb.get_task(conn, tid)
    assert task.claim_lock is None
    assert task.status == "ready"
    assert task.consecutive_failures == 0
    assert task.current_run_id is None
    assert conn.execute(
        "SELECT id FROM task_runs WHERE task_id = ?", (tid,),
    ).fetchall() == []
    events = _guard_events(conn, tid)
    assert len(events) == 1
    assert events[0].payload["reason"] == "live_worker_process"
    assert events[0].payload["pids"] == [pid]
    assert events[0].payload["ppids"] == [ppid]


def test_review_lane_exact_token_refuses_spawn(
    board, all_assignees_spawnable, monkeypatch,
):
    """The fence is not ready-lane-only: a review row is refused the same way."""
    conn = board
    tid = kb.create_task(conn, title="review-live", assignee="alice")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
    token = kbd.worker_task_argv_token(tid)
    monkeypatch.setattr(
        kbd, "_live_worker_proc_iter",
        lambda: [_Proc(99, 1, [token])],
    )
    calls, spawn_fn = _spawn_recorder()
    consumed, result = _dispatch(conn, tid, spawn_fn, lane="review")
    assert consumed is False
    assert calls == []
    assert result.respawn_guarded == [(tid, "live_worker_process")]
    assert kb.get_task(conn, tid).claim_lock is None
    assert len(_guard_events(conn, tid)) == 1


def test_prefix_and_embedded_token_do_not_match(
    board, all_assignees_spawnable, monkeypatch,
):
    """Prefix-sharing ids and a token buried in a longer element are not a hit.

    ``t_00`` vs ``t_007efb64`` (both directions) must not fence each other, and
    the token inside a larger argv string is not an element match. Spawn proceeds.
    """
    conn = board
    long_id = _force_task_id(conn, "t_007efb64")
    short_id = _force_task_id(conn, "t_00")
    long_token = kbd.worker_task_argv_token(long_id)
    short_token = kbd.worker_task_argv_token(short_id)
    longer_token = kbd.worker_task_argv_token("t_007efb64abcd")
    # Token for t_00 (and a longer id, and the long token buried in one
    # element) must not fence t_007efb64.
    monkeypatch.setattr(
        kbd, "_live_worker_proc_iter",
        lambda: [
            _Proc(11, 1, ["hermes", short_token]),
            _Proc(12, 1, ["hermes", longer_token]),
            _Proc(13, 1, ["hermes", f"please {long_token} now"]),
            _Proc(14, 1, [f"prefix-{long_token}"]),
        ],
    )
    assert kbd.find_live_workers_for_task(long_id) == []
    calls, spawn_fn = _spawn_recorder()
    consumed, result = _dispatch(conn, long_id, spawn_fn)
    assert consumed is True
    assert calls == [long_id]
    assert result.respawn_guarded == []
    assert result.spawned

    # Vice versa: token for t_007efb64 must not fence t_00.
    monkeypatch.setattr(
        kbd, "_live_worker_proc_iter",
        lambda: [
            _Proc(21, 1, ["hermes", long_token]),
            _Proc(22, 1, [f"x {short_token} y"]),
        ],
    )
    assert kbd.find_live_workers_for_task(short_id) == []
    calls_short, spawn_short = _spawn_recorder()
    consumed_short, result_short = _dispatch(conn, short_id, spawn_short)
    assert consumed_short is True
    assert calls_short == [short_id]
    assert result_short.respawn_guarded == []


def test_zombie_with_token_is_ignored(
    board, all_assignees_spawnable, monkeypatch,
):
    """A zombie still holding the token is not a live worker. Spawn proceeds."""
    conn = board
    tid = kb.create_task(conn, title="zombie", assignee="alice")
    token = kbd.worker_task_argv_token(tid)
    monkeypatch.setattr(
        kbd, "_live_worker_proc_iter",
        lambda: [_Proc(55, 1, ["hermes", token], status=psutil.STATUS_ZOMBIE)],
    )
    calls, spawn_fn = _spawn_recorder()
    consumed, result = _dispatch(conn, tid, spawn_fn)
    assert consumed is True
    assert calls == [tid]
    assert result.respawn_guarded == []


def test_access_denied_and_empty_cmdline_are_skipped(
    board, all_assignees_spawnable, monkeypatch,
):
    """Unreadable and empty-cmdline processes are skipped. Spawn proceeds."""
    conn = board
    tid = kb.create_task(conn, title="denied", assignee="alice")
    monkeypatch.setattr(
        kbd, "_live_worker_proc_iter",
        lambda: [
            _Denied(),
            _Proc(4, 1, None),
            _Proc(5, 1, []),
            _Proc(6, 1, ""),
        ],
    )
    calls, spawn_fn = _spawn_recorder()
    consumed, result = _dispatch(conn, tid, spawn_fn)
    assert consumed is True
    assert calls == [tid]
    assert result.respawn_guarded == []
    assert kbd.find_live_workers_for_task(tid) == []


def test_proc_iter_exception_fails_open_and_tick_spawns(
    board, all_assignees_spawnable, monkeypatch, caplog,
):
    """A scanner crash returns [] and the tick still spawns (fail open)."""
    def _boom():
        raise RuntimeError("scanner down")

    monkeypatch.setattr(kbd, "_live_worker_proc_iter", _boom)
    with caplog.at_level("WARNING", logger="hermes_cli.kanban_db"):
        assert kbd.find_live_workers_for_task("t_any") == []
    assert any("failing open" in r.getMessage() for r in caplog.records)

    conn = board
    tid = kb.create_task(conn, title="open", assignee="alice")
    calls, spawn_fn = _spawn_recorder()
    result = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
    assert calls == [tid]
    assert any(row[0] == tid for row in result.spawned)
    assert result.respawn_guarded == []


@pytest.mark.platforms("posix")
def test_real_process_blocks_dispatch_until_exit(
    board, all_assignees_spawnable, monkeypatch,
):
    """A real host process (no injected proc_iter) fences two ticks, then one spawn."""
    monkeypatch.setattr(kbd, "_live_worker_proc_iter", None)
    conn = board
    tid = kb.create_task(conn, title="real", assignee="alice")
    token = kbd.worker_task_argv_token(tid)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", token],
    )
    calls = []

    def spawn_fn(task, workspace, board=None):
        calls.append(task.id)
        return 1

    try:
        first = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        second = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        assert calls == []
        assert first.respawn_guarded == [(tid, "live_worker_process")]
        assert second.respawn_guarded == [(tid, "live_worker_process")]
        guarded = _guard_events(conn, tid)
        assert len(guarded) == 2
        assert all(
            e.payload.get("reason") == "live_worker_process" and proc.pid in e.payload.get("pids", [])
            for e in guarded
        )
        proc.terminate()
        proc.wait(timeout=10)
        third = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        assert calls == [tid]
        assert any(row[0] == tid for row in third.spawned)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
