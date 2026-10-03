"""A live worker is never reclaimed or duplicated because its start-time fingerprint drifted.

macOS ``psutil`` create_time drifts by seconds on a healthy worker (measured: exactly the
drift-tolerance edge). Fingerprint mismatch alone then looks like PID recycle, the dispatcher
marks the task crashed, and the next tick spawns a second worker beside the one still running.
A descendant of this process cannot be a recycled stranger, and a live process whose argv
names the task id is proof a worker is still in flight.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _sleep_with_token(token: str) -> subprocess.Popen:
    """Child of this process, so it is a dispatcher descendant, with ``token`` as its own argv token."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", token],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=5)


def _drifted_fingerprint(pid: int) -> str:
    """Composed fingerprint for ``pid`` shifted far past the start-time tolerance."""
    recorded = kbd._process_fingerprint(pid)
    assert recorded is not None and "|" in recorded
    epoch, start = recorded.rsplit("|", 1)
    return f"{epoch}|{int(start) + 10_000}"


def _claim_running(conn, tid: str, pid: int, started_at) -> None:
    kb.claim_task(conn, tid)
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, worker_started_at = ?, started_at = ?, "
            "claim_expires = ? WHERE id = ?",
            (pid, started_at, old, old, tid),
        )


def _dead_pid() -> int:
    candidate = 2_147_483_647
    while candidate > 1 and kbd._pid_alive(candidate):
        candidate -= 1
    return candidate


def test_drifted_descendant_is_not_reclaimed(board):
    """A real child whose fingerprint is past the drift tolerance is still our worker.

    Old code: ``_pid_recycled`` is true, ``_worker_alive`` is false, and
    ``_reclaim_dead_workers`` releases the claim. The descendant guard must hold the claim.
    """
    conn = board
    tid = kb.create_task(conn, title="job", assignee="worker")
    proc = _sleep_with_token(tid)
    try:
        drifted = _drifted_fingerprint(proc.pid)
        assert kbd._pid_recycled(proc.pid, drifted) is True
        assert kbd._is_dispatcher_descendant(proc.pid) is True
        # Parentage is not identity; the task token plus ancestry is.
        assert kbd._worker_alive(proc.pid, drifted) is False
        assert kbd.is_live_task_worker(proc.pid, tid) is True
        _claim_running(conn, tid, proc.pid, drifted)

        sweep = kbd._reclaim_dead_workers(conn)
        task = kb.get_task(conn, tid)
        assert tid not in sweep.crashed
        assert task.status == "running"
        assert task.worker_pid == proc.pid
        assert task.claim_lock
    finally:
        _stop(proc)


def test_dispatch_refuses_spawn_while_live_worker_exists(board, all_assignees_spawnable):
    """A ready task whose id is already a live argv token must not be spawned again."""
    conn = board
    tid = kb.create_task(conn, title="job", assignee="worker")
    proc = _sleep_with_token(tid)
    calls = {"n": 0}

    def spawn_fn(*_args, **_kwargs):
        calls["n"] += 1
        return 4242

    try:
        result = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        assert calls["n"] == 0
        assert not any(row[0] == tid for row in result.spawned)
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.claim_lock is None
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None
        events = [e for e in kb.list_events(conn, tid) if e.kind == "spawn_refused_live_worker"]
        assert events
        assert proc.pid in events[-1].payload["pids"]
    finally:
        _stop(proc)


def test_dead_worker_still_reclaimed_and_prefix_id_does_not_match(board):
    """No live worker -> reclaim proceeds. ``t_abc`` must not match a ``t_abcd`` token."""
    conn = board
    tid = kb.create_task(conn, title="job", assignee="worker")
    prefix_proc = _sleep_with_token(tid + "x")
    try:
        assert prefix_proc.pid not in kbd._live_worker_pids_for_task(tid)
        assert kbd._argv_has_exact_task_token([tid + "x"], tid) is False
        assert kbd._argv_has_exact_task_token(["work kanban task " + tid], tid) is True
        _claim_running(conn, tid, _dead_pid(), "1|1")

        sweep = kbd._reclaim_dead_workers(conn)
        task = kb.get_task(conn, tid)
        assert tid in sweep.crashed
        assert task.status == "ready"
        assert task.worker_pid is None
    finally:
        _stop(prefix_proc)


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # Kill only the grandchild this test spawned after reparenting.
def test_reparented_marked_process_is_not_task_liveness(board):
    """A reparented grandchild is outside this dispatcher's ancestry, so it is not liveness."""
    conn = board
    tid = kb.create_task(conn, title="orphan", assignee="worker")
    # The intermediate process exits, leaving a live grandchild reparented to init.
    parent = subprocess.Popen(
        [sys.executable, "-c", "import os, time\npid = os.fork()\n"
         "if pid:\n print(pid, flush=True)\n os._exit(0)\n"
         "os.setsid()\ntime.sleep(120)", tid],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    pid = int(parent.stdout.readline().strip())
    parent.wait(timeout=5)
    try:
        assert not kbd._is_dispatcher_descendant(pid)
        assert kbd.is_live_task_worker(pid, tid) is False
        _claim_running(conn, tid, pid, _drifted_fingerprint(pid))
        # Ancestry is required. A reparented process is not this task's worker,
        # so a drifted fingerprint follows the normal reclaim path.
        assert kb.release_stale_claims(conn) == 1
        task = kb.get_task(conn, tid)
        assert task.status != "running"
        assert task.worker_pid != pid
        assert kbd._pid_alive(pid)
    finally:
        os.kill(pid, signal.SIGKILL)


def test_delayed_spawn_refusal_cannot_release_successor(board):
    conn = board
    tid = kb.create_task(conn, title="race", assignee="worker")
    first = kb.claim_task(conn, tid)
    assert first and first.current_run_id and first.claim_lock
    # A is superseded while its process scan is in flight. B now owns the card.
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
            "claim_expires = NULL, current_run_id = NULL WHERE id = ?", (tid,),
        )
    successor = kb.claim_task(conn, tid)
    assert successor and successor.current_run_id != first.current_run_id
    kbd._release_claim_spawn_refused_live_worker(
        conn, tid, [12345], run_id=first.current_run_id, claim_lock=first.claim_lock,
    )
    task = kb.get_task(conn, tid)
    assert task.status == "running"
    assert task.claim_lock == successor.claim_lock
    assert task.current_run_id == successor.current_run_id
    run = conn.execute("SELECT status, ended_at FROM task_runs WHERE id = ?",
                       (successor.current_run_id,)).fetchone()
    assert run["status"] == "running" and run["ended_at"] is None
    assert not any(e.kind == "spawn_refused_live_worker" for e in kb.list_events(conn, tid))


def test_done_reopened_is_dispatchable_even_with_recent_success_and_pr(
    board, monkeypatch, all_assignees_spawnable,
):
    conn = board
    tid = kb.create_task(conn, title="rework", assignee="worker")
    claimed = kb.claim_task(conn, tid)
    assert claimed
    kb.add_comment(conn, tid, author="worker", body="https://github.com/org/repo/pull/123")
    assert kb.complete_task(conn, tid, result="bad result")
    assert kb.reopen_done_task(conn, tid, actor="operator")[0]
    # Same second for the PR comment, the completed run, and done_reopened.
    # Bypass is event-id order, not a strict timestamp.
    pinned = int(time.time())
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_comments SET created_at = ? WHERE task_id = ?",
            (pinned, tid),
        )
        conn.execute(
            "UPDATE task_events SET created_at = ? "
            "WHERE task_id = ? AND kind IN ('commented', 'done_reopened')",
            (pinned, tid),
        )
        conn.execute(
            "UPDATE task_runs SET ended_at = ? WHERE task_id = ? AND outcome = 'completed'",
            (pinned, tid),
        )
    monkeypatch.setattr(kbd, "_active_pr_guard_applies", lambda _url: (True, None))
    assert kbd.check_respawn_guard(conn, tid) is None
    # Once the success window elapses, the open-PR guard must also recognize
    # this deliberate reopen (not mistake it for an accidental duplicate).
    with kb.write_txn(conn):
        conn.execute("UPDATE task_runs SET ended_at = ended_at - 7200 WHERE task_id = ?", (tid,))
    assert kbd.check_respawn_guard(conn, tid) is None
    spawned = []
    result = kbd.dispatch_once(conn, spawn_fn=lambda task, *_args: spawned.append(task.id) or None)
    assert tid in spawned
    assert any(row[0] == tid for row in result.spawned)


def test_wrong_task_descendant_is_not_this_task_worker(board):
    """A child whose argv names another task is not liveness for this task."""
    conn = board
    task_a = kb.create_task(conn, title="a", assignee="worker")
    child = _sleep_with_token("t_task_b")
    try:
        assert kbd.is_live_task_worker(child.pid, "t_task_a") is False
        _claim_running(conn, task_a, child.pid, _drifted_fingerprint(child.pid))
        assert kb.release_stale_claims(conn) == 1
        task = kb.get_task(conn, task_a)
        assert task.status != "running"
        assert task.worker_pid is None
    finally:
        _stop(child)


def test_dead_recorded_pid_rebinds_to_other_live_descendant(board, all_assignees_spawnable):
    """Recorded pid dead, another marked descendant alive: do not reclaim, rebind."""
    conn = board
    tid = kb.create_task(conn, title="a", assignee="worker")
    live = _sleep_with_token(tid)
    try:
        _claim_running(conn, tid, _dead_pid(), "1|1")
        assert kb.release_stale_claims(conn) == 0
        task = kb.get_task(conn, tid)
        assert task.status == "running"
        assert task.consecutive_failures == 0
        assert task.worker_pid == live.pid
        run = conn.execute(
            "SELECT worker_pid FROM task_runs WHERE id = ?", (task.current_run_id,),
        ).fetchone()
        assert run["worker_pid"] == live.pid

        dead = _dead_pid()
        old = int(time.time()) - 3600
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_pid = ?, worker_started_at = ?, claim_expires = ? "
                "WHERE id = ?",
                (dead, "1|1", old, tid),
            )
            conn.execute(
                "UPDATE task_runs SET worker_pid = ?, worker_started_at = ? WHERE id = ?",
                (dead, "1|1", task.current_run_id),
            )
        result = kbd.dispatch_once(conn, spawn_fn=lambda *_a, **_k: 9999)
        task = kb.get_task(conn, tid)
        assert result.reclaimed == 0
        assert tid not in result.crashed
        assert task.status == "running"
        assert task.consecutive_failures == 0
        assert task.worker_pid == live.pid
    finally:
        _stop(live)


def test_same_second_assignment_does_not_bypass_active_pr(board, monkeypatch):
    """A same-second assign is not a reopen. Fail closed."""
    conn = board
    tid = kb.create_task(conn, title="assigned", assignee="worker")
    kb.add_comment(conn, tid, author="worker", body="https://github.com/org/repo/pull/9")
    pinned = int(time.time())
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_comments SET created_at = ? WHERE task_id = ?", (pinned, tid),
        )
        conn.execute(
            "UPDATE task_events SET created_at = ? WHERE task_id = ? AND kind = 'commented'",
            (pinned, tid),
        )
        kb._append_event(
            conn, tid, "assigned", {"assignee": "other", "from": "worker"},
        )
        conn.execute(
            "UPDATE task_events SET created_at = ? WHERE task_id = ? AND kind = 'assigned'",
            (pinned, tid),
        )
    monkeypatch.setattr(kbd, "_active_pr_guard_applies", lambda _url: (True, None))
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_rebind_cas_does_not_clobber_successor_claim(board, tmp_path, monkeypatch):
    """Scan observes run N; a successor claim on another connection wins the rebind CAS."""
    conn = board
    tid = kb.create_task(conn, title="race", assignee="worker")
    live = _sleep_with_token(tid)
    other = kbc.connect(tmp_path / "kanban.db")
    try:
        recorded = _dead_pid()
        successor_pid = recorded - 1
        while successor_pid > 1 and (
            successor_pid == live.pid or kbd._pid_alive(successor_pid)
        ):
            successor_pid -= 1
        _claim_running(conn, tid, recorded, "1|1")
        observed = conn.execute(
            "SELECT current_run_id, worker_pid, claim_lock FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        state = {"done": False, "succ": None}

        def _steal(task_id, *, prefer=None):
            if state["done"] or task_id != tid:
                return live.pid
            state["done"] = True
            with kb.write_txn(other):
                other.execute(
                    "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                    "claim_expires = NULL, current_run_id = NULL, worker_pid = NULL, "
                    "worker_started_at = NULL WHERE id = ?",
                    (tid,),
                )
            succ = kb.claim_task(other, tid)
            assert succ and succ.current_run_id != observed["current_run_id"]
            with kb.write_txn(other):
                other.execute(
                    "UPDATE tasks SET worker_pid = ? WHERE id = ?",
                    (successor_pid, tid),
                )
                other.execute(
                    "UPDATE task_runs SET worker_pid = ? WHERE id = ?",
                    (successor_pid, succ.current_run_id),
                )
            state["succ"] = succ
            return live.pid

        monkeypatch.setattr(kbd, "find_live_task_worker", _steal)
        assert kb.release_stale_claims(conn) == 0
        task = kb.get_task(conn, tid)
        succ = state["succ"]
        assert succ is not None
        assert task.status == "running"
        assert task.current_run_id == succ.current_run_id
        assert task.worker_pid == successor_pid
        assert task.claim_lock == succ.claim_lock
        assert task.consecutive_failures == 0
        run = conn.execute(
            "SELECT worker_pid FROM task_runs WHERE id = ?",
            (succ.current_run_id,),
        ).fetchone()
        assert run["worker_pid"] == successor_pid
        assert not any(e.kind == "reclaimed" for e in kb.list_events(conn, tid))
    finally:
        other.close()
        _stop(live)


def test_live_worker_veto_before_workspace_failure(
    board, monkeypatch, all_assignees_spawnable,
):
    """A live same-task worker is refused before a workspace resolve can count a failure."""
    conn = board
    tid = kb.create_task(conn, title="job", assignee="worker")
    proc = _sleep_with_token(tid)
    calls = {"n": 0, "resolve": 0}

    def boom(*_args, **_kwargs):
        calls["resolve"] += 1
        raise RuntimeError("workspace unavailable")

    monkeypatch.setattr(kbd._kbw, "resolve_workspace", boom)
    monkeypatch.setattr(kbd._kbw, "_resolve_worktree_workspace", boom)

    def spawn_fn(*_args, **_kwargs):
        calls["n"] += 1
        return 4242

    try:
        kbd.dispatch_once(conn, spawn_fn=spawn_fn)
        task = kb.get_task(conn, tid)
        assert calls["n"] == 0
        assert calls["resolve"] == 0
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None
        assert not any(e.kind == "spawn_failed" for e in kb.list_events(conn, tid))
    finally:
        _stop(proc)


def test_done_reopened_after_pr_comment_ignores_later_same_second_comment(
    board, monkeypatch,
):
    """PR comment, then done_reopened, then an unrelated comment, all one second.

    The reopen is after THAT comment's event, so the guard must not return
    active_pr. A same-second assignment is still not a reopen (covered separately).
    """
    conn = board
    tid = kb.create_task(conn, title="rework", assignee="worker")
    kb.add_comment(conn, tid, author="worker", body="https://github.com/org/repo/pull/77")
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "done_reopened", {"actor": "operator", "status": "ready"})
    kb.add_comment(conn, tid, author="other", body="unrelated note")
    pinned = int(time.time())
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_comments SET created_at = ? WHERE task_id = ?",
            (pinned, tid),
        )
        conn.execute(
            "UPDATE task_events SET created_at = ? "
            "WHERE task_id = ? AND kind IN ('commented', 'done_reopened')",
            (pinned, tid),
        )
    monkeypatch.setattr(kbd, "_active_pr_guard_applies", lambda _url: (True, None))
    assert kbd.check_respawn_guard(conn, tid) is None


def test_manual_reclaim_does_not_signal_other_task_pid(board):
    """Operator reclaim releases the card but does not signal another task's live pid."""
    conn = board
    task_a = kb.create_task(conn, title="a", assignee="worker")
    task_b = kb.create_task(conn, title="b", assignee="worker")
    child = _sleep_with_token(task_b)
    signals = []
    try:
        fp = kbd._process_fingerprint(child.pid) or "1|1"
        _claim_running(conn, task_a, child.pid, fp)
        assert kb.reclaim_task(
            conn, task_a, reason="operator",
            signal_fn=lambda pid, sig: signals.append((pid, sig)),
        )
        assert signals == []
        assert child.poll() is None
        task = kb.get_task(conn, task_a)
        assert task.status != "running"
        assert task.claim_lock is None
    finally:
        _stop(child)
