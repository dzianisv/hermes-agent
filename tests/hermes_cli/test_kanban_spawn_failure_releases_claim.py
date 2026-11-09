"""Spawn faults release claims — phantom claims (runs 3045-3057) and spawn-fault accounting.

13 ready cards were claimed in 42s with no worker spawned because a workspace
fault escaped the tick without releasing the claim. The dispatcher must:

1. Release a claim that cannot be placed in the same tick (spawn fault).
2. Cap ATTEMPTS per tick, not only successful spawns.
3. Book a missing-interpreter exit as an infrastructure spawn fault, not a crash.
4. Release phantom claims (running, no pid, no heartbeat) on the next tick.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from tools.process_registry import RestartSafeScopeUnavailable


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Any profile name exists for testing
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda name: True)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home) -> sqlite3.Connection:
    """A writable kanban connection."""
    with kbc.connect() as c:
        yield c


def _exited_status(code: int) -> int:
    """Raw wait-status for a WIFEXITED child with the given exit code."""
    return code << 8


def test_raising_spawn_releases_the_claim_in_the_same_tick(kanban_home, conn, monkeypatch):
    """A failed spawn must release the claim immediately, not leave a phantom."""
    tid = kb.create_task(conn, title="spawn raises", assignee="dev")
    conn.commit()

    def raising_spawn(task, workspace, board=None):
        raise RestartSafeScopeUnavailable("test: cannot spawn")

    res = kbd.dispatch_once(conn, spawn_fn=raising_spawn)

    # The attempt was counted
    assert res.claim_attempts == 1
    # The claim was released
    task = kb.get_task(conn, tid)
    assert task.status == "ready"
    assert task.claim_lock is None
    assert task.worker_pid is None
    assert task.consecutive_failures == 0

    # No running tasks at all
    running_cnt = conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE status='running'"
    ).fetchone()[0]
    assert running_cnt == 0

    # Run recorded as spawn_failed
    outcomes = [r["outcome"] for r in conn.execute(
        "SELECT outcome FROM task_runs WHERE task_id=?", (tid,)
    ).fetchall()]
    assert "spawn_failed" in outcomes

    assert res.auto_blocked == []


def test_spawn_without_a_pid_is_a_phantom_claim_released(kanban_home, conn):
    """A spawn_fn returning None places no worker; the claim must be released."""
    tid = kb.create_task(conn, title="no pid", assignee="dev")
    conn.commit()

    def none_spawn(task, workspace, board=None):
        return None

    res = kbd.dispatch_once(conn, spawn_fn=none_spawn)

    assert res.claim_attempts == 1
    task = kb.get_task(conn, tid)
    assert task.status == "ready"
    assert task.claim_lock is None
    assert task.worker_pid is None
    assert task.consecutive_failures == 0

    # A no-pid spawn is booked as an infrastructure spawn fault, not a success.
    assert res.spawned == []


def test_oserror_spawn_is_infrastructure(kanban_home, conn, monkeypatch):
    """ENOENT from a missing interpreter is infrastructure: budget unchanged, card ready."""
    tid = kb.create_task(conn, title="enoent spawn", assignee="dev")
    conn.commit()

    def enoent_spawn(task, workspace, board=None):
        raise FileNotFoundError("Failed to find executable /x/bin/python3: No such file or directory")

    res = kbd.dispatch_once(conn, spawn_fn=enoent_spawn)

    assert res.claim_attempts == 1
    task = kb.get_task(conn, tid)
    assert task.status == "ready"
    assert task.consecutive_failures == 0


def test_pre_spawn_fault_releases_the_claim(kanban_home, conn, monkeypatch):
    """A fault between claim and spawn (resume plan) releases the claim."""
    tid = kb.create_task(conn, title="pre-spawn fault", assignee="dev")
    conn.commit()

    # The spawn would succeed, but pre-spawn faults
    good_spawn = lambda task, workspace, board=None: 1234

    # Patch _apply_worker_resume_plan to raise
    called = []
    orig_resume = kbd._apply_worker_resume_plan
    def faulting_resume(c, task):
        called.append(task.id)
        raise RuntimeError("boom")
    monkeypatch.setattr(kbd, "_apply_worker_resume_plan", faulting_resume)

    res = kbd.dispatch_once(conn, spawn_fn=good_spawn)

    assert res.claim_attempts == 1
    task = kb.get_task(conn, tid)
    assert task.status == "ready"
    assert task.claim_lock is None
    assert task.worker_pid is None

    # Run recorded
    outcomes = [r["outcome"] for r in conn.execute(
        "SELECT outcome FROM task_runs WHERE task_id=?", (tid,)
    ).fetchall()]
    assert "spawn_failed" in outcomes


def test_next_tick_retries_after_a_spawn_fault(kanban_home, conn, monkeypatch):
    """A spawn fault card is re-dispatchable on the next tick."""
    # Disable infrastructure cooldown so the retry happens immediately
    monkeypatch.setattr(kb, "_resolve_rate_limit_cooldown_seconds", lambda: 0)

    tid = kb.create_task(conn, title="retry after fault", assignee="dev")
    conn.commit()

    # First tick: fault
    def fault_spawn(task, workspace, board=None):
        raise RestartSafeScopeUnavailable("test fault")

    res1 = kbd.dispatch_once(conn, spawn_fn=fault_spawn)
    assert res1.claim_attempts == 1
    assert res1.spawned == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready"

    # Second tick: success
    def good_spawn(task, workspace, board=None):
        return 4321

    res2 = kbd.dispatch_once(conn, spawn_fn=good_spawn)
    # We may see a phantom release event or not; what matters is the spawn now works
    assert any(tid == t[0] for t in res2.spawned), f"Expected {tid} in spawned {res2.spawned}"
    conn.commit()
    task = kb.get_task(conn, tid)
    assert task.status == "running"
    assert task.worker_pid == 4321


def test_phantom_claim_is_released_by_the_dispatcher(kanban_home, conn):
    """A claim that survived the tick with no pid/heartbeat is released next tick."""
    tid = kb.create_task(conn, title="phantom", assignee="dev")
    conn.commit()

    # Manually claim without spawning (simulates a dispatcher crash mid-tick)
    claimed = kb.claim_task(conn, tid, ttl_seconds=900)
    assert claimed is not None
    run_id = claimed.current_run_id
    conn.commit()

    # Age the run past the grace window
    conn.execute(
        "UPDATE task_runs SET started_at = started_at - 600 WHERE id = ?",
        (run_id,)
    )
    conn.commit()

    # Call release_phantom_claims directly
    released = kbd.release_phantom_claims(conn)

    assert released == [tid]
    task = kb.get_task(conn, tid)
    assert task.status == "ready"
    assert task.claim_lock is None
    assert task.consecutive_failures == 0

    # Run outcome is spawn_failed with phantom_claim reason
    row = conn.execute(
        "SELECT outcome, metadata FROM task_runs WHERE id=?", (run_id,)
    ).fetchone()
    assert row["outcome"] == "spawn_failed"
    meta = kb._json_dict(row["metadata"])
    assert meta.get("reason") == "phantom_claim"

    # Event recorded
    events = [e for e in kb.list_events(conn, tid) if e.kind == "spawn_failed"]
    assert len(events) >= 1
    assert events[-1].payload.get("reason") == "phantom_claim"


def test_a_failing_tick_cannot_claim_more_cards_than_the_budget(kanban_home, conn):
    """A systemic spawn fault cannot claim the entire ready queue (13 cards, 42s)."""
    tids = []
    for i in range(5):
        t = kb.create_task(conn, title=f"card {i}", assignee="dev")
        tids.append(t)
    conn.commit()

    def raising_spawn(task, workspace, board=None):
        raise RestartSafeScopeUnavailable("systemic")

    res = kbd.dispatch_once(conn, spawn_fn=raising_spawn, max_spawn=2)

    # Exactly 2 attempts, not 5
    assert res.claim_attempts == 2

    # All cards still ready
    running = conn.execute("SELECT COUNT(*) FROM tasks WHERE status='running'").fetchone()[0]
    assert running == 0
    ready = conn.execute("SELECT COUNT(*) FROM tasks WHERE status='ready'").fetchone()[0]
    assert ready == 5

    # Exactly 2 claimed events were written this tick
    claimed_events = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE kind='claimed'"
    ).fetchone()[0]
    assert claimed_events == 2


def test_interpreter_missing_exit_is_a_spawn_fault_not_a_crash(kanban_home, conn, monkeypatch):
    """Modeled on test_rate_limit_exit_requeues: missing interpreter is infrastructure."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="interpreter missing", assignee="dev")

    # Simulate 3 spawn-fault exits
    for i in range(3):
        pid = 80000 + i
        kb.claim_task(conn, tid, claimer=f"{host}:w{i}")
        conn.execute(
            "UPDATE tasks SET worker_pid=?, consecutive_failures=? WHERE id=?",
            (pid, 0, tid),
        )
        conn.commit()

        # Write the worker log with the spawn fault signature
        log_dir = kb.worker_logs_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{tid}.log"
        log_path.write_text(
            f"Failed to find executable /home/engineer/.hermes/tools/python-3.14.7+xxxx/bin/python3: "
            f"No such file or directory\n",
            encoding="utf-8"
        )

        # Record exit and detect
        kbd._record_worker_exit(pid, _exited_status(1))
        crashed = kbd.detect_crashed_workers(conn)

        # Should NOT be in crashed (it's a spawn fault)
        assert tid not in crashed, f"iteration {i}: spawn fault should not be a crash"
        sf = getattr(kbd.detect_crashed_workers, "_last_spawn_faults", [])
        assert tid in sf, f"iteration {i}: should be in spawn_faults"

        task = kb.get_task(conn, tid)
        assert task.status == "ready", f"iteration {i}: status={task.status}"
        assert task.consecutive_failures == 0, f"iteration {i}: failures={task.consecutive_failures}"

    # After 3 exits, still not blocked (spawn faults don't trip breaker)
    assert tid not in getattr(kbd.detect_crashed_workers, "_last_auto_blocked", [])

    # Run outcomes are spawn_failed, not crashed
    outcomes = [r["outcome"] for r in conn.execute(
        "SELECT outcome FROM task_runs WHERE task_id=?", (tid,)
    ).fetchall()]
    for o in outcomes:
        assert o == "spawn_failed", f"Expected spawn_failed, got {o}"
