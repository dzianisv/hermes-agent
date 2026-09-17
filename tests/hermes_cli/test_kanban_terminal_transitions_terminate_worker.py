"""Tests: the TERMINAL transitions must not orphan a live worker, and a
verified-bad ``done`` card must have a supported route back to rework.

Part A. ``complete_task`` and ``block_task`` used to NULL
``claim_lock``/``claim_expires``/``worker_pid`` and end the run without ever
terminating the worker process — the same shape already fixed in
``request_review``/``request_changes``. The row then held no handle on a
process that could still be writing in the card's workspace. Unlike the review
paths there is no duplicate writer (``done``/``blocked``/``triage`` are not
re-claimed), but ``block_task(kind='dependency')`` lands in ``todo``, which IS
immediately re-claimable.

Part B. ``reopen_review_task`` is ``WHERE status = 'review'`` and
``promote_task`` accepts only ``todo``/``blocked``, so a ``done`` card that is
later verified bad had no supported transition back. ``reopen_done_task`` is
that route, and it must obey the same terminate-before-release invariant and
park only self-healing holds.

The invariant asserted throughout: after a transition driven by someone other
than the current worker, there is never a RELEASED (claimable) task sitting
beside a live previous writer. Either the worker is dead, or the task is held.

Every process signalled in this file is spawned and owned by the test itself.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


@pytest.fixture
def owned_sleeper():
    """A real process this test owns; always reaped."""
    procs: list[subprocess.Popen] = []

    def _spawn() -> subprocess.Popen:
        p = subprocess.Popen(["sleep", "30"])
        procs.append(p)
        return p

    try:
        yield _spawn
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                p.kill()
                p.wait(timeout=5)


def _wait_dead(proc: subprocess.Popen, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.05)
    return proc.poll() is not None


def _task_row(conn, tid):
    return conn.execute(
        "SELECT status, claim_lock, claim_expires, worker_pid, current_run_id, "
        "consecutive_failures FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()


def _held(row) -> bool:
    """A task is held (not claimable) while it carries a claim_lock."""
    return row["claim_lock"] is not None


def _noop_signal(pid, sig):
    """Swallow the kill so the test keeps owning its own process."""


def _reap(proc) -> None:
    """Kill and REAP the test-owned worker (a zombie still reads as alive)."""
    proc.kill()
    proc.wait(timeout=5)


def _running_task_with_worker(conn, sleeper, *, title="terminal", assignee="impl"):
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title=title, assignee=assignee)
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    kb._set_worker_pid(conn, tid, sleeper.pid)
    return tid


def _expire_hold(conn, tid):
    """Age the hold's TTL past now, as the dispatcher would observe it."""
    row = _task_row(conn, tid)
    assert _held(row), "expected a live hold to expire"
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (int(time.time()) - 1, tid),
        )


# --- A. terminal transitions must not orphan a live worker -----------------


def test_complete_leaves_no_released_task_with_live_worker(conn, owned_sleeper):
    sleeper = owned_sleeper()
    tid = _running_task_with_worker(conn, sleeper, title="complete")

    assert kb.complete_task(conn, tid, summary="done by someone else") is True

    row = _task_row(conn, tid)
    assert row["status"] == "done"
    worker_dead = _wait_dead(sleeper)
    assert worker_dead or _held(row), (
        "complete_task released the task while the previous writer was alive"
    )


@pytest.mark.parametrize(
    "kind,landing",
    [(None, "blocked"), ("dependency", "todo"), ("needs_input", "blocked")],
)
def test_block_leaves_no_released_task_with_live_worker(
    conn, owned_sleeper, kind, landing,
):
    """Every block branch, including the immediately-claimable ``todo`` one."""
    sleeper = owned_sleeper()
    tid = _running_task_with_worker(conn, sleeper, title=f"block-{kind}")

    assert kb.block_task(conn, tid, reason="stuck", kind=kind) is True

    row = _task_row(conn, tid)
    assert row["status"] == landing
    worker_dead = _wait_dead(sleeper)
    assert worker_dead or _held(row), (
        f"block_task(kind={kind!r}) released the task while the previous "
        f"writer was alive"
    )


def test_surviving_worker_keeps_a_blocked_dependency_card_unclaimable(
    conn, owned_sleeper,
):
    """``todo`` is the sharp landing: it is claimable on the next tick."""
    signalled: list[tuple[int, int]] = []

    def _record(pid, sig):
        signalled.append((int(pid), int(sig)))

    sleeper = owned_sleeper()
    tid = _running_task_with_worker(conn, sleeper, title="survivor-dep")

    assert kb.block_task(
        conn, tid, reason="waiting on t_x", kind="dependency", signal_fn=_record,
    ) is True

    assert signalled, "worker was never signalled"
    assert sleeper.poll() is None, "test-owned worker should still be alive"
    row = _task_row(conn, tid)
    assert row["status"] == "todo"
    assert _held(row), "a claimable todo card was released beside a live writer"
    assert kb.claim_task(conn, tid) is None, (
        "a second writer was claimed into the workspace of a live worker"
    )


def test_surviving_worker_holds_a_completed_card(conn, owned_sleeper):
    signalled: list[tuple[int, int]] = []

    def _record(pid, sig):
        signalled.append((int(pid), int(sig)))

    sleeper = owned_sleeper()
    tid = _running_task_with_worker(conn, sleeper, title="survivor-done")

    assert kb.complete_task(conn, tid, summary="s", signal_fn=_record) is True

    assert signalled, "worker was never signalled"
    assert sleeper.poll() is None
    row = _task_row(conn, tid)
    assert row["status"] == "done"
    assert _held(row), "completion released the card beside a live writer"
    kinds = {
        r["kind"]
        for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ?", (tid,)
        ).fetchall()
    }
    assert "reclaim_deferred" in kinds


def test_self_terminal_transition_does_not_kill_the_calling_worker(
    conn, owned_sleeper,
):
    """The load-bearing guard: a worker completing/blocking its OWN run."""
    signalled: list[int] = []

    def _record(pid, sig):  # pragma: no cover - must never fire
        signalled.append(int(pid))

    complete_sleeper = owned_sleeper()
    complete_tid = _running_task_with_worker(
        conn, complete_sleeper, title="self-complete",
    )
    run_id = int(_task_row(conn, complete_tid)["current_run_id"])
    assert kb.complete_task(
        conn, complete_tid, summary="self", expected_run_id=run_id,
        signal_fn=_record,
    ) is True
    assert signalled == [], "a worker completing its own run was signalled"
    assert complete_sleeper.poll() is None, "the calling worker was killed"
    complete_row = _task_row(conn, complete_tid)
    assert complete_row["status"] == "done"
    assert complete_row["claim_lock"] is None

    block_sleeper = owned_sleeper()
    block_tid = _running_task_with_worker(conn, block_sleeper, title="self-block")
    block_run = int(_task_row(conn, block_tid)["current_run_id"])
    assert kb.block_task(
        conn, block_tid, reason="need input", kind="needs_input",
        expected_run_id=block_run, signal_fn=_record,
    ) is True
    assert signalled == [], "a worker blocking its own run was signalled"
    assert block_sleeper.poll() is None
    block_row = _task_row(conn, block_tid)
    assert block_row["status"] == "blocked"
    assert block_row["claim_lock"] is None


def test_terminal_holds_self_heal_once_the_worker_dies(conn, owned_sleeper):
    """A hold on a landed card must not become a permanent dispatch stall."""
    dep_sleeper = owned_sleeper()
    dep_tid = _running_task_with_worker(conn, dep_sleeper, title="heal-dep")
    assert kb.block_task(
        conn, dep_tid, reason="dep", kind="dependency", signal_fn=_noop_signal,
    ) is True
    assert _held(_task_row(conn, dep_tid)), "expected the survivor hold"

    done_sleeper = owned_sleeper()
    done_tid = _running_task_with_worker(conn, done_sleeper, title="heal-done")
    assert kb.complete_task(
        conn, done_tid, summary="s", signal_fn=_noop_signal,
    ) is True
    assert _held(_task_row(conn, done_tid)), "expected the survivor hold"

    landings = {
        _task_row(conn, dep_tid)["status"],
        _task_row(conn, done_tid)["status"],
    }
    assert "running" not in landings, (
        "a landed terminal status collided with the main running sweep"
    )

    for tid, sleeper in ((dep_tid, dep_sleeper), (done_tid, done_sleeper)):
        _reap(sleeper)
        _expire_hold(conn, tid)

    kb.release_stale_claims(conn, signal_fn=_noop_signal)

    dep_row = _task_row(conn, dep_tid)
    # 'ready' is allowed: once the hold clears, recompute_ready legitimately
    # promotes a parentless todo. What must NOT happen is the card landing
    # anywhere outside its dispatchable lane.
    assert dep_row["status"] in ("todo", "ready"), (
        "the sweep moved the card out of its dispatchable lane"
    )
    assert not _held(dep_row), "the hold outlived its worker — card is stuck"
    assert kb.claim_task(conn, dep_tid) is not None, (
        "card stayed unclaimable after its held worker died"
    )

    done_row = _task_row(conn, done_tid)
    assert done_row["status"] == "done", "the sweep must not move the card"
    assert not _held(done_row), "the hold outlived its worker — card is stuck"
    assert done_row["worker_pid"] is None


def test_terminal_hold_survives_the_sweep_while_the_worker_is_alive(
    conn, owned_sleeper,
):
    """The sweep must never hand a live writer's card to a second worker."""
    sleeper = owned_sleeper()
    tid = _running_task_with_worker(conn, sleeper, title="heal-negative")
    assert kb.block_task(
        conn, tid, reason="dep", kind="dependency", signal_fn=_noop_signal,
    ) is True
    assert _held(_task_row(conn, tid))

    _expire_hold(conn, tid)
    kb.release_stale_claims(conn, signal_fn=_noop_signal)

    assert sleeper.poll() is None, "test-owned worker should still be alive"
    row = _task_row(conn, tid)
    assert _held(row), "the sweep released a hold beside a live writer"
    assert row["status"] == "todo"
    assert kb.claim_task(conn, tid) is None
    assert int(row["claim_expires"]) > int(time.time())


# --- B. the terminal-reopen / rework route ---------------------------------


def test_done_card_has_a_supported_route_back_to_rework(conn, owned_sleeper):
    """The premise of part B: the pre-existing verbs refuse, reopen_done works."""
    sleeper = owned_sleeper()
    tid = _running_task_with_worker(conn, sleeper, title="reopen")
    run_id = int(_task_row(conn, tid)["current_run_id"])
    assert kb.complete_task(
        conn, tid, summary="claimed done", expected_run_id=run_id,
    ) is True
    assert _task_row(conn, tid)["status"] == "done"

    # The routes that exist refuse a done card.
    assert kb.reopen_review_task(conn, tid) is False
    promoted, why = kb.promote_task(conn, tid, actor="operator")
    assert promoted is False and "done" in (why or "")

    ok, status = kb.reopen_done_task(
        conn, tid, actor="operator", reason="never pushed; no PR, no review",
    )
    assert ok is True
    row = _task_row(conn, tid)
    assert status == row["status"] == "ready"
    assert kb.claim_task(conn, tid) is not None, (
        "a reopened card must be dispatchable again"
    )


def test_reopen_done_refuses_a_card_that_is_not_terminal(conn, owned_sleeper):
    sleeper = owned_sleeper()
    tid = _running_task_with_worker(conn, sleeper, title="not-terminal")
    ok, why = kb.reopen_done_task(conn, tid, actor="operator")
    assert ok is False
    assert "running" in (why or "")
    assert _task_row(conn, tid)["status"] == "running"


def test_reopen_done_waits_in_todo_when_a_parent_is_not_done(conn):
    parent = kb.create_task(conn, title="parent", assignee="impl")
    child = kb.create_task(conn, title="child", assignee="impl", parents=[parent])
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
            (int(time.time()), child),
        )

    ok, status = kb.reopen_done_task(conn, child, actor="operator")
    assert ok is True
    assert status == "todo", (
        "reopening past an unfinished parent would bypass parent gating"
    )
    assert kb.claim_task(conn, child) is None


def test_reopen_done_leaves_no_released_task_with_live_worker(conn, owned_sleeper):
    """Terminate-before-release, on a landing that is immediately claimable."""
    signalled: list[tuple[int, int]] = []

    def _record(pid, sig):
        signalled.append((int(pid), int(sig)))

    sleeper = owned_sleeper()
    tid = _running_task_with_worker(conn, sleeper, title="reopen-live")
    # A done card carrying a live claim handle: exactly what the terminal-path
    # orphan bug produced before part A landed.
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
            (int(time.time()), tid),
        )
    assert _task_row(conn, tid)["worker_pid"] == sleeper.pid

    ok, _status = kb.reopen_done_task(
        conn, tid, actor="operator", signal_fn=_record,
    )
    assert ok is True
    assert signalled, "worker was never signalled"
    assert sleeper.poll() is None, "test-owned worker should still be alive"
    row = _task_row(conn, tid)
    assert _held(row), "reopen released a claimable card beside a live writer"
    assert kb.claim_task(conn, tid) is None, (
        "a second writer was claimed into the workspace of a live worker"
    )


def test_reopen_done_hold_self_heals(conn, owned_sleeper):
    sleeper = owned_sleeper()
    tid = _running_task_with_worker(conn, sleeper, title="reopen-heal")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
            (int(time.time()), tid),
        )
    ok, landed = kb.reopen_done_task(
        conn, tid, actor="operator", signal_fn=_noop_signal,
    )
    assert ok is True
    assert _held(_task_row(conn, tid)), "expected the survivor hold"

    _reap(sleeper)
    _expire_hold(conn, tid)
    kb.release_stale_claims(conn, signal_fn=_noop_signal)

    row = _task_row(conn, tid)
    assert row["status"] == landed, "the sweep must not move the card"
    assert not _held(row), "the reopen hold outlived its worker — card is stuck"
    assert kb.claim_task(conn, tid) is not None


def test_reopen_done_does_not_signal_the_calling_worker(conn, owned_sleeper):
    signalled: list[int] = []

    def _record(pid, sig):  # pragma: no cover - must never fire
        signalled.append(int(pid))

    sleeper = owned_sleeper()
    tid = _running_task_with_worker(conn, sleeper, title="reopen-self")
    run_id = int(_task_row(conn, tid)["current_run_id"])
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
            (int(time.time()), tid),
        )

    ok, _status = kb.reopen_done_task(
        conn, tid, actor="operator", expected_run_id=run_id, signal_fn=_record,
    )
    assert ok is True
    assert signalled == [], "a caller owning the current run was signalled"
    assert sleeper.poll() is None


def test_reopen_done_retracts_descendants_that_assumed_the_result(conn):
    parent = kb.create_task(conn, title="parent", assignee="impl")
    child = kb.create_task(conn, title="child", assignee="impl", parents=[parent])
    now = int(time.time())
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
            (now, parent),
        )
        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
            (now, child),
        )

    ok, _status = kb.reopen_done_task(conn, parent, actor="operator")
    assert ok is True
    child_row = _task_row(conn, child)
    assert child_row["status"] == "todo", (
        "a descendant built on a retracted premise stayed dispatchable/done"
    )
    kinds = {
        r["kind"]
        for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ?", (child,)
        ).fetchall()
    }
    assert "descendant_invalidated" in kinds


def test_reopen_done_resets_the_failure_breaker(conn):
    """Operator reset rule — the opposite of reopen_review_task's preserve."""
    tid = kb.create_task(conn, title="breaker", assignee="impl")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = ?, "
            "consecutive_failures = 2 WHERE id = ?",
            (int(time.time()), tid),
        )

    ok, _status = kb.reopen_done_task(conn, tid, actor="operator")
    assert ok is True
    assert _task_row(conn, tid)["consecutive_failures"] == 0
