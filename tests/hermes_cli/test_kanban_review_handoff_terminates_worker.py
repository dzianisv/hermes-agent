"""Tests: the review-handoff transitions must not orphan a live worker.

``request_review`` / ``request_changes`` used to NULL out
``claim_lock``/``claim_expires``/``worker_pid`` and end the run without ever
terminating the worker process. The row then held no handle on a process that
was still writing in the card's workspace, the task landed in review/ready, and
the next dispatcher tick claimed it and spawned a SECOND writer into the same
directory.

The invariant asserted here: after a handoff driven by someone other than the
current worker, there is never a released (claimable) task with a live previous
writer. Either the worker is dead, or the task is still held.

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
        "SELECT status, claim_lock, claim_expires, worker_pid, current_run_id "
        "FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()


def _held(row) -> bool:
    """A task is held (not claimable) while it carries a claim_lock."""
    return row["claim_lock"] is not None


def _running_task_with_worker(conn, sleeper, *, title="handoff"):
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title=title, assignee="impl")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    kb._set_worker_pid(conn, tid, sleeper.pid)
    return tid


def _into_review_run(conn, sleeper, *, title="changes"):
    """Drive a card to a live review run owned by ``sleeper``."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title=title, assignee="impl")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    run_id = _task_row(conn, tid)["current_run_id"]
    assert kb.request_review(
        conn, tid, reviewer="reviewer", expected_run_id=int(run_id)
    ) is True
    assert kb.claim_review_task(conn, tid, claimer=f"{host}:R") is not None
    kb._set_worker_pid(conn, tid, sleeper.pid)
    return tid


def test_request_review_leaves_no_released_task_with_live_worker(conn, owned_sleeper):
    sleeper = owned_sleeper()
    tid = _running_task_with_worker(conn, sleeper)

    assert kb.request_review(conn, tid, reviewer="reviewer", force=True) is True

    row = _task_row(conn, tid)
    worker_dead = _wait_dead(sleeper)
    assert worker_dead or _held(row), (
        "request_review released the task while the previous writer was alive"
    )


def test_request_changes_leaves_no_released_task_with_live_worker(conn, owned_sleeper):
    sleeper = owned_sleeper()
    tid = _into_review_run(conn, sleeper)

    ok, implementer = kb.request_changes(conn, tid, reason="needs rework")
    assert ok is True
    assert implementer == "impl"

    row = _task_row(conn, tid)
    worker_dead = _wait_dead(sleeper)
    assert worker_dead or _held(row), (
        "request_changes released the task while the previous writer was alive"
    )


def test_surviving_worker_blocks_the_next_claim(conn, owned_sleeper):
    """When the kill does not take, the card must stay unclaimable."""
    signalled: list[tuple[int, int]] = []

    def _noop_signal(pid, sig):
        signalled.append((int(pid), int(sig)))

    review_sleeper = owned_sleeper()
    review_tid = _running_task_with_worker(conn, review_sleeper, title="survivor-a")
    assert kb.request_review(
        conn, review_tid, reviewer="reviewer", force=True, signal_fn=_noop_signal,
    ) is True
    assert signalled, "worker was never signalled"
    assert review_sleeper.poll() is None, "test-owned worker should still be alive"
    assert _held(_task_row(conn, review_tid))
    assert kb.claim_review_task(conn, review_tid) is None, (
        "a second writer was claimed into the workspace of a live worker"
    )
    assert kb.claim_task(conn, review_tid) is None

    changes_sleeper = owned_sleeper()
    changes_tid = _into_review_run(conn, changes_sleeper, title="survivor-b")
    ok, _implementer = kb.request_changes(
        conn, changes_tid, reason="needs rework", signal_fn=_noop_signal,
    )
    assert ok is True
    assert changes_sleeper.poll() is None
    assert _held(_task_row(conn, changes_tid))
    assert kb.claim_task(conn, changes_tid) is None, (
        "a second writer was claimed into the workspace of a live worker"
    )

    kinds = {
        r["kind"]
        for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id IN (?, ?)",
            (review_tid, changes_tid),
        ).fetchall()
    }
    assert "reclaim_deferred" in kinds


def test_self_transition_does_not_kill_the_calling_worker(conn, owned_sleeper):
    """A worker handing off its OWN run must not be signalled."""
    signalled: list[int] = []

    def _record_signal(pid, sig):  # pragma: no cover - must never fire
        signalled.append(int(pid))

    sleeper = owned_sleeper()
    tid = _running_task_with_worker(conn, sleeper, title="self")
    run_id = int(_task_row(conn, tid)["current_run_id"])

    assert kb.request_review(
        conn, tid, reviewer="reviewer", expected_run_id=run_id,
        signal_fn=_record_signal,
    ) is True
    assert signalled == [], "a worker transitioning its own run was signalled"
    assert sleeper.poll() is None, "the calling worker's process was killed"

    row = _task_row(conn, tid)
    assert row["status"] == "review"
    assert row["claim_lock"] is None

    # Same for request_changes: the reviewer ends its own run.
    reviewer_sleeper = owned_sleeper()
    host = kb._claimer_id().split(":", 1)[0]
    assert kb.claim_review_task(conn, tid, claimer=f"{host}:R") is not None
    kb._set_worker_pid(conn, tid, reviewer_sleeper.pid)
    review_run = int(_task_row(conn, tid)["current_run_id"])

    ok, implementer = kb.request_changes(
        conn, tid, reason="rework", expected_run_id=review_run,
        signal_fn=_record_signal,
    )
    assert (ok, implementer) == (True, "impl")
    assert signalled == []
    assert reviewer_sleeper.poll() is None


# --- the hold must self-heal once the survivor finally dies -----------------
#
# The hold parks a claim on a card that has already LANDED (review / ready),
# which no status='running' recovery path can see. If nothing ever clears it,
# the card is permanently unclaimable once the held worker dies: a silent
# dispatch stall. ``release_stale_claims`` — the sweeper the dispatcher
# already runs every tick — is what heals it.


def _expire_hold(conn, tid):
    """Age the hold's TTL past now, as the dispatcher would observe it."""
    row = _task_row(conn, tid)
    assert _held(row), "expected a live hold to expire"
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (int(time.time()) - 1, tid),
        )


def _reap(proc) -> None:
    """Kill and REAP the test-owned worker (a zombie still reads as alive)."""
    proc.kill()
    proc.wait(timeout=5)


def _noop_signal(pid, sig):
    """Swallow the kill so the test keeps owning its own process."""


def test_review_hold_is_released_once_the_worker_dies(conn, owned_sleeper):
    sleeper = owned_sleeper()
    tid = _running_task_with_worker(conn, sleeper, title="heal-review")
    assert kb.request_review(
        conn, tid, reviewer="reviewer", force=True, signal_fn=_noop_signal,
    ) is True
    assert _held(_task_row(conn, tid)), "expected the survivor hold"

    _reap(sleeper)
    _expire_hold(conn, tid)

    kb.release_stale_claims(conn, signal_fn=_noop_signal)

    row = _task_row(conn, tid)
    assert row["status"] == "review", "the sweep must not move the card"
    assert not _held(row), "the hold outlived its worker — card is stuck"
    assert row["worker_pid"] is None
    assert kb.claim_review_task(conn, tid) is not None, (
        "card stayed unclaimable after its held worker died"
    )


def test_changes_hold_is_released_once_the_worker_dies(conn, owned_sleeper):
    sleeper = owned_sleeper()
    tid = _into_review_run(conn, sleeper, title="heal-changes")
    ok, _implementer = kb.request_changes(
        conn, tid, reason="needs rework", signal_fn=_noop_signal,
    )
    assert ok is True
    held_row = _task_row(conn, tid)
    assert _held(held_row), "expected the survivor hold"
    landed_status = held_row["status"]
    assert landed_status == "ready"

    _reap(sleeper)
    _expire_hold(conn, tid)

    kb.release_stale_claims(conn, signal_fn=_noop_signal)

    row = _task_row(conn, tid)
    assert row["status"] == landed_status, "the sweep must not move the card"
    assert not _held(row), "the hold outlived its worker — card is stuck"
    assert kb.claim_task(conn, tid) is not None, (
        "card stayed unclaimable after its held worker died"
    )


def test_hold_survives_the_sweep_while_the_worker_is_alive(conn, owned_sleeper):
    """The sweep must never hand a live writer's card to a second worker."""
    sleeper = owned_sleeper()
    tid = _running_task_with_worker(conn, sleeper, title="heal-negative")
    assert kb.request_review(
        conn, tid, reviewer="reviewer", force=True, signal_fn=_noop_signal,
    ) is True
    assert _held(_task_row(conn, tid))

    _expire_hold(conn, tid)
    kb.release_stale_claims(conn, signal_fn=_noop_signal)

    assert sleeper.poll() is None, "test-owned worker should still be alive"
    row = _task_row(conn, tid)
    assert _held(row), "the sweep released a hold beside a live writer"
    assert row["status"] == "review"
    assert kb.claim_review_task(conn, tid) is None
    assert kb.claim_task(conn, tid) is None
    # The hold was pushed forward, not left expired, so the next tick still
    # sees a held card rather than a claimable one.
    assert int(row["claim_expires"]) > int(time.time())


def test_swept_hold_statuses_cover_where_the_handoffs_actually_land(
    conn, owned_sleeper,
):
    """Both handoff landings are shapes the sweep can see.

    The sweep matches on hold SHAPE (non-running card + host-local claim +
    expired TTL), so this ties that shape to the statuses the two transitions
    really produce instead of trusting a hand-written status list.
    """
    review_sleeper = owned_sleeper()
    review_tid = _running_task_with_worker(conn, review_sleeper, title="land-a")
    assert kb.request_review(
        conn, review_tid, reviewer="reviewer", force=True, signal_fn=_noop_signal,
    ) is True

    changes_sleeper = owned_sleeper()
    changes_tid = _into_review_run(conn, changes_sleeper, title="land-b")
    ok, _impl = kb.request_changes(
        conn, changes_tid, reason="rework", signal_fn=_noop_signal,
    )
    assert ok is True

    landings = {
        _task_row(conn, review_tid)["status"],
        _task_row(conn, changes_tid)["status"],
    }
    assert "running" not in landings, (
        "a landed handoff status collided with the main running sweep"
    )

    for tid, sleeper in ((review_tid, review_sleeper), (changes_tid, changes_sleeper)):
        _reap(sleeper)
        _expire_hold(conn, tid)

    kb.release_stale_claims(conn, signal_fn=_noop_signal)

    for tid in (review_tid, changes_tid):
        row = _task_row(conn, tid)
        assert not _held(row), f"{row['status']} landing was not swept"

    kinds = {
        r["kind"]
        for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id IN (?, ?)",
            (review_tid, changes_tid),
        ).fetchall()
    }
    assert "claim_hold_released" in kinds
