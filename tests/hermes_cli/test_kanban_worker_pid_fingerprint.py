"""A recycled worker PID is never mistaken for our worker.

``tasks.worker_pid`` survives a reboot; the number can then belong to an unrelated process. Every
liveness decision (extend/defer the claim) and every kill (SIGTERM/SIGKILL on timeout or reclaim)
must require the spawn-time start fingerprint to match, never bare PID existence.
"""

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


def _claimed_running(conn, *, pid: int, started_at, max_runtime=None) -> str:
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=max_runtime,
                         allow_duplicate=True)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = ?, started_at = ?, claim_expires = ? WHERE id = ?",
                     (started_at, old, old, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))
    return tid


def test_recycled_pid_is_reclaimed_without_being_signalled(board):
    """Our own live PID with a foreign fingerprint models a post-reboot recycle: the claim is released
    (dead worker), no signal is sent, and max-runtime enforcement does not SIGTERM the stranger either."""
    conn = board
    killed = []
    stranger_fingerprint = 1  # no live process started at tick 1
    tid = _claimed_running(conn, pid=os.getpid(), started_at=stranger_fingerprint, max_runtime=1)

    assert kbd._worker_alive(os.getpid(), stranger_fingerprint) is False
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.worker_pid is None

    tid2 = _claimed_running(conn, pid=os.getpid(), started_at=stranger_fingerprint)
    assert kb.release_stale_claims(conn, signal_fn=lambda pid, sig: killed.append((pid, sig))) == 1
    assert killed == []
    assert kb.get_task(conn, tid2).status == "ready"


def test_matching_fingerprint_keeps_the_live_worker(board):
    """The same PID with ITS OWN fingerprint (recorded at spawn) is our worker: the expired claim is
    extended rather than reclaimed, and the timeout path signals it."""
    from gateway.status import get_process_start_time

    conn = board
    killed = []
    tid = _claimed_running(conn, pid=os.getpid(), started_at=get_process_start_time(os.getpid()))
    assert kbd._worker_alive(os.getpid(), get_process_start_time(os.getpid())) is True
    assert kb.release_stale_claims(conn) == 0
    assert kb.get_task(conn, tid).status == "running"
    kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert "claim_extended" in kinds

    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET max_runtime_seconds = 1 WHERE id = ?", (tid,))
    kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed and killed[0] == (os.getpid(), signal.SIGTERM)


def test_same_pid_and_start_tick_on_another_boot_is_foreign(board, monkeypatch):
    """A row that survived a reboot: the PID AND the boot-relative start tick both match a process on
    this boot (the Linux start time is clock ticks since boot, so that recurs), but the persisted
    instantiation epoch does not. The worker is foreign: claim released, zero signals."""
    from gateway import drain_control

    conn = board
    killed = []
    live_fingerprint = kbd._process_fingerprint(os.getpid())
    assert live_fingerprint is not None and live_fingerprint.split("|", 1)[1] == str(
        __import__("gateway.status", fromlist=["x"]).get_process_start_time(os.getpid()))
    tid = _claimed_running(conn, pid=os.getpid(), started_at=live_fingerprint, max_runtime=1)
    assert kbd._worker_alive(os.getpid(), live_fingerprint) is True

    # Same PID, same start tick, different boot identity.
    other_boot = "deadbeef-boot:1|" + live_fingerprint.split("|", 1)[1]
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = ? WHERE id = ?", (other_boot, tid))
    assert kbd._worker_alive(os.getpid(), other_boot) is False
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.worker_pid is None

    # The same value re-derived on THIS boot still identifies our worker (the witness is stable
    # within a boot, unlike the recorded epoch of a previous one).
    drain_control.current_instantiation_epoch.cache_clear()
    assert kbd._process_fingerprint(os.getpid()) == live_fingerprint


def _drift_start_time(monkeypatch, delta: int):
    """Later ``get_process_start_time`` reads return the real value plus ``delta`` (same scale)."""
    from gateway import status

    real = status.get_process_start_time

    def drifted(pid: int):
        start = real(pid)
        return None if start is None else start + delta

    monkeypatch.setattr(status, "get_process_start_time", drifted)
    return real


def test_one_second_start_drift_is_still_our_worker(monkeypatch):
    """macOS start-time readings drift ~1s / 100 units between spawn and a later read (#117505).

    Observed: recorded ``|179088627896``, current ``|179088627796``. Exact equality on the composed
    fingerprint declared the live worker dead.
    """
    pid = os.getpid()
    recorded = kbd._process_fingerprint(pid)
    assert recorded is not None and "|" in recorded
    _drift_start_time(monkeypatch, -100)
    assert kbd._pid_recycled(pid, recorded) is False
    assert kbd._worker_alive(pid, recorded) is True


def test_large_start_drift_is_recycled(monkeypatch):
    pid = os.getpid()
    recorded = kbd._process_fingerprint(pid)
    assert recorded is not None
    _drift_start_time(monkeypatch, 10000)
    assert kbd._pid_recycled(pid, recorded) is True
    assert kbd._worker_alive(pid, recorded) is False


def test_different_epoch_is_recycled_even_when_start_matches(monkeypatch):
    from gateway import drain_control

    pid = os.getpid()
    recorded = kbd._process_fingerprint(pid)
    assert recorded is not None
    monkeypatch.setattr(drain_control, "current_instantiation_epoch", lambda: "other-boot")
    assert kbd._pid_recycled(pid, recorded) is True
    assert kbd._worker_alive(pid, recorded) is False


def test_unreadable_or_junk_fingerprint_is_recycled(monkeypatch):
    from gateway import status

    pid = os.getpid()
    recorded = kbd._process_fingerprint(pid)
    assert recorded is not None
    assert kbd._pid_recycled(pid, recorded.rsplit("|", 1)[0] + "|not-a-number") is True
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: None)
    assert kbd._pid_recycled(pid, recorded) is True


def test_non_positive_composed_start_parts_are_not_the_same_worker(monkeypatch):
    """Drift tolerance must not run until both composed start parts are strictly positive.

    ``('0','0')`` and ``('-1','-1')`` compare equal under a bare abs-diff, which would keep a
    junk fingerprint as "our worker". Empty and non-numeric are the same refusal.
    """
    from gateway import status

    pid = os.getpid()
    live = kbd._process_fingerprint(pid)
    assert live is not None and "|" in live
    epoch = live.rsplit("|", 1)[0]
    for junk in ("0", "-1", "", "not-a-number"):
        assert kbd._start_parts_match(junk, junk) is False
        monkeypatch.setattr(status, "get_process_start_time", lambda _pid, value=junk: value)
        assert kbd._pid_recycled(pid, f"{epoch}|{junk}") is True


def test_legacy_integer_fingerprint_tolerates_drift_but_not_junk(monkeypatch):
    """Pre-boot-witness rows store the ×100 start time only. Same tolerance as the composed start
    part; a non-positive reading stays foreign so the kill guard does not signal junk."""
    from gateway import status

    pid = os.getpid()
    recorded = status.get_process_start_time(pid)
    assert recorded is not None and int(recorded) > 0
    _drift_start_time(monkeypatch, -100)
    assert kbd._pid_recycled(pid, recorded) is False
    monkeypatch.undo()
    _drift_start_time(monkeypatch, 10000)
    assert kbd._pid_recycled(pid, recorded) is True
    monkeypatch.undo()
    monkeypatch.setattr(status, "get_process_start_time", lambda _pid: 0)
    assert kbd._pid_recycled(pid, recorded) is True
    assert kbd._pid_recycled(pid, 0) is True


def test_start_drift_does_not_crash_reclaim_a_live_worker(board, monkeypatch):
    """Dispatcher crash sweep must not emit ``pid N not alive`` or release a live worker whose
    start-time reading drifted by 100 — that is the respawn-a-duplicate path."""
    conn = board
    pid = os.getpid()
    recorded = kbd._process_fingerprint(pid)
    assert recorded is not None
    tid = _claimed_running(conn, pid=pid, started_at=recorded)
    _drift_start_time(monkeypatch, -100)

    assert kbd.detect_crashed_workers(conn) == []
    task = kb.get_task(conn, tid)
    assert task.status == "running"
    assert task.worker_pid == pid
    assert "crashed" not in [e.kind for e in kb.list_events(conn, tid)]
    row = conn.execute("SELECT worker_started_at, status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "running" and row["worker_started_at"] == recorded


def test_unverified_fingerprint_capture_never_authorizes_a_signal(board, monkeypatch):
    """Fingerprint capture fails for a new spawn: the row is NOT a legacy NULL row. A live PID under
    it is never SIGTERM/SIGKILLed by any reclaim/timeout path, and the claim is held (not released
    beside the live process); once the PID is gone the claim is reclaimed normally."""
    import gateway.status as status

    conn = board
    killed = []
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=1)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    row = conn.execute("SELECT worker_started_at FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["worker_started_at"] == kbd.UNVERIFIED_WORKER_FINGERPRINT
    monkeypatch.undo()
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ?, claim_expires = ? WHERE id = ?", (old, old, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))

    sig = lambda pid, s: killed.append((pid, s))  # noqa: E731
    assert kbd.enforce_max_runtime(conn, signal_fn=sig) == []
    assert kb.release_stale_claims(conn, signal_fn=sig) == 0
    assert killed == []
    assert kb.get_task(conn, tid).status == "running"
    # An explicit operator reclaim releases the claim (human override) but still sends nothing.
    assert kb.reclaim_task(conn, tid, reason="operator", signal_fn=sig) is True
    assert killed == []

    # The process is gone (a dead PID): the row is reclaimed like any dead worker, still no signal.
    tid2 = kb.create_task(conn, title="job2", assignee="worker")
    kb.claim_task(conn, tid2)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_pid = ?, worker_started_at = ?, claim_expires = ? WHERE id = ?",
                     (os.getpid(), kbd.UNVERIFIED_WORKER_FINGERPRINT, old, tid2))
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    assert kb.release_stale_claims(conn, signal_fn=sig) == 1
    assert killed == [] and kb.get_task(conn, tid2).status == "ready"


def _stop_children(procs) -> None:
    for proc in procs:
        if proc.poll() is not None:
            continue
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)


def _counting_live_spawn(calls: list, procs: list):
    """spawn_fn that starts a real child and returns its pid. Counts every call."""
    def spawn_fn(task, workspace, board=None):
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            start_new_session=True,
        )
        procs.append(proc)
        calls.append(task.id)
        return proc.pid
    return spawn_fn


def _dispatch_live_worker(conn, monkeypatch, tmp_path):
    """One manual tick that spawns a real child and records its fingerprint."""
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    calls: list = []
    procs: list = []
    tid = kb.create_task(conn, title="drift", assignee="worker")
    first = kbd.dispatch_once(conn, spawn_fn=_counting_live_spawn(calls, procs))
    assert calls == [tid], first
    assert first.crashed == []
    task = kb.get_task(conn, tid)
    assert task.status == "running" and task.worker_pid == procs[0].pid
    assert procs[0].poll() is None
    recorded = conn.execute(
        "SELECT worker_started_at FROM tasks WHERE id = ?", (tid,),
    ).fetchone()["worker_started_at"]
    assert recorded and "|" in str(recorded) and recorded != kbd.UNVERIFIED_WORKER_FINGERPRINT
    return tid, calls, procs, recorded


def test_dispatch_once_keeps_live_worker_across_macos_start_drift(
    board, all_assignees_spawnable, monkeypatch, tmp_path,
):
    """A later manual tick must not crash/respawn a live child whose start time drifted by 100.

    That is the observed macOS gap (#117505). ``dispatch_once`` reads ``get_process_start_time``
    via the late import in the dispatch module, so the patch lands on that name.
    """
    from gateway.status import START_TIME_DRIFT_TOLERANCE

    conn = board
    tid, calls, procs, recorded = _dispatch_live_worker(conn, monkeypatch, tmp_path)
    try:
        assert 100 <= START_TIME_DRIFT_TOLERANCE
        _drift_start_time(monkeypatch, -100)
        second = kbd.dispatch_once(conn, spawn_fn=_counting_live_spawn(calls, procs))
        assert calls == [tid]
        assert second.crashed == []
        task = kb.get_task(conn, tid)
        assert task.status == "running" and task.worker_pid == procs[0].pid
        assert procs[0].poll() is None
        assert "crashed" not in [e.kind for e in kb.list_events(conn, tid)]
        still = conn.execute(
            "SELECT worker_started_at, status FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert still["status"] == "running" and still["worker_started_at"] == recorded
    finally:
        _stop_children(procs)


def test_dispatch_once_far_start_drift_detects_recycled_pid(
    board, all_assignees_spawnable, monkeypatch, tmp_path,
):
    """The same tick path must still crash a live pid whose start time is far outside tolerance."""
    from gateway.status import START_TIME_DRIFT_TOLERANCE

    conn = board
    tid, calls, procs, _recorded = _dispatch_live_worker(conn, monkeypatch, tmp_path)
    try:
        _drift_start_time(monkeypatch, START_TIME_DRIFT_TOLERANCE + 10_000)
        second = kbd.dispatch_once(conn, spawn_fn=_counting_live_spawn(calls, procs))
        assert tid in second.crashed
        assert "crashed" in [e.kind for e in kb.list_events(conn, tid)]
    finally:
        _stop_children(procs)
