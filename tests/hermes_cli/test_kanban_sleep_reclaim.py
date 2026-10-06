"""#5: host sleep must not make a live worker's heartbeat look stale.

Wall clock advances during sleep, ``time.monotonic`` does not. A live
host-local worker whose heartbeat is only "old" because the Mac slept must
get its claim extended (same session resumes), not reclaimed + killed.
"""
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(kb, "_clock_sample", None, raising=False)
    monkeypatch.setattr(kb, "_sleep_intervals", [], raising=False)
    kb.init_db()
    return tmp_path


def _running_task(conn, monkeypatch, wall):
    tid = kb.create_task(conn, title="t", assignee="worker")
    kb.claim_task(conn, tid)
    conn.execute(
        "UPDATE tasks SET worker_pid = 4242, last_heartbeat_at = ?, claim_expires = ? WHERE id = ?",
        (int(wall), int(wall) + 900, tid),
    )
    conn.commit()
    monkeypatch.setattr(kb, "_worker_alive", lambda pid, started_at=None: True)
    return tid


def _tick(monkeypatch, conn, wall, mono, killed):
    monkeypatch.setattr(kb.time, "time", lambda: wall)
    monkeypatch.setattr(kb.time, "monotonic", lambda: mono)
    return kb.release_stale_claims(conn, signal_fn=lambda *a: killed.append(a))


def test_wall_clock_jump_from_sleep_extends_live_worker(kanban_home, monkeypatch):
    conn = kbc.connect()
    try:
        t0 = time.time()
        tid = _running_task(conn, monkeypatch, t0)
        killed = []
        assert _tick(monkeypatch, conn, t0 + 30, 1000.0, killed) == 0
        # 3h of sleep: wall jumps 10800s, monotonic only 60s.
        assert _tick(monkeypatch, conn, t0 + 30 + 10800, 1060.0, killed) == 0
        assert killed == []
        row = conn.execute("SELECT status, worker_pid FROM tasks WHERE id=?", (tid,)).fetchone()
        assert row["status"] == "running" and row["worker_pid"] == 4242
        kinds = [r[0] for r in conn.execute("SELECT kind FROM task_events WHERE task_id=?", (tid,))]
        assert "claim_extended" in kinds and "reclaimed" not in kinds
    finally:
        conn.close()


def test_real_silence_without_sleep_still_reclaims(kanban_home, monkeypatch):
    conn = kbc.connect()
    try:
        t0 = time.time()
        tid = _running_task(conn, monkeypatch, t0)
        monkeypatch.setattr(kb, "_terminate_reclaimed_worker", lambda *a, **k: {"terminated": True})
        killed = []
        assert _tick(monkeypatch, conn, t0 + 30, 1000.0, killed) == 0
        # 3h awake with no heartbeat: monotonic advances too -> wedged, reclaim.
        assert _tick(monkeypatch, conn, t0 + 30 + 10800, 1000.0 + 10800, killed) == 1
        assert conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()[0] != "running"
    finally:
        conn.close()
