"""#5: host sleep must not make a live worker's heartbeat look stale.

Fixture is a real board event (t_ba46ae97): last heartbeat 1790584473, dispatcher
woke at 1790604784 (5.6h of lid-closed sleep) and reclaimed + SIGTERMed a live
worker as ``heartbeat_stale``; a fresh session was then spawned.
"""
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

REAL_HB = 1790584473
REAL_WAKE = 1790604784


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(kb, "_sleep_prev", None, raising=False)
    monkeypatch.setattr(kb, "_sleep_intervals", [], raising=False)
    kb.init_db()
    c = kbc.connect()
    yield c
    c.close()


def _running_task(conn, monkeypatch):
    tid = kb.create_task(conn, title="t", assignee="worker")
    kb.claim_task(conn, tid)
    conn.execute(
        "UPDATE tasks SET worker_pid = 78328, last_heartbeat_at = ?, claim_expires = ?, "
        "started_at = ? WHERE id = ?", (REAL_HB, REAL_HB + 900, REAL_HB - 600, tid))
    conn.execute("UPDATE task_runs SET started_at = ? WHERE task_id = ?", (REAL_HB - 600, tid))
    conn.commit()
    monkeypatch.setattr(kb, "_worker_alive", lambda pid, started_at=None: True)
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", lambda *a, **k: {"terminated": True})
    return tid


def _clock(monkeypatch, wall, slept_total):
    monkeypatch.setattr(kb.time, "time", lambda: wall)
    monkeypatch.setattr(kb, "_host_sleep_total", lambda: slept_total)


def _kinds(conn, tid):
    return [r[0] for r in conn.execute("SELECT kind FROM task_events WHERE task_id=?", (tid,))]


def test_real_sleep_reclaim_is_extended_not_reclaimed(conn, monkeypatch):
    tid = _running_task(conn, monkeypatch)
    _clock(monkeypatch, REAL_HB + 30, 0.0)          # last tick before lid closed
    assert kb.release_stale_claims(conn) == 0
    _clock(monkeypatch, REAL_WAKE, REAL_WAKE - REAL_HB - 60)  # woke: slept ~all of it
    assert kb.release_stale_claims(conn) == 0
    row = conn.execute("SELECT status, worker_pid, claim_expires FROM tasks WHERE id=?", (tid,)).fetchone()
    assert row["status"] == "running" and row["worker_pid"] == 78328
    assert row["claim_expires"] > REAL_WAKE           # lease renewed on wake
    assert "claim_extended" in _kinds(conn, tid) and "reclaimed" not in _kinds(conn, tid)


def test_detect_stale_running_credits_sleep(conn, monkeypatch):
    tid = _running_task(conn, monkeypatch)
    _clock(monkeypatch, REAL_HB + 30, 0.0)
    kb.release_stale_claims(conn)
    _clock(monkeypatch, REAL_WAKE, REAL_WAKE - REAL_HB - 60)
    assert kbd.detect_stale_running(conn, stale_timeout_seconds=3600) == []
    assert conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()[0] == "running"


def test_awake_silence_still_reclaims(conn, monkeypatch):
    tid = _running_task(conn, monkeypatch)
    _clock(monkeypatch, REAL_HB + 30, 0.0)
    kb.release_stale_claims(conn)
    _clock(monkeypatch, REAL_WAKE, 0.0)               # awake whole time: wedged
    assert kb.release_stale_claims(conn) == 1
    assert "reclaimed" in _kinds(conn, tid)


def test_wall_clock_step_without_sleep_counter_is_not_sleep(conn, monkeypatch):
    """An NTP wall-clock step must not be credited when the OS sleep counter is flat."""
    kb._note_sleep_sample(1000.0, 0.0)
    kb._note_sleep_sample(1000.0 + 7200, 0.0)
    assert kb._slept_seconds_since(0) == 0


def test_host_sleep_total_is_nonnegative_on_this_host():
    v = kb._host_sleep_total()
    assert v is None or v >= 0
