"""Every worker exit leaves a structured ``exit_reason`` on its run, and the dispatcher acts
on the class: transient deaths requeue without spending the failure budget, two
consecutive harness exits (rc=0 without a handoff, non-zero rc) block the card with a
``BLOCKER:HARNESS`` comment instead of looping.

Real kanban DB under a temp HERMES_HOME; real worker log files; the only fakes are the
PID liveness probe and the reap registry entry a real ``waitpid`` would have produced.
"""

from __future__ import annotations

import signal
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_worker_exit as kwe

KEYS = {"reason", "rc", "signal", "stderr_tail", "handoff_called"}


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kbd, "_worker_alive", lambda *_a: False)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return h


def _running(conn, tid, pid, *, log: str = "", started_ago: int = 3600):
    """Claim ``tid`` as a real dispatcher would and plant a worker pid + log."""
    assert kb.claim_task(conn, tid) is not None
    past = int(time.time()) - started_ago
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_pid = ?, started_at = ? WHERE id = ?", (pid, past, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = "
                     "(SELECT current_run_id FROM tasks WHERE id = ?)", (past, tid))
    path = kb.worker_log_path(tid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(log)


def _exit(pid, *, rc=None, sig=None):
    kbd._record_worker_exit(pid, (rc << 8) if rc is not None else sig)


def _last_run(conn, tid):
    return kb.list_runs(conn, tid)[-1]


def _reason(conn, tid):
    er = _last_run(conn, tid).metadata["exit_reason"]
    assert set(er) == KEYS
    return er


def test_rc0_without_handoff_is_classified_with_rc_and_stderr_tail(home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _running(conn, tid, 81001, log="x" * 5000 + "\nTraceback: final words\n")
        _exit(81001, rc=0)
        kbd.detect_crashed_workers(conn)
        er = _reason(conn, tid)
    assert er["reason"] == "rc0_no_handoff"
    assert er["rc"] == 0 and er["signal"] is None
    assert er["handoff_called"] is False
    assert er["stderr_tail"].endswith("final words\n")
    assert len(er["stderr_tail"].encode()) <= 2048


def test_two_rc0_no_handoff_in_a_row_block_with_harness_comment(home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _running(conn, tid, 82001)
        _exit(82001, rc=0)
        kbd.detect_crashed_workers(conn)
        assert kb.get_task(conn, tid).status == "ready"

        _running(conn, tid, 82002)
        _exit(82002, rc=0)
        assert tid in kbd.detect_crashed_workers(conn)
        assert tid in kbd.detect_crashed_workers._last_auto_blocked
        task = kb.get_task(conn, tid)
        comments = [c.body for c in kb.list_comments(conn, tid)]
    assert task.status == "blocked"
    assert any(c.startswith("BLOCKER:HARNESS rc0_no_handoff") for c in comments)


def test_two_nonzero_rc_in_a_row_block_with_harness_comment(home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        for pid in (83001, 83002):
            _running(conn, tid, pid, log="ImportError: boom\n")
            _exit(pid, rc=1)
            kbd.detect_crashed_workers(conn)
            assert _reason(conn, tid)["rc"] == 1
        task = kb.get_task(conn, tid)
        comments = [c.body for c in kb.list_comments(conn, tid)]
    assert task.status == "blocked"
    assert any(c.startswith("BLOCKER:HARNESS rc_nonzero") for c in comments)


def test_streak_is_per_class_and_any_other_exit_breaks_it(home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        # rc0 / kill / rc0 / rc1: no two consecutive exits share a harness class.
        for pid, kw in ((84001, {"rc": 0}), (84002, {"sig": int(signal.SIGKILL)}),
                        (84003, {"rc": 0}), (84005, {"rc": 1})):
            _running(conn, tid, pid)
            _exit(pid, **kw)
            kbd.detect_crashed_workers(conn)
            assert kb.get_task(conn, tid).status == "ready", pid
        _running(conn, tid, 84006)
        _exit(84006, rc=1)
        kbd.detect_crashed_workers(conn)
        assert kb.get_task(conn, tid).status == "blocked"


@pytest.mark.parametrize("sig", [int(signal.SIGKILL), int(signal.SIGSEGV)])
def test_signal_killed_requeues_without_spending_failure_budget(home, sig):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        for i in range(4):
            _running(conn, tid, 85000 + i)
            _exit(85000 + i, sig=sig)
            kbd.detect_crashed_workers(conn)
            er = _reason(conn, tid)
            assert er["reason"] == "signal_killed" and er["signal"] == sig and er["rc"] is None
        task = kb.get_task(conn, tid)
        # The run closes ``crashed`` so the resume path (RESUMABLE_RUN_OUTCOMES) re-enters it.
        assert _last_run(conn, tid).outcome in kbd.RESUMABLE_RUN_OUTCOMES
    assert task.status == "ready"
    assert task.consecutive_failures == 0


def test_wrapper_rc_128_plus_n_is_a_signal_kill(home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _running(conn, tid, 86001)
        _exit(86001, rc=137)
        kbd.detect_crashed_workers(conn)
        er = _reason(conn, tid)
        assert kb.get_task(conn, tid).consecutive_failures == 0
    assert er == {**er, "reason": "signal_killed", "rc": 137, "signal": 9}


def test_unrecorded_death_of_a_run_older_than_the_dispatcher_is_gateway_shutdown(home):
    """No reap record and no exit trailer: the worker never reached its exit epilogue. A run
    started before this dispatcher process came up lost its parent gateway."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _running(conn, tid, 87001, started_ago=int(time.time() - kwe._PROCESS_STARTED_AT) + 600)
        kbd.detect_crashed_workers(conn)
        er = _reason(conn, tid)
        task = kb.get_task(conn, tid)
    assert er["reason"] == "gateway_shutdown"
    assert er["rc"] is None and er["signal"] is None
    assert task.status == "ready" and task.consecutive_failures == 0


def test_transient_loop_is_eventually_flagged_not_looped_forever(home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        for i in range(kwe.TRANSIENT_STREAK_LIMIT):
            _running(conn, tid, 88000 + i)
            _exit(88000 + i, sig=int(signal.SIGKILL))
            kbd.detect_crashed_workers(conn)
        task = kb.get_task(conn, tid)
        comments = [c.body for c in kb.list_comments(conn, tid)]
    assert task.status == "blocked"
    assert any(c.startswith("BLOCKER:HARNESS signal_killed") for c in comments)


def test_timeout_records_timeout_reason(home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _running(conn, tid, 89001, log="still working\n")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET max_runtime_seconds = 1 WHERE id = ?", (tid,))
        assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda *_a, **_k: None)
        er = _reason(conn, tid)
    assert er["reason"] == "timeout"
    assert er["signal"] in (int(signal.SIGTERM), int(signal.SIGKILL))
    assert er["stderr_tail"] == "still working\n"


def test_manual_reclaim_records_reclaimed_reason(home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _running(conn, tid, 90001)
        assert kb.reclaim_task(conn, tid, reason="manual", signal_fn=lambda *_a, **_k: None)
        er = _reason(conn, tid)
    assert er["reason"] == "reclaimed"


def test_exit_reason_is_readable_from_a_read_only_connection(home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _running(conn, tid, 91001)
        _exit(91001, rc=0)
        kbd.detect_crashed_workers(conn)
    ro = sqlite3.connect(f"file:{kb.kanban_db_path()}?mode=ro", uri=True)
    try:
        meta = ro.execute("SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                          (tid,)).fetchone()[0]
    finally:
        ro.close()
    assert '"rc0_no_handoff"' in meta
