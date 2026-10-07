"""Structured exit reason for every Kanban worker exit, and what the dispatcher does with it.

Every run the dispatcher closes on a worker's behalf (dead pid, timeout, stale/expired
claim, orphan) carries ``metadata.exit_reason``::

    {"reason": <class>, "rc": int|None, "signal": int|None,
     "stderr_tail": str (<= 2 KB of the worker log), "handoff_called": bool}

Classes: ``signal_killed``, ``gateway_shutdown``, ``rc_nonzero``, ``rc0_no_handoff``,
``timeout``, ``reclaimed``. Policy (applied in ``kanban_db_dispatch._account_crashes``):
transient classes requeue WITHOUT counting a failure (the resume path re-enters the
interrupted session); ``HARNESS_STREAK_LIMIT`` consecutive runs of the same class
``rc0_no_handoff`` or ``rc_nonzero`` block the card with a ``BLOCKER:HARNESS <reason>`` comment instead
of looping. A transient streak past ``TRANSIENT_STREAK_LIMIT`` is blocked the same way so
a worker that is OOM-killed every attempt cannot loop forever either.
"""

from __future__ import annotations

import signal
import sqlite3
import time
from typing import Optional

STDERR_TAIL_BYTES = 2048

SIGNAL_KILLED = "signal_killed"
GATEWAY_SHUTDOWN = "gateway_shutdown"
RC_NONZERO = "rc_nonzero"
RC0_NO_HANDOFF = "rc0_no_handoff"
TIMEOUT = "timeout"
RECLAIMED = "reclaimed"

TRANSIENT_REASONS = frozenset({SIGNAL_KILLED, GATEWAY_SHUTDOWN})
HARNESS_REASONS = frozenset({RC0_NO_HANDOFF, RC_NONZERO})
HARNESS_STREAK_LIMIT = 2
TRANSIENT_STREAK_LIMIT = 6
_STREAK_SCAN_LIMIT = 50

# Handoff events a worker emits by calling kanban_complete / kanban_block / request_review.
_HANDOFF_EVENTS = ("completed", "blocked", "review_requested")
# Shell convention 128+N: a wrapper that died of signal N reports this rc.
_SIGNAL_RC_BASE = 128
# Signals a supervisor sends when it is going down (systemd stop, launchd unload, Ctrl-C).
_SHUTDOWN_SIGNALS = frozenset(int(s) for s in (
    getattr(signal, "SIGTERM", 15), getattr(signal, "SIGHUP", 1), getattr(signal, "SIGINT", 2)))

_PROCESS_STARTED_AT = time.time()


def stderr_tail(task_id: Optional[str], board: Optional[str] = None) -> str:
    """Last <= 2 KB of the worker's log (stdout+stderr land there). "" when absent."""
    if not task_id:
        return ""
    from hermes_cli import kanban_db as kb

    try:
        raw = kb.read_worker_log(task_id, tail_bytes=STDERR_TAIL_BYTES * 2, board=board) or ""
    except Exception:
        return ""
    data = raw.encode("utf-8", "replace")[-STDERR_TAIL_BYTES:]
    return data.decode("utf-8", "ignore")


def handoff_called(conn: sqlite3.Connection, task_id: str, run_id: Optional[int]) -> bool:
    """Did this run's worker call a terminal kanban tool (complete/block/request_review)?"""
    if run_id is None:
        return False
    marks = ",".join("?" * len(_HANDOFF_EVENTS))
    try:
        row = conn.execute(
            f"SELECT 1 FROM task_events WHERE task_id = ? AND run_id = ? AND kind IN ({marks}) LIMIT 1",
            (task_id, int(run_id), *_HANDOFF_EVENTS),
        ).fetchone()
    except sqlite3.Error:
        return False
    return row is not None


def _gateway_went_down(run_started_at: Optional[int]) -> bool:
    """The worker's run predates THIS dispatcher process: its parent gateway/dispatcher was
    restarted under it, which is why the reap registry has no record of the exit."""
    return run_started_at is not None and float(run_started_at) < _PROCESS_STARTED_AT


def classify_dead_worker(kind: str, code: Optional[int], *, run_started_at: Optional[int]) -> str:
    """Map ``_classify_worker_exit``'s ``(kind, code)`` to an exit-reason class."""
    if kind == "clean_exit":
        return RC0_NO_HANDOFF
    if kind == "signaled":
        if code in _SHUTDOWN_SIGNALS and _gateway_went_down(run_started_at):
            return GATEWAY_SHUTDOWN
        return SIGNAL_KILLED
    if kind in ("nonzero_exit", "terminal_provider", "rate_limited"):
        if kind == "nonzero_exit" and code is not None and code > _SIGNAL_RC_BASE \
                and code - _SIGNAL_RC_BASE < 65:
            return SIGNAL_KILLED
        return RC_NONZERO
    # ``unknown``: no reap record and no exit trailer — the worker never reached its exit
    # epilogue, i.e. it was killed. Attribute it to a gateway restart when the run predates us.
    return GATEWAY_SHUTDOWN if _gateway_went_down(run_started_at) else SIGNAL_KILLED


def build_exit_reason(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: str,
    run_id: Optional[int],
    rc: Optional[int] = None,
    sig: Optional[int] = None,
    board: Optional[str] = None,
) -> dict:
    """The ``exit_reason`` record stored on the closed run. Never raises."""
    if sig is None and rc is not None and rc > _SIGNAL_RC_BASE and rc - _SIGNAL_RC_BASE < 65:
        sig = rc - _SIGNAL_RC_BASE
    return {
        "reason": reason,
        "rc": rc,
        "signal": sig,
        "stderr_tail": stderr_tail(task_id, board=board),
        "handoff_called": handoff_called(conn, task_id, run_id),
    }


def exit_reason_for_dead_worker(
    conn: sqlite3.Connection, task_id: str, *, kind: str, code: Optional[int],
    run_id: Optional[int], run_started_at: Optional[int], board: Optional[str] = None,
) -> dict:
    reason = classify_dead_worker(kind, code, run_started_at=run_started_at)
    rc = code if kind in ("clean_exit", "nonzero_exit", "terminal_provider", "rate_limited") else None
    sig = code if kind == "signaled" else None
    return build_exit_reason(conn, task_id, reason=reason, run_id=run_id, rc=rc, sig=sig, board=board)


def run_started_at(conn: sqlite3.Connection, run_id: Optional[int]) -> Optional[int]:
    if run_id is None:
        return None
    row = conn.execute("SELECT started_at FROM task_runs WHERE id = ?", (int(run_id),)).fetchone()
    return int(row["started_at"]) if row and row["started_at"] is not None else None


def _run_reason(metadata_json: Optional[str]) -> Optional[str]:
    from hermes_cli import kanban_db as kb

    er = kb._json_dict(metadata_json).get("exit_reason")
    return er.get("reason") if isinstance(er, dict) else None


def reason_streak(conn: sqlite3.Connection, task_id: str, reasons: frozenset) -> tuple[int, Optional[str]]:
    """Trailing count of closed runs whose exit reason is in ``reasons`` (``rate_limited``
    runs are neutral), plus the newest such reason."""
    rows = conn.execute(
        "SELECT outcome, metadata FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY id DESC LIMIT ?", (task_id, _STREAK_SCAN_LIMIT),
    ).fetchall()
    streak, newest = 0, None
    for row in rows:
        if row["outcome"] == "rate_limited":
            continue
        reason = _run_reason(row["metadata"])
        if reason not in reasons:
            break
        newest = newest or reason
        streak += 1
    return streak, newest


def harness_block(conn: sqlite3.Connection, task_id: str, reason: str, streak: int, detail: str) -> bool:
    """Block the card (sticky, operator must unblock) with a ``BLOCKER:HARNESS`` comment."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    text = (f"BLOCKER:HARNESS {reason} — {streak} consecutive worker exits classified {reason}; "
            f"blocked instead of re-spawning. {detail}").strip()
    tripped = kbd._record_task_failure(
        conn, task_id, error=text, outcome="crashed", force_trip=True,
        release_claim=False, end_run=False,
        event_payload_extra={"harness_reason": reason, "harness_streak": streak},
    )
    if tripped:
        try:
            kb.add_comment(conn, task_id, "dispatcher", text)
        except Exception:
            pass
    return tripped

