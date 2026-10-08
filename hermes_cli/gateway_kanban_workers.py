"""Kanban workers that are running on this host, as seen by ``hermes gateway restart`` / ``stop``.

Design: a kanban worker is NOT a child the gateway has to keep alive. ``_default_spawn`` starts each
worker with ``start_new_session=True``, so it has its own session and process group. The command
line is a plain ``hermes -p <profile> --cli chat``, not ``gateway run``, and the worker logs to its
own file, not to the gateway's pipes. Stopping the gateway with SIGTERM, a launchd
``kickstart -k``, or SIGKILL of its process group therefore does not reach the worker. The next
gateway's dispatcher re-adopts the worker by the PID and start fingerprint stored on its
``task_runs`` row. ``detect_crashed_workers`` leaves a live, fingerprint-matching PID alone and
reads the exit code from the worker's log trailer once the worker exits.

A restart therefore does not drain: making it wait for hours of agent work would add nothing. This
module reports what is running so the operator can see that the workers continue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OpenWorkerRun:
    board: str
    task_id: str
    run_id: int
    worker_pid: int
    profile: Optional[str]
    alive: bool


def open_host_worker_runs() -> list[OpenWorkerRun]:
    """Open runs (``task_runs.ended_at IS NULL``) on this host that have a worker PID, across every board.

    "On this host" means the claim lock carries this host's prefix; other hosts' PIDs mean nothing
    here. Best effort: a board that cannot be read is skipped.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    host_prefix = kb._host_prefix()
    out: list[OpenWorkerRun] = []
    try:
        boards = [b.get("slug") or kb.DEFAULT_BOARD for b in kb.list_boards(include_archived=False)]
    except Exception:
        boards = [kb.DEFAULT_BOARD]
    for board in boards:
        try:
            path = kb.kanban_db_path(board=board)
            if not path.exists():
                continue
            conn = kbc.connect(path)
        except Exception:
            continue
        try:
            rows = conn.execute(
                "SELECT id, task_id, worker_pid, worker_started_at, claim_lock, profile FROM task_runs "
                "WHERE ended_at IS NULL AND worker_pid IS NOT NULL ORDER BY id"
            ).fetchall()
        except Exception:
            rows = []
        finally:
            conn.close()
        for row in rows:
            if not str(row["claim_lock"] or "").startswith(host_prefix):
                continue
            pid = int(row["worker_pid"])
            out.append(OpenWorkerRun(
                board=board, task_id=row["task_id"], run_id=int(row["id"]), worker_pid=pid,
                profile=row["profile"], alive=kbd._worker_alive(pid, row["worker_started_at"]),
            ))
    return out


def announce_surviving_workers(verb: str) -> list[OpenWorkerRun]:
    """Print the live kanban workers that survive this gateway ``verb``. Never raises."""
    try:
        runs = [r for r in open_host_worker_runs() if r.alive]
    except Exception:
        return []
    if not runs:
        return runs
    print(f"ℹ {len(runs)} kanban worker(s) running; they keep running through this gateway {verb}:")
    for r in runs:
        board = "" if r.board == "default" else f" [{r.board}]"
        print(f"    {r.task_id}{board}  pid {r.worker_pid}  ({r.profile or '?'})")
    print("  Workers run in their own process session; the next gateway's dispatcher re-adopts them by pid.")
    return runs
