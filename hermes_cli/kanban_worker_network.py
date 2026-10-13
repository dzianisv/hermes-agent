"""Last-chance board handoff for a dispatcher-owned kanban worker about to exit (harness #32).

Invariant: a worker never exits without a terminal board call. If the turn ended with the card
still ``running`` on this worker's run — the stop-loop nudges ran out while the host was offline
(runs 6693/6695: rc=0, no handoff) or the provider stayed unreachable past every retry and the
network wait (run 6801: rc=75) — the worker itself records an explicit ``kanban_block`` with the
reason (``network_unavailable`` when the failure was connectivity) instead of leaving the
dispatcher to guess from a bare exit code.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

NETWORK_UNAVAILABLE = "network_unavailable"
NO_HANDOFF = "no_handoff"


def _run_id() -> Optional[int]:
    try:
        raw = os.environ.get("HERMES_KANBAN_RUN_ID") or ""
        return int(raw) if raw else None
    except ValueError:
        return None


def _card_still_ours(conn, task_id: str, run_id: Optional[int]) -> bool:
    row = conn.execute("SELECT status, current_run_id FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None or row[0] != "running":
        return False
    return run_id is None or row[1] is None or int(row[1]) == int(run_id)


def exit_reason(result: Any, exit_code: int, *, network_ok: Callable[[], bool]) -> str:
    """``network_unavailable`` when the turn failed on connectivity or the host is offline now."""
    if isinstance(result, dict) and result.get("failure_reason") == NETWORK_UNAVAILABLE:
        return NETWORK_UNAVAILABLE
    if isinstance(result, dict):
        from agent.network_wait import is_network_error
        if is_network_error(result.get("error")):
            return NETWORK_UNAVAILABLE
    try:
        if not network_ok():
            return NETWORK_UNAVAILABLE
    except Exception:
        pass
    return NO_HANDOFF


def ensure_terminal_board_call(
    result: Any, exit_code: int, *, conn=None, network_ok: Optional[Callable[[], bool]] = None,
) -> int:
    """Block this worker's still-running card with an explicit reason; return the exit code to use.

    No-op (returns ``exit_code``) outside a dispatcher-owned worker, on interrupt (130), or when
    the card already left ``running`` for this run (a terminal tool landed). After a successful
    self-block the worker made its terminal board call, so it exits 0."""
    from agent.delegation_context import owned_kanban_task

    task_id = owned_kanban_task()
    if not task_id or exit_code == 130:
        return exit_code
    if network_ok is None:
        from agent.network_wait import network_available as network_ok  # type: ignore[assignment]
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    own = conn is None
    try:
        conn = conn or kbc.connect()
    except Exception:
        logger.warning("kanban exit guard: board unreachable; exiting rc=%s", exit_code, exc_info=True)
        return exit_code
    try:
        run_id = _run_id()
        if not _card_still_ours(conn, task_id, run_id):
            return exit_code
        reason = exit_reason(result, exit_code, network_ok=network_ok)
        err = str(result.get("error") or "")[:300] if isinstance(result, dict) else ""
        if reason == NETWORK_UNAVAILABLE:
            text = (f"{NETWORK_UNAVAILABLE}: model provider unreachable (host offline / DNS / connect "
                    f"failure) after bounded retries and connectivity wait; worker exit rc={exit_code}. "
                    f"Unblock to resume the same session once the network is back. {err}").strip()
            kind = "transient"
        else:
            text = (f"{NO_HANDOFF}: worker turn ended (rc={exit_code}) with no kanban_complete/"
                    f"kanban_request_review/kanban_block after stop-loop nudges. {err}").strip()
            kind = "needs_input"
        ok = kb.block_task(conn, task_id, reason=text, kind=kind, expected_run_id=run_id)
        if ok:
            try:
                kb.add_comment(conn, task_id, "worker", f"BLOCKER:{reason.upper()} {text}")
            except Exception:
                pass
            print(f"⛔ kanban worker recorded explicit block on {task_id}: {reason}", flush=True)
            logger.warning("kanban exit guard blocked %s run=%s reason=%s rc=%s", task_id, run_id, reason, exit_code)
            return 0
        return exit_code
    except Exception:
        logger.warning("kanban exit guard failed", exc_info=True)
        return exit_code
    finally:
        if own:
            try:
                conn.close()
            except Exception:
                pass


__all__ = ["NETWORK_UNAVAILABLE", "NO_HANDOFF", "ensure_terminal_board_call", "exit_reason"]
