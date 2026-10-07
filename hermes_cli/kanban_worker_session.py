"""Record a dispatcher-spawned Kanban worker's session id on its live run.

The dispatcher resumes an interrupted worker's session (``chat --resume``) on the next
attempt; it can only do that if the id is on the run before the worker dies, and handoff
tools (complete/block) stamp it too late for a crash. Best-effort: never raises.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def stamp_worker_session_on_run(session_id: str | None) -> bool:
    """Merge ``worker_session_id`` into the worker's own open run. True when written."""
    try:
        from agent.delegation_context import owned_kanban_task

        task_id = owned_kanban_task()
        raw_run = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
        if not (task_id and raw_run and session_id):
            return False
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        with kbc.connect_closing() as conn:
            return kb.merge_run_metadata(conn, task_id, int(raw_run), {"worker_session_id": str(session_id)})
    except Exception as exc:
        logger.debug("kanban worker session stamp skipped: %s", exc)
        return False
