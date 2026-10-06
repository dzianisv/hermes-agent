"""Turn-end guard for kanban workers, which must end with a terminal board tool that hands
the card to whoever owns it next (``kanban_complete``, ``kanban_block``,
``kanban_request_review``, ``kanban_request_changes``). Some models narrate the next step
and stop with no tool calls; Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``. Policy-only: return a bounded synthetic nudge so the loop continues
instead of exiting.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from agent.delegation_context import owned_kanban_task


# Every tool that ends this worker's responsibility for the card, not just the two that
# close it out: ``kanban_request_review`` moves it to ``review`` (goals.py's continuation /
# finalize prompts tell builders to call it) and ``kanban_request_changes`` returns it to
# ``ready`` (the sdlc-review skill tells reviewers to). Nudging after either asks a worker
# that did the right thing to ``kanban_complete`` a card it must not close.
_TERMINAL_KANBAN_TOOLS = frozenset({
    "kanban_complete",
    "kanban_block",
    "kanban_request_review",
    "kanban_request_changes",
    "kanban_approve",
})

_DEFAULT_MAX_ATTEMPTS = 2


def kanban_stop_nudge_enabled() -> bool:
    """On when ``HERMES_KANBAN_TASK`` is set for the dispatcher-owned worker, unless
    ``HERMES_KANBAN_STOP_NUDGE`` disables it. In-process delegate_task children and cron runs
    inherit the env var but own no board task and carry no kanban toolset."""
    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(owned_kanban_task())


def _tool_call_name(tc: Any) -> str:
    """Tool name from a dict or object tool call (``function.name`` first, then ``name``)."""
    if isinstance(tc, dict):
        fn = tc.get("function")
        return str((fn.get("name") if isinstance(fn, dict) else tc.get("name")) or "")
    fn = getattr(tc, "function", None)
    return str((getattr(fn, "name", "") if fn is not None else getattr(tc, "name", "")) or "")


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True if this conversation already invoked a terminal kanban tool."""
    for msg in filter(lambda m: isinstance(m, dict), messages or ()):
        role = msg.get("role")
        if role == "assistant" and any(
            _tool_call_name(tc) in _TERMINAL_KANBAN_TOOLS for tc in msg.get("tool_calls") or []
        ):
            return True
        if role == "tool" and str(msg.get("name") or "") in _TERMINAL_KANBAN_TOOLS:
            return True
    return False


def _bound_run_id() -> Optional[int]:
    raw = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def bound_run_disposition(task_id: Optional[str] = None) -> str:
    """Persisted disposition of the dispatcher-bound run, read-only (opened ``mode=ro``).

    ``"active"``  — this run is still the task's current, open run: the worker has not
                    recorded a status, so a same-session correction is warranted.
    ``"settled"`` — the run is closed (completed / blocked / review hand-off /
                    scheduled / changes requested / reclaimed or reassigned) or a
                    successor run owns the card. Nothing for this session to do; never
                    nudge it into touching a successor's run.
    ``"unknown"`` — no bound run id or the board is unreadable.

    The transcript is NOT consulted: a ``kanban_complete`` call the tool rejected (or
    whose result never arrived) leaves the run ``active`` and must still be corrected.
    """
    run_id = _bound_run_id()
    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if run_id is None or not tid:
        return "unknown"
    try:
        import sqlite3

        from hermes_cli.kanban_db import kanban_db_path

        path = kanban_db_path()
        if not path.exists():
            return "unknown"
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
        try:
            run = conn.execute(
                "SELECT task_id, status, ended_at FROM task_runs WHERE id = ?", (run_id,)
            ).fetchone()
            task = conn.execute(
                "SELECT status, current_run_id FROM tasks WHERE id = ?", (tid,)
            ).fetchone()
        finally:
            conn.close()
    except Exception:
        return "unknown"
    if run is None or task is None or run[0] != tid:
        return "unknown"
    if run[2] is None and run[1] == "running" and task[1] == run_id and task[0] == "running":
        return "active"
    return "settled"


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already completed/blocked, budget exhausted)."""
    if not kanban_stop_nudge_enabled() or attempts >= max_attempts:
        return None
    # The board, not the transcript, is the status source (#3): a terminal tool the board
    # rejected leaves the run open. Only a still-active bound run is nudged; a settled run
    # (handed off, or superseded by a successor) is left alone. When the run cannot be read,
    # fall back to the transcript so an unbound worker is not nudged forever — the
    # dispatcher's bounded protocol-violation budget still catches a real silent exit.
    disposition = bound_run_disposition(task_id)
    if disposition == "settled":
        return None
    if disposition == "unknown" and session_called_kanban_terminal(messages):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    # Reached only while the bound run is still open, so it never tells a worker to close a
    # card it already sent to review.
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` has not been handed off: this session made no terminal board "
        "call (`kanban_complete` / `kanban_request_review` / `kanban_block`). Ending now "
        "causes a protocol violation (clean exit with the card still `running`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work is done "
        "and needs no review, `kanban_request_review(summary=...)` if it is a code "
        "change that needs same-card review, OR `kanban_block(reason=...)` if you are "
        "blocked. Reviewers approve with `kanban_approve(summary=...)` (merge pending) or "
        "`kanban_complete` (nothing left to merge), or send the card back with "
        "`kanban_request_changes(reason=...)`.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


__all__ = ["bound_run_disposition", "build_kanban_stop_nudge", "kanban_stop_nudge_enabled", "session_called_kanban_terminal"]
