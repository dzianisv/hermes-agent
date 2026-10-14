"""Which routing entry owns a handed-off session's current lane."""
from __future__ import annotations

from contextlib import suppress
from typing import Any


def _origin_platform(origin: Any) -> str:
    plat = getattr(origin, "platform", "")
    return str(getattr(plat, "value", plat) or "").lower()


def handoff_lane_ok(runner, session_id: str, entry: Any, store: Any) -> Any:
    """Return the routing entry that owns ``session_id``'s current lane.

    A handed-off session keeps its old routing key (store entries are never deleted on handoff), so
    "most recently updated" can still be the *source* channel for hours — a Telegram-born goal moved
    to Slack kept waking into Telegram, and heartbeats restored from the stale Telegram entry kept
    posting there. The session row names the destination platform; only ``handoff_platform`` is
    trustworthy because the row's chat/thread get re-stamped by whichever lane last ran a turn.
    Returns ``(lane_entry, on_lane)``: the newest entry on the handoff lane when one exists, else
    ``entry`` itself with ``on_lane=False`` when it is a stale source lane (the caller decides whether
    a stale lane is still an acceptable fallback).
    """
    session_db = getattr(runner, "_goal_session_db", None)
    if session_db is None:
        with suppress(Exception):
            from hermes_cli.goals import _get_session_db
            session_db = _get_session_db()
    if session_db is None:
        return entry, True
    with suppress(Exception):
        row = session_db.get_session(session_id) or {}
        if row.get("handoff_state") != "completed":
            return entry, True
        want = str(row.get("handoff_platform") or "").lower()
        if not want or _origin_platform(entry.origin) == want:
            return entry, True
        for other in store.list_sessions():  # newest first
            if other.session_id == session_id and other.origin is not None \
                    and not getattr(other, "suspended", False) and _origin_platform(other.origin) == want:
                return other, True
        return entry, False
    return entry, True
