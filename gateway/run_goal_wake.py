"""Gateway-side wake for goals parked on a wait barrier.

A judge ``wait`` verdict (or the goal tool's ``wait_on`` / ``wait_on_session`` / ``wait_for_seconds``)
parks the goal loop: no turn runs until the barrier lifts (timer elapsed, pid exited, watched session
finished). The CLI lifts it from its idle hook (``_maybe_resume_parked_goal``); the gateway has no
idle hook — a parked chat session stayed parked until an unrelated message arrived. This chore is
the gateway's equivalent, run from the housekeeping tick: for every ACTIVE goal that carries a
barrier, if the session is idle and ``GoalManager.is_waiting()`` says the barrier is gone, the
canonical continuation prompt is injected as the session's next turn (same synthetic-event path the
post-turn judge and the heartbeat poller use).

Once per barrier by construction: ``is_waiting()`` clears the barrier when it reports False, so the
next scan's ``list_parked_goals`` no longer returns the goal. A busy session is not evaluated at all
(the barrier stays on disk, the next tick retries); the agent's own post-turn judge handles it if the
turn in flight ends first. Multiplex: every served profile's store is scanned under its own runtime
scope (``_handoff_watch_scopes`` shape) and each hit is delivered through that profile's adapters.
"""
from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any, Optional

logger = logging.getLogger("gateway.run")

_LABEL = "goal barrier wake"


def _origin_platform(origin: Any) -> str:
    plat = getattr(origin, "platform", "")
    return str(getattr(plat, "value", plat) or "").lower()


def _scope_for(profile_home):
    """Async runtime scope for one served store: a profile home binds its own scope; ``None`` (the
    launch profile's store) binds the launch scope only once the process multiplexes."""
    if profile_home is not None:
        from gateway.run import _async_profile_runtime_scope

        return _async_profile_runtime_scope(profile_home)
    from tui_gateway.launch_profile_policy import async_launch_profile_scope_if_multiplexed

    return async_launch_profile_scope_if_multiplexed()


def _session_is_busy(runner, session_key: str, adapter: Any) -> bool:
    """Running turn, adapter-claimed session or a non-empty FIFO: the wake would race live work."""
    if runner._is_session_running(session_key):
        return True
    if session_key in (getattr(adapter, "_active_sessions", None) or ()):
        return True
    try:
        return runner._queue_depth(session_key, adapter=adapter) > 0
    except Exception:
        return False


async def _wake_one(runner, session_id: str, entry: Any, profile_name: Optional[str], *, idle: bool = False) -> bool:
    """Lift one parked goal if its barrier is gone and the session is idle, or re-poke an idle-stalled
    active goal (``idle=True``). True when a continuation was injected."""
    from hermes_cli.goals import GoalManager

    source = runner._restored_source(entry)
    if source is None:
        return False
    if profile_name and not getattr(source, "profile", None):
        source.profile = profile_name
    session_key = getattr(entry, "session_key", None) or runner._session_key_for_source(source)
    store = getattr(runner, "session_store", None)
    if store is not None and store.peek_session_id(session_key) != session_id:
        return False  # a reset/compression published a new owner; the old goal row is stale
    adapter = runner._delivery_adapter_for(source)
    if adapter is None or not getattr(adapter, "_message_handler", None):
        return False
    if _session_is_busy(runner, session_key, adapter):
        return False  # deferred: barrier stays on disk, the next tick retries

    with runner._profile_scope_for_source(source):
        def _lift() -> Optional[str]:
            # is_waiting() evaluates deadline / pid liveness / watched-session state and CLEARS a
            # lifted barrier (persisted) — that write is the once-per-barrier throttle.
            mgr = GoalManager(session_id)
            if mgr.is_waiting():
                return None
            if idle and not mgr.state_is_idle_stalled():
                return None  # a turn landed since the scan
            if idle:
                # Throttle: an idle wake counts as a turn boundary so the next one waits a full
                # idle_wake_seconds again even if the agent answers with nothing.
                mgr.touch_last_turn()
            return mgr.next_continuation_prompt()

        prompt = await runner._run_in_executor_with_context(_lift)
        if not prompt:
            return False
        event = runner._synthetic_prompt_event(source, prompt)  # system:* lanes become internal inside
        event.metadata["gateway_session_key"] = session_key
        logger.info(
            "%s: %s for session %s — injecting continuation (%s chat=%s thread=%s)",
            _LABEL, "idle-active goal re-poked" if idle else "barrier lifted", session_id, getattr(source, "platform", None), getattr(source, "chat_id", None),
            getattr(source, "thread_id", None),
        )
        # handle_message, not the FIFO: the FIFO alone never wakes an idle session.
        await adapter.handle_message(event)
        return True


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


async def _scan_one_store(runner, profile_name: Optional[str]) -> int:
    from hermes_cli.goals import list_idle_active_goals, list_parked_goals

    await runner._warm_goals_session_db(_LABEL)
    parked = await runner._run_in_executor_with_context(list_parked_goals)
    idle_rows = await runner._run_in_executor_with_context(list_idle_active_goals)
    if not parked and not idle_rows:
        return 0
    store = getattr(runner, "session_store", None)
    entries_by_sid: dict = {}
    if store is not None:
        for entry in store.list_sessions():
            if entry.session_id and entry.origin is not None and not getattr(entry, "suspended", False):
                entries_by_sid.setdefault(entry.session_id, entry)
        for sid, entry in list(entries_by_sid.items()):
            entries_by_sid[sid] = handoff_lane_ok(runner, sid, entry, store)[0]
    woken = 0
    for rows, idle in ((parked, False), (idle_rows, True)):
        for session_id, _state in rows:
            entry = entries_by_sid.get(session_id)
            if entry is None:
                continue  # CLI / TUI-owned goal: its own idle hook drives it
            try:
                if await _wake_one(runner, session_id, entry, profile_name, idle=idle):
                    woken += 1
            except Exception as exc:
                logger.warning("%s: resume for session %s failed: %s", _LABEL, session_id, exc)
    return woken


async def wake_lifted_goal_barriers(runner) -> int:
    """One pass over every served profile's goal store. Returns the number of sessions woken."""
    from agent.estop import check_paused
    from gateway.run import _handoff_watch_scopes

    if check_paused("goal", logger):
        return 0  # `hermes pause`: barriers stay on disk and lift after `hermes resume`
    woken = 0
    for profile_name, profile_home in _handoff_watch_scopes(runner):
        try:
            async with _scope_for(profile_home):
                woken += await _scan_one_store(runner, profile_name)
        except Exception as exc:
            logger.debug("%s: scan of %s failed: %s", _LABEL, profile_name or "default", exc)
    return woken


def housekeeping_goal_barrier_wake(runner, loop) -> None:
    """Housekeeping-thread entry: schedule one wake pass onto the gateway loop (fire-and-forget; the
    pass logs its own failures, so a slow adapter never wedges the housekeeping heartbeat)."""
    if runner is None or loop is None:
        return
    from agent.async_utils import safe_schedule_threadsafe

    safe_schedule_threadsafe(
        wake_lifted_goal_barriers(runner), loop, logger=logger,
        log_message="goal barrier wake scheduling error")
