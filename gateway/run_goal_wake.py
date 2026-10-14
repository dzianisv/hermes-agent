"""Gateway wake for goals: lift a parked goal when its barrier changes, re-poke an idle active one.
Runs from the heartbeat poller; ``GoalManager.is_waiting()`` clears a lifted barrier (once per barrier)."""
from __future__ import annotations

import logging

from gateway.handoff_lane import handoff_lane_ok  # noqa: F401  (re-export)

logger = logging.getLogger("gateway.run")
_LABEL = "goal barrier wake"


async def _wake_one(runner, sid, entry, profile_name, *, idle=False) -> bool:
    from hermes_cli.goals import GoalManager
    from gateway.session_wake import claim, stamp, track
    source = runner._restored_source(entry)
    if source is None:
        return False
    if profile_name and not getattr(source, "profile", None):
        source.profile = profile_name
    key = getattr(entry, "session_key", None) or runner._session_key_for_source(source)
    store = getattr(runner, "session_store", None)
    adapter = runner._delivery_adapter_for(source)
    if (store is not None and store.peek_session_id(key) != sid) or not getattr(adapter, "_message_handler", None) \
            or runner._is_session_running(key) or key in (getattr(adapter, "_active_sessions", None) or ()):
        return False
    try:
        if runner._queue_depth(key, adapter=adapter) > 0:
            return False
    except Exception:
        pass
    with runner._profile_scope_for_source(source):
        def _lift():
            mgr = GoalManager(sid)
            if mgr.is_waiting() or (idle and not mgr.state_is_idle_stalled()):
                return None
            if idle:
                mgr.touch_last_turn()
            prompt = mgr.next_continuation_prompt()
            token = prompt and claim(sid, "goal", revision=mgr.state.revision)
            return (prompt, token) if token else None
        admitted = await runner._run_in_executor_with_context(_lift)
        if not admitted:
            return False
        event = runner._synthetic_prompt_event(source, admitted[0])
        stamp(event, sid, admitted[1])
        event.metadata["gateway_session_key"] = key
        logger.info("%s: %s for session %s — injecting continuation (%s chat=%s thread=%s)", _LABEL,
                    "idle-active goal re-poked" if idle else "barrier lifted", sid, getattr(source, "platform", None),
                    getattr(source, "chat_id", None), getattr(source, "thread_id", None))
        try:
            await adapter.handle_message(event)
        finally:
            track(adapter, key, event)
        return True


async def _scan_one_store(runner, profile_name) -> int:
    from hermes_cli.goals import list_idle_active_goals, list_parked_goals
    await runner._warm_goals_session_db(_LABEL)
    rows = [(r, False) for r in await runner._run_in_executor_with_context(list_parked_goals)]
    rows += [(r, True) for r in await runner._run_in_executor_with_context(list_idle_active_goals)]
    store = getattr(runner, "session_store", None)
    if not rows or store is None:
        return 0
    lanes: dict = {}
    for e in store.list_sessions():
        if e.session_id and e.origin is not None and not getattr(e, "suspended", False):
            lanes.setdefault(e.session_id, e)
    woken = 0
    for (sid, _state), idle in rows:
        if sid not in lanes:
            continue  # CLI / TUI-owned goal: its own idle hook drives it
        lane, on_lane = handoff_lane_ok(runner, sid, lanes[sid], store)
        try:
            woken += bool(on_lane and await _wake_one(runner, sid, lane, profile_name, idle=idle))
        except Exception as exc:
            logger.warning("%s: resume for session %s failed: %s", _LABEL, sid, exc)
    return woken


async def wake_lifted_goal_barriers(runner) -> int:
    from agent.estop import check_paused
    from gateway.run import _async_profile_runtime_scope, _handoff_watch_scopes
    from tui_gateway.launch_profile_policy import async_launch_profile_scope_if_multiplexed
    if check_paused("goal", logger):
        return 0
    woken = 0
    for name, home in _handoff_watch_scopes(runner):
        try:
            scope = _async_profile_runtime_scope(home) if home is not None else async_launch_profile_scope_if_multiplexed()
            async with scope:
                woken += await _scan_one_store(runner, name)
        except Exception as exc:
            logger.debug("%s: scan of %s failed: %s", _LABEL, name or "default", exc)
    return woken


def housekeeping_goal_barrier_wake(runner, loop) -> None:
    if runner is not None and loop is not None:
        loop.call_soon_threadsafe(runner._start_heartbeat_poller)
