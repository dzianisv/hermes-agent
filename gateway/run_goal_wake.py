"""Gateway wake for goals: lift a parked goal when its barrier changes, re-poke an idle active one.
Runs from the heartbeat poller; ``GoalManager.is_waiting()`` clears a lifted barrier (once per barrier)."""
import logging

logger = logging.getLogger("gateway.run")


async def _wake_one(runner, sid, entry, profile, idle) -> bool:
    from hermes_cli.goals import GoalManager
    from gateway.session_wake import claim, stamp, track
    source = runner._restored_source(entry)
    if source is None:
        return False
    source.profile = getattr(source, "profile", None) or profile or None
    key = getattr(entry, "session_key", None) or runner._session_key_for_source(source)
    adapter = runner._delivery_adapter_for(source)
    if runner.session_store.peek_session_id(key) != sid or not runner._lane_free(key, adapter):
        return False

    def _admit():
        mgr = GoalManager(sid)
        if mgr.is_waiting() or (idle and not mgr.state_is_idle_stalled()):
            return None
        idle and mgr.touch_last_turn()
        prompt = mgr.next_continuation_prompt()
        token = prompt and claim(sid, "goal", revision=mgr.state.revision)
        return token and (prompt, token)
    with runner._profile_scope_for_source(source):
        if not (admitted := await runner._run_in_executor_with_context(_admit)):
            return False
        event = runner._synthetic_prompt_event(source, admitted[0])
        stamp(event, sid, admitted[1])
        event.metadata["gateway_session_key"] = key
        logger.info("goal barrier wake: %s for session %s — injecting continuation (%s chat=%s)",
                    "idle-active goal re-poked" if idle else "barrier lifted", sid, source.platform, source.chat_id)
        try:
            await adapter.handle_message(event)
        finally:
            track(adapter, key, event)
        return True


async def wake_lifted_goal_barriers(runner) -> int:
    from agent.estop import check_paused
    from gateway.handoff_lane import handoff_lane_ok
    from gateway.run import _handoff_watch_scopes, profile_store_scope
    from hermes_cli.goals import list_idle_active_goals, list_parked_goals
    woken, store = 0, runner.session_store
    for name, home in [] if check_paused("goal", logger) else _handoff_watch_scopes(runner):
        try:
            async with profile_store_scope(home):
                await runner._warm_goals_session_db("goal barrier wake")
                rows = [(r, False) for r in await runner._run_in_executor_with_context(list_parked_goals)]
                rows += [(r, True) for r in await runner._run_in_executor_with_context(list_idle_active_goals)]
                lanes = {}
                for e in store.list_sessions() if rows else ():
                    if e.session_id and e.origin is not None and not getattr(e, "suspended", False):
                        lanes.setdefault(e.session_id, e)
                for (sid, _), idle in rows:  # sid not in lanes = CLI/TUI goal, its own idle hook drives it
                    lane, ok = handoff_lane_ok(runner, sid, lanes[sid], store) if sid in lanes else (None, False)
                    try:
                        woken += bool(ok and await _wake_one(runner, sid, lane, name, idle))
                    except Exception as exc:
                        logger.warning("goal barrier wake: resume for session %s failed: %s", sid, exc)
        except Exception as exc:
            logger.debug("goal barrier wake: scan of %s failed: %s", name or "default", exc)
    return woken
