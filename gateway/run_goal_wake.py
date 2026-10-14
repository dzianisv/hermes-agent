"""Gateway wake for goals: lift a parked goal when its barrier changes, re-poke an idle active one.
Runs from the heartbeat poller per served profile store; admission is ``session_wake.claim_goal``."""
import logging

logger = logging.getLogger("gateway.run")


async def wake_lifted_goal_barriers(runner) -> int:
    from agent.estop import check_paused
    from gateway.handoff_lane import handoff_lane_ok, live_lanes
    from gateway.run import _handoff_watch_scopes, profile_store_scope
    from gateway.session_wake import claim_goal
    from hermes_cli.goals import list_idle_active_goals, list_parked_goals
    woken, store, run = 0, runner.session_store, runner._run_in_executor_with_context
    for name, home in [] if check_paused("goal", logger) else _handoff_watch_scopes(runner):
        try:
            async with profile_store_scope(home):
                await runner._warm_goals_session_db("goal barrier wake")
                rows = [(r[0], False) for r in await run(list_parked_goals)]
                rows += [(r[0], True) for r in await run(list_idle_active_goals)]
                lanes = live_lanes(store) if rows else {}
                for sid, idle in rows:  # sid not in lanes = CLI/TUI goal: its own idle hook drives it
                    lane, ok = handoff_lane_ok(runner, sid, lanes[sid], store) if sid in lanes else (None, False)
                    try:
                        if not (source := ok and runner._restored_source(lane)):
                            continue
                        source.profile = getattr(source, "profile", None) or name or None
                        key = getattr(lane, "session_key", None) or runner._session_key_for_source(source)
                        adapter = runner._delivery_adapter_for(source)
                        if store.peek_session_id(key) != sid or not runner._lane_free(key, adapter):
                            continue
                        with runner._profile_scope_for_source(source):
                            if not (got := await run(claim_goal, sid, idle)):
                                continue
                            logger.info("goal barrier wake: %s for session %s — injecting continuation (%s chat=%s thread=%s)",
                                        "idle-active goal re-poked" if idle else "barrier lifted", sid,
                                        source.platform, source.chat_id, getattr(source, "thread_id", None))
                            await runner._inject_owned_prompt(adapter, source, key, sid, got[1], got[0])
                            woken += 1
                    except Exception as exc:
                        logger.warning("goal barrier wake: resume for session %s failed: %s", sid, exc)
        except Exception as exc:
            logger.debug("goal barrier wake: scan of %s failed: %s", name or "default", exc)
    return woken
