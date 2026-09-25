"""WIP caps must apply on the next dispatcher tick, not only after a gateway restart.

``_kanban_dispatcher_watcher`` used to bake ``max_in_progress`` /
``max_in_progress_per_profile`` into ``_DispatcherSettings`` once at boot. An
operator who edited those keys into config.yaml after the gateway was already
up kept dispatching under the old (often uncapped) numbers until restart.
Same class of bug as auto-decompose (#49638): the caps are re-read every tick,
and a failed re-read keeps the last good cap instead of falling open.
"""

from __future__ import annotations

import asyncio

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _seed_profile_load(running: int = 3) -> str:
    """3 (or N) running tasks plus one ready task, all assigned to ``default``.

    ``default`` always passes the spawn-profile gate, so the only thing that
    can refuse the ready task is the concurrency cap under test.
    """
    kb.create_board(slug="default", name="Caps")
    with kbc.connect_closing() as conn:
        running_ids = [
            kb.create_task(conn, title=f"running-{i}", assignee="default")
            for i in range(running)
        ]
        ready_id = kb.create_task(conn, title="ready-over-cap", assignee="default")
        for tid in running_ids:
            assert kb.claim_task(conn, tid) is not None, tid
        assert kb.get_task(conn, ready_id).status == "ready"
    return ready_id


def _drive_two_ticks(monkeypatch, state, after_first):
    """Run the embedded watcher for exactly two ticks against ``state``.

    ``load_config`` returns a fresh dict each call, matching a real YAML
    re-read. ``after_first`` runs between ticks and may mutate ``state``.
    Dispatch is dry-run so the ready task is still there for tick 2 — the
    cap, not an empty queue, is what must refuse the second claim.
    """
    import gateway.kanban_watchers as watchers
    from hermes_cli import kanban_db_dispatch as kbd

    seen: list[tuple[dict, object]] = []
    real_dispatch = kbd.dispatch_once

    def spy(conn, **kwargs):
        kwargs = dict(kwargs)
        kwargs["dry_run"] = True
        result = real_dispatch(conn, **kwargs)
        seen.append((kwargs, result))
        return result

    monkeypatch.setattr(kbd, "dispatch_once", spy)

    def load_config(*_args, **_kwargs):
        if state.get("raise"):
            raise RuntimeError("config unreadable")
        return {"kanban": dict(state["kanban"])}

    monkeypatch.setattr("hermes_cli.config.load_config", load_config)

    async def noop_sleep(_delay=0):
        return None

    async def inline(func, *args):
        return func(*args)

    monkeypatch.setattr(watchers.asyncio, "sleep", noop_sleep)
    monkeypatch.setattr(watchers, "_to_thread_process_service", inline)
    monkeypatch.delenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", raising=False)

    runner = watchers.GatewayKanbanWatchersMixin()
    runner._running = True
    between = {"n": 0}

    async def sleep_between(_interval):
        between["n"] += 1
        if between["n"] == 1:
            after_first()
        else:
            runner._running = False

    runner._sleep_between_ticks = sleep_between
    asyncio.run(runner._kanban_dispatcher_watcher())
    return seen


def _base_kanban(**extra):
    cfg = {
        "dispatch_in_gateway": True,
        "auto_decompose": False,
        "dispatch_interval_seconds": 60,
        "max_in_progress": 50,
        "reconcile_orphans": False,
    }
    cfg.update(extra)
    return cfg


def test_per_profile_cap_applies_on_next_tick_without_restart(monkeypatch, tmp_path):
    """Unset per-profile cap allows a 4th running task; cap=3 on the next tick refuses it.

    The dispatcher object is the one the watcher built at boot. Config is
    changed between ticks, not by reconstructing the watcher.
    """
    db_path = kb.kanban_db_path()
    assert str(tmp_path) in str(db_path), db_path
    ready_id = _seed_profile_load(3)
    state = {"kanban": _base_kanban(), "raise": False}

    def tighten():
        state["kanban"]["max_in_progress_per_profile"] = 3

    seen = _drive_two_ticks(monkeypatch, state, tighten)
    assert len(seen) == 2, seen

    first_kwargs, first = seen[0]
    second_kwargs, second = seen[1]
    assert first_kwargs.get("max_in_progress_per_profile") is None
    assert [row[0] for row in first.spawned] == [ready_id]
    assert second_kwargs.get("max_in_progress_per_profile") == 3
    assert second.spawned == []
    assert any(row[0] == ready_id for row in second.skipped_per_profile_capped)


def test_board_cap_applies_on_next_tick_without_restart(monkeypatch, tmp_path):
    """Board-wide ``max_in_progress`` is the same live re-read, not a boot snapshot."""
    db_path = kb.kanban_db_path()
    assert str(tmp_path) in str(db_path), db_path
    ready_id = _seed_profile_load(3)
    state = {"kanban": _base_kanban(max_in_progress=10), "raise": False}

    def tighten():
        state["kanban"]["max_in_progress"] = 3

    seen = _drive_two_ticks(monkeypatch, state, tighten)
    assert len(seen) == 2, seen

    first_kwargs, first = seen[0]
    second_kwargs, second = seen[1]
    assert first_kwargs.get("max_in_progress") == 10
    assert [row[0] for row in first.spawned] == [ready_id]
    assert second_kwargs.get("max_in_progress") == 3
    assert second.spawned == []
    assert not any(row[0] == ready_id for row in second.skipped_per_profile_capped)


def test_config_reread_failure_keeps_last_cap():
    """A mid-run config error must not drop a live cap back to uncapped."""
    from gateway.kanban_watchers_dispatcher import (
        _DispatcherSettings,
        _reread_dispatcher_settings,
    )

    previous = _DispatcherSettings(
        interval=60.0,
        max_spawn=2,
        max_in_progress=4,
        failure_limit=2,
        stale_timeout_seconds=0,
        reconcile_orphans=True,
        default_assignee=None,
        max_in_progress_per_profile=3,
    )

    def boom():
        raise RuntimeError("yaml mid-write")

    kept = _reread_dispatcher_settings(boom, kb, previous)
    assert kept is previous
    assert kept.max_in_progress_per_profile == 3
    assert kept.max_in_progress == 4
    assert kept.max_spawn == 2

    def garbage():
        return None

    kept_garbage = _reread_dispatcher_settings(garbage, kb, previous)
    assert kept_garbage is previous


def test_config_reread_failure_does_not_uncap_running_watcher(monkeypatch, tmp_path):
    """Watcher-level fail-safe: tick 2's config exception still enforces tick 1's cap."""
    db_path = kb.kanban_db_path()
    assert str(tmp_path) in str(db_path), db_path
    ready_id = _seed_profile_load(3)
    state = {
        "kanban": _base_kanban(max_in_progress_per_profile=3),
        "raise": False,
    }

    def break_config():
        state["raise"] = True

    seen = _drive_two_ticks(monkeypatch, state, break_config)
    assert len(seen) == 2, seen
    _first_kwargs, first = seen[0]
    second_kwargs, second = seen[1]
    assert first.spawned == []
    assert any(row[0] == ready_id for row in first.skipped_per_profile_capped)
    assert second_kwargs.get("max_in_progress_per_profile") == 3
    assert second.spawned == []
    assert any(row[0] == ready_id for row in second.skipped_per_profile_capped)
