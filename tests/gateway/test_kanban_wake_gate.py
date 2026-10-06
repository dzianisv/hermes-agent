"""Issue #17: every cron/alert card wakes the same supervisor session.

Real data: product-lead-vibebrowser session 20260929_075228_af900b got 602
``[kanban]`` wake turns (plus 474 heartbeats) from unrelated cards, pulling the
EM off its active outcome. With ``kanban.wake_gate`` enabled, only the active
outcome's cards and production incidents wake immediately; the rest are
coalesced into one deduped digest wake at the next checkpoint.
"""
import asyncio
import types

import pytest

from evals.heartbeat_idle_wire import WireAdapter
from gateway.config import Platform, PlatformConfig
from gateway import kanban_watchers_notifier as kwn
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


def _gate_cfg(active):
    cfg = {"wake_gate": {"enabled": True, "active_task_ids": active, "digest_interval_seconds": 600}}
    try:
        from gateway.kanban_wake_gate import WakeGateConfig
    except ImportError:  # unmodified base: no gate exists, config is ignored
        return types.SimpleNamespace(enabled=True)
    return WakeGateConfig.from_kanban_cfg(cfg)


@pytest.mark.asyncio
async def test_unrelated_wakes_coalesce_into_one_digest(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    adapter = WireAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    adapter.wire = []
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner.adapters = {adapter.platform: adapter}
    runner._delivery_adapter_for = lambda source: adapter
    runner._kanban_dispatcher_lock_handle = object()
    source = SessionSource(platform=adapter.platform, chat_id="42", user_id="42", chat_type="dm")
    key = build_session_key(source)

    conn = kbc.connect()
    try:
        outcome = kb.create_task(conn, title="ship release", assignee="worker", session_id=key)
        child = kb.create_task(conn, title="release step", assignee="worker", session_id=key)
        noise = [kb.create_task(conn, title=f"cron: disk check {i}", assignee="worker", session_id=key) for i in range(3)]
        incident = kb.create_task(conn, title="[incident] prod API 500s", assignee="worker", session_id=key)
        for tid in [child, *noise, incident]:
            kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="42",
                               user_id="42", chat_type="dm", delivery_mode="wake")
        for tid in [child, *noise, incident]:
            kb.complete_task(conn, tid, summary="done")
        # The same noisy card fires again: must dedupe into its digest line.
        kb.block_task(conn, noise[0], reason="flaky", kind="transient")
        kb.unblock_task(conn, noise[0])
        kb.complete_task(conn, noise[0], summary="done again")
        kb.link_tasks(conn, outcome, child)  # child belongs to the active outcome
    finally:
        conn.close()

    runner._kanban_wake_gate = _gate_cfg([outcome])
    received = []

    async def handler(event):
        received.append(event.text)

    adapter.set_message_handler(handler)
    await adapter.connect()
    try:
        for _ in range(2):
            for d in await asyncio.to_thread(kwn._notifier_collect, runner, kb, notifier_profile=None,
                                             gc_due=False, gc_retention_days=30):
                await kwn._KanbanNotification(runner, d, platform_cls=Platform, sub_fail_counts={}).deliver()
        while adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks))
        # Immediate: the active outcome's child and the incident only.
        assert len(received) == 2, received
        assert any(child in t for t in received) and any(incident in t for t in received)

        flush = getattr(kwn, "flush_wake_digest")
        assert await flush(runner, platform_cls=Platform) == 0  # checkpoint not due yet
        import time
        assert await flush(runner, platform_cls=Platform, now=time.time() + 601) == 1
        while adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks))
        assert len(received) == 3
        digest = received[-1]
        assert digest.startswith("[kanban digest]")
        for tid in noise:
            assert digest.count(tid) == 1
        assert await flush(runner, platform_cls=Platform, now=time.time() + 1201) == 0  # settled
    finally:
        await adapter.disconnect()


def test_gate_disabled_by_default():
    from gateway.kanban_wake_gate import WakeGateConfig, is_immediate
    cfg = WakeGateConfig.from_kanban_cfg({})
    assert cfg.enabled is False
    assert is_immediate(cfg, {"sub": {"task_id": "t"}, "task": None}, [])
