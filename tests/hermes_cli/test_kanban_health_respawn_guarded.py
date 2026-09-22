"""Health telemetry must not call a deliberate guard deferral "stuck".

During the t_6a6ac2d3 incident the dispatcher logged

    kanban dispatcher stuck: ready queue non-empty for N consecutive ticks
    but 0 workers spawned.

while it was in fact deferring the card on purpose via the respawn guard
(``respawn_guarded {"reason":"active_pr"}``, ~170 times). ``skipped_nonspawnable``
was already excluded from the bad-tick count on exactly those grounds;
``respawn_guarded`` must be too.

TWO implementations carry the counter and both are covered here:

* ``gateway/kanban_watchers.py`` + ``gateway/kanban_watchers_dispatcher.py`` —
  ``bad_ticks``, the live path;
* ``hermes_cli/kanban_ops.py`` — ``health_state``, the ``--force`` daemon.

Both tests drive the REAL loop body against a REAL temp board (only
``dispatch_once`` is stubbed) and assert on the warning the operator would
actually see. The negative controls forbid the lazy fix of blanket-disabling
the telemetry.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _ready_card(title="waiting") -> str:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title=title, assignee="a")
        conn.commit()
    return tid


def _result(*, guarded=()) -> kbd.DispatchResult:
    return kbd.DispatchResult(respawn_guarded=list(guarded))


# ---------------------------------------------------------------------------
# The gateway implementation (the live path)
# ---------------------------------------------------------------------------


def _run_gateway_ticks(monkeypatch, caplog, tick_result_fn, ticks):
    """Drive ``_kanban_dispatcher_watcher`` for ``ticks`` iterations.

    Only the spawn side is stubbed (``dispatch_once``). Board listing, the DB,
    ``guard_deferred_ids``, ``_KanbanDispatcher.ready_nonempty`` and
    ``has_spawnable_ready`` are all real, so the telemetry decision under test
    runs for real.
    """
    import gateway.kanban_watchers as kw

    runner = object.__new__(kw.GatewayKanbanWatchersMixin)
    runner._running = True
    monkeypatch.setattr(
        runner, "_kanban_dispatcher_boot",
        # interval=1 so one loop iteration consumes exactly one patched sleep.
        lambda: (lambda: {}, kb, {"dispatch_interval_seconds": 1}), raising=False,
    )
    monkeypatch.setattr(kbd, "dispatch_once", tick_result_fn)

    counter = {"n": 0}

    async def _direct(fn, *args):
        return fn(*args)

    async def _sleep(_delay):
        # The loop's boot delay burns the first sleep; the rest are one per tick.
        counter["n"] += 1
        if counter["n"] > ticks + 1:
            runner._running = False

    monkeypatch.setattr(kw, "_to_thread_process_service", _direct)
    monkeypatch.setattr(kw, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(kw, "_resolve_auto_decompose_settings", lambda load_config: (False, 0))
    monkeypatch.setattr(kbd, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(kw.asyncio, "sleep", _sleep)

    with caplog.at_level(logging.WARNING, logger=kw.logger.name):
        asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=30.0))
    return [r.getMessage() for r in caplog.records]


def test_gateway_guard_deferred_tick_is_not_stuck(
    kanban_home, monkeypatch, caplog, all_assignees_spawnable,
):
    """Only-guarded ready work + 0 spawns must NOT accumulate bad ticks."""
    tid = _ready_card()
    msgs = _run_gateway_ticks(
        monkeypatch, caplog,
        lambda conn, **kwargs: _result(guarded=[(tid, "active_pr")]),
        ticks=kbd_health_window() + 3,
    )
    assert not [m for m in msgs if "dispatcher stuck" in m], (
        f"a guard deferral is a healthy dispatcher deciding, not a stuck one; "
        f"got {msgs}"
    )


def test_gateway_spawnable_work_with_zero_spawns_still_warns(
    kanban_home, monkeypatch, caplog, all_assignees_spawnable,
):
    """Negative control: the telemetry must not be blanket-disabled."""
    _ready_card()
    msgs = _run_gateway_ticks(
        monkeypatch, caplog,
        lambda conn, **kwargs: _result(),  # nothing spawned, nothing guarded
        ticks=kbd_health_window() + 3,
    )
    assert [m for m in msgs if "dispatcher stuck" in m], (
        f"genuinely spawnable ready work with 0 spawns must still warn; got {msgs}"
    )


def test_gateway_other_spawnable_card_beside_a_guarded_one_still_warns(
    kanban_home, monkeypatch, caplog, all_assignees_spawnable,
):
    """Exclusion is per-card, not per-tick: one guarded card must not mask a
    second, genuinely spawnable one."""
    guarded = _ready_card(title="guarded")
    _ready_card(title="genuinely stuck")
    msgs = _run_gateway_ticks(
        monkeypatch, caplog,
        lambda conn, **kwargs: _result(guarded=[(guarded, "active_pr")]),
        ticks=kbd_health_window() + 3,
    )
    assert [m for m in msgs if "dispatcher stuck" in m], (
        f"a guarded card must not suppress the warning for OTHER spawnable "
        f"work; got {msgs}"
    )


def kbd_health_window() -> int:
    """The gateway's bad-tick window, read from the module (never frozen here)."""
    import gateway.kanban_watchers as kw
    return kw._HEALTH_WINDOW


# ---------------------------------------------------------------------------
# The CLI `--force` daemon implementation
# ---------------------------------------------------------------------------


# Comfortably more than the CLI daemon's private HEALTH_WINDOW, so the warning
# has every chance to fire in the negative controls.
_CLI_TICKS = 12


def _run_cli_ticks(monkeypatch, capsys, results, ticks):
    """Drive ``_cmd_daemon``'s real ``_on_tick`` closure ``ticks`` times."""
    from hermes_cli import kanban_ops

    captured = {}

    def fake_run_daemon(**kwargs):
        captured["on_tick"] = kwargs["on_tick"]

    monkeypatch.setattr(kbd, "run_daemon", fake_run_daemon)

    args = argparse.Namespace(
        force=True, interval=1, max=None, pidfile=None, verbose=False,
        failure_limit=2,
    )
    assert kanban_ops._cmd_daemon(args) == 0
    on_tick = captured["on_tick"]
    for _ in range(ticks):
        on_tick(results())
    return capsys.readouterr().err


def test_cli_daemon_guard_deferred_tick_is_not_stuck(
    kanban_home, monkeypatch, capsys, all_assignees_spawnable,
):
    """Only-guarded ready work + 0 spawns must NOT accumulate bad ticks."""
    tid = _ready_card()
    err = _run_cli_ticks(
        monkeypatch, capsys, lambda: _result(guarded=[(tid, "active_pr")]), ticks=_CLI_TICKS,
    )
    assert "dispatcher stuck" not in err, err


def test_cli_daemon_spawnable_work_with_zero_spawns_still_warns(
    kanban_home, monkeypatch, capsys, all_assignees_spawnable,
):
    """Negative control for the CLI counter."""
    _ready_card()
    err = _run_cli_ticks(monkeypatch, capsys, lambda: _result(), ticks=_CLI_TICKS)
    assert "dispatcher stuck" in err, err


def test_cli_daemon_other_spawnable_card_beside_a_guarded_one_still_warns(
    kanban_home, monkeypatch, capsys, all_assignees_spawnable,
):
    """Per-card exclusion for the CLI counter too."""
    guarded = _ready_card(title="guarded")
    _ready_card(title="genuinely stuck")
    err = _run_cli_ticks(
        monkeypatch, capsys,
        lambda: _result(guarded=[(guarded, "active_pr")]), ticks=_CLI_TICKS,
    )
    assert "dispatcher stuck" in err, err


# ---------------------------------------------------------------------------
# The shared derivation both implementations use
# ---------------------------------------------------------------------------


def test_guard_deferred_ids_accepts_every_tick_shape():
    """One helper, so the two implementations cannot drift."""
    one = _result(guarded=[("t_a", "active_pr")])
    two = _result(guarded=[("t_b", "recent_success")])
    assert kbd.guard_deferred_ids(one) == {"t_a"}
    assert kbd.guard_deferred_ids([one, two]) == {"t_a", "t_b"}
    # The gateway's multi-board shape: (slug, result) pairs, Nones included.
    assert kbd.guard_deferred_ids(
        [("main", one), ("other", None), ("third", two)]
    ) == {"t_a", "t_b"}
    assert kbd.guard_deferred_ids(None) == set()
    assert kbd.guard_deferred_ids([]) == set()


def test_has_spawnable_ready_honours_exclude_ids(kanban_home, all_assignees_spawnable):
    """The probe narrows per card and never turns itself off wholesale."""
    tid = _ready_card()
    other = _ready_card(title="second")
    with kbc.connect_closing() as conn:
        assert kbd.has_spawnable_ready(conn) is True
        assert kbd.has_spawnable_ready(conn, {tid}) is True  # `other` remains
        assert kbd.has_spawnable_ready(conn, {tid, other}) is False
