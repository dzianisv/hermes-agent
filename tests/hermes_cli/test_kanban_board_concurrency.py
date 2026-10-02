"""Per-board ``max_concurrency`` (board.json) caps live workers on that board only.

The cap is folded into ``max_spawn`` (the existing live running+spawn cap) on
every dispatch entry point: the gateway's ``tick_once_for_board``, ``hermes
kanban dispatch`` and the standalone daemon. It can only tighten.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from gateway.kanban_watchers_dispatcher import _DispatcherSettings, _KanbanDispatcher
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_ops

_WORKTREE = Path(__file__).resolve().parents[2]


@pytest.fixture
def boards(tmp_path, monkeypatch, all_assignees_spawnable):
    """Isolated kanban root with two boards; spawns recorded instead of run."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT",
                "HERMES_KANBAN_HOME", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    kb._INITIALIZED_PATHS.clear()
    # Host memory must not be what limits spawns in these tests.
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda sample=None: "ok")
    spawns: list[tuple[str, str]] = []

    def fake_spawn(task, workspace, *, board=None):
        # CLI/daemon paths pass no board; titles carry it ("<slug>-<i>").
        spawns.append((task.title.rsplit("-", 1)[0], task.id))
        return 4242

    monkeypatch.setattr(kbd, "_default_spawn", fake_spawn)
    kb.create_board("capped")
    kb.create_board("other")
    for slug in ("capped", "other"):
        with kbc.connect_closing(board=slug) as conn:
            for i in range(3):
                kb.create_task(conn, title=f"{slug}-{i}", assignee="alice")
    return spawns


def _gateway(max_spawn=None) -> _KanbanDispatcher:
    settings = _DispatcherSettings(
        interval=60.0, max_spawn=max_spawn, max_in_progress=None, failure_limit=2,
        stale_timeout_seconds=0, reconcile_orphans=False, default_assignee=None,
        max_in_progress_per_profile=None,
    )
    return _KanbanDispatcher(kb, settings)


def _spawned_on(spawns, slug):
    return [tid for board, tid in spawns if board == slug]


def test_gateway_cap_one_spawns_one_then_zero_other_board_unaffected(boards):
    kb.write_board_metadata("capped", max_concurrency=1)
    disp = _gateway()

    disp.tick_once_for_board("capped")
    disp.tick_once_for_board("other")
    assert len(_spawned_on(boards, "capped")) == 1
    assert len(_spawned_on(boards, "other")) == 3

    # One worker is now running on the capped board: the next tick spawns nothing.
    disp.tick_once_for_board("capped")
    assert len(_spawned_on(boards, "capped")) == 1


def test_board_cap_only_tightens_max_spawn(boards):
    kb.write_board_metadata("capped", max_concurrency=2)
    kb.write_board_metadata("other", max_concurrency=5)
    disp = _gateway(max_spawn=1)
    disp.tick_once_for_board("capped")
    disp.tick_once_for_board("other")
    # Profile-wide max_spawn=1 is tighter than both board caps.
    assert len(_spawned_on(boards, "capped")) == 1
    assert len(_spawned_on(boards, "other")) == 1


@pytest.mark.parametrize("bad", ["lots", -3, 0, True, None, [1]])
def test_bad_board_cap_means_no_cap(boards, bad):
    path = kb.board_metadata_path("capped")
    meta = json.loads(path.read_text())
    meta["max_concurrency"] = bad
    path.write_text(json.dumps(meta))
    _gateway().tick_once_for_board("capped")
    assert len(_spawned_on(boards, "capped")) == 3


def test_cli_dispatch_honours_board_cap(boards, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"kanban": {"max_in_progress": 50}})
    kb.write_board_metadata("capped", max_concurrency=1)
    args = argparse.Namespace(dry_run=False, max=None, failure_limit=2, json=True)
    for _ in range(2):
        with kb.scoped_current_board("capped"):
            kanban_ops._cmd_dispatch(args)
    with kb.scoped_current_board("other"):
        kanban_ops._cmd_dispatch(args)
    assert len(_spawned_on(boards, "capped")) == 1
    assert len(_spawned_on(boards, "other")) == 3


def test_daemon_honours_board_cap(boards, monkeypatch):
    kb.set_current_board("capped")
    kb.write_board_metadata("capped", max_concurrency=1)
    monkeypatch.setattr(kbd, "configured_max_in_progress", lambda: 50)
    stop = threading.Event()
    ticks = []

    def on_tick(res):
        ticks.append(res)
        if len(ticks) >= 2:
            stop.set()

    kbd.run_daemon(interval=0.01, stop_event=stop, on_tick=on_tick)
    assert len(_spawned_on(boards, "capped")) == 1


def _cli(args, home):
    env = dict(os.environ, PYTHONPATH=str(_WORKTREE), HERMES_HOME=str(home))
    env.pop("HERMES_KANBAN_HOME", None)
    env.pop("HERMES_KANBAN_BOARD", None)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban", *args],
        env=env, capture_output=True, text=True, cwd=str(_WORKTREE), timeout=60,
    )


def test_cli_set_and_clear_round_trip(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    assert _cli(["boards", "create", "content-machine"], home).returncode == 0

    r = _cli(["boards", "set-concurrency", "content-machine", "1"], home)
    assert r.returncode == 0, r.stderr
    listed = {b["slug"]: b for b in json.loads(_cli(["boards", "list", "--json"], home).stdout)}
    assert listed["content-machine"]["max_concurrency"] == 1
    assert listed["default"]["max_concurrency"] is None
    assert "max 1 at once" in _cli(["boards", "list"], home).stdout

    r = _cli(["boards", "set-concurrency", "content-machine", "0"], home)
    assert r.returncode == 0, r.stderr
    listed = {b["slug"]: b for b in json.loads(_cli(["boards", "list", "--json"], home).stdout)}
    assert listed["content-machine"]["max_concurrency"] is None
    assert "at once" not in _cli(["boards", "list"], home).stdout

    assert _cli(["boards", "set-concurrency", "content-machine", "-1"], home).returncode != 0
    assert _cli(["boards", "set-concurrency", "nope", "1"], home).returncode != 0


def test_cap_counts_worker_still_alive_after_its_card_went_done(boards, monkeypatch):
    """running -> done clears ``tasks.worker_pid`` while the process may still be
    exiting (reaper grace window). That process still holds the board's slot."""
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        monkeypatch.setattr(kbd, "_default_spawn",
                            lambda task, ws, *, board=None: boards.append(("capped", task.id)) or worker.pid)
        kb.write_board_metadata("capped", max_concurrency=1)
        disp = _gateway()
        disp.tick_once_for_board("capped")
        assert len(_spawned_on(boards, "capped")) == 1
        first = _spawned_on(boards, "capped")[0]

        with kbc.connect_closing(board="capped") as conn:
            assert kb.complete_task(conn, first, summary="done", force=True)
            assert conn.execute("SELECT status, worker_pid FROM tasks WHERE id = ?",
                                (first,)).fetchone()[:] == ("done", None)

        # Card is done but its worker is alive: still one live worker, no new spawn.
        disp.tick_once_for_board("capped")
        assert len(_spawned_on(boards, "capped")) == 1

        worker.kill()
        worker.wait(timeout=10)
        # Process gone: the slot frees on the next tick.
        disp.tick_once_for_board("capped")
        assert len(_spawned_on(boards, "capped")) == 2
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=10)


def test_board_json_write_is_atomic(boards, monkeypatch):
    path = kb.board_metadata_path("capped")
    calls = []
    real_replace = os.replace

    def spy_replace(src, dst, *a, **kw):
        calls.append((Path(src).parent, Path(dst)))
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", spy_replace)
    kb.write_board_metadata("capped", max_concurrency=2)
    assert (path.parent, path) in calls
    assert json.loads(path.read_text())["max_concurrency"] == 2


def test_unreadable_board_json_never_lifts_the_cap(boards, monkeypatch):
    path = kb.board_metadata_path("capped")
    kb._LAST_BOARD_CAPS.clear()

    # Fresh process, torn file, no cap seen yet: spawn nothing rather than run uncapped.
    path.write_text('{"max_concurrency": 1, "na')
    _gateway().tick_once_for_board("capped")
    assert _spawned_on(boards, "capped") == []

    # A valid cap seen once is kept while the file is torn/unreadable.
    kb.write_board_metadata("capped", max_concurrency=2)
    assert kb.board_max_concurrency("capped") == 2
    path.write_text("{")
    assert kb.board_capped_max_spawn(None, "capped") == 2
    monkeypatch.setattr(kb, "_read_board_json", lambda slug: (_ for _ in ()).throw(OSError("EIO")))
    assert kb.board_capped_max_spawn(10, "capped") == 2
    for _ in range(2):
        _gateway().tick_once_for_board("capped")
    assert len(_spawned_on(boards, "capped")) == 2


def test_real_board_flag_parser_routes_dispatch_to_the_capped_board(boards, monkeypatch):
    from hermes_cli import kanban as kc

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"kanban": {"max_in_progress": 50}})
    kb.write_board_metadata("capped", max_concurrency=1)
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    kc.build_parser(parser.add_subparsers(dest="command"))
    for slug in ("capped", "capped", "other"):
        args = parser.parse_args(["kanban", "--board", slug, "dispatch", "--json"])
        assert kc.kanban_command(args) == 0
    assert kb.get_current_board() == kb.DEFAULT_BOARD
    assert len(_spawned_on(boards, "capped")) == 1
    assert len(_spawned_on(boards, "other")) == 3
