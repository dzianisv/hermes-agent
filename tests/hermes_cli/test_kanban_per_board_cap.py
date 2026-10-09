"""Per-board running-card cap: ``kanban.boards.<slug>.max_in_progress``.

Real temp sqlite kanban DBs; the only stand-in is the worker spawn callable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    prof = home / "profiles" / "alpha"
    prof.mkdir(parents=True)
    (prof / "config.yaml").write_text("{}\n")  # identity marker
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "normal")
    kb.init_db()
    return home


def _write_config(home: Path, kanban: dict) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump({"kanban": kanban}))
    from hermes_cli import config as cfg
    for name in ("_invalidate_config_cache", "invalidate_config_cache", "clear_config_cache"):
        fn = getattr(cfg, name, None)
        if callable(fn):
            fn()


def _spawn(spawns):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 4242
    return fake_spawn


def _seed(board, n):
    with kbc.connect_closing(board=board) as conn:
        for i in range(n):
            kb.create_task(conn, title=f"{board}-{i}", assignee="alpha")


def _tick(board, spawns, **kw):
    with kbc.connect_closing(board=board) as conn:
        return kbd.dispatch_once(conn, spawn_fn=_spawn(spawns), board=board,
                                 reconcile_orphans=False, **kw)


def _running(board):
    with kbc.connect_closing(board=board) as conn:
        return kbd.count_running_tasks(conn)


def test_board_cap_limits_running_cards_across_ticks(kanban_home):
    kb.create_board("proj-a")
    _write_config(kanban_home, {"boards": {"proj-a": {"max_in_progress": 2}}})
    _seed("proj-a", 5)
    spawns: list = []
    _tick("proj-a", spawns)
    assert _running("proj-a") == 2
    _tick("proj-a", spawns)  # still 2 running -> no new spawn
    assert _running("proj-a") == 2
    assert len(spawns) == 2


def test_board_cap_is_per_board_and_falls_back_to_global(kanban_home):
    kb.create_board("proj-a")
    kb.create_board("proj-b")
    _write_config(kanban_home, {
        "max_in_progress": 12,
        "boards": {"proj-a": {"max_in_progress": 1}},
    })
    _seed("proj-a", 4)
    _seed("proj-b", 4)
    spawns: list = []
    _tick("proj-a", spawns, max_in_progress=12)
    _tick("proj-b", spawns, max_in_progress=12)
    assert _running("proj-a") == 1
    assert _running("proj-b") == 4  # no per-board key -> global 12 only


def test_board_cap_tighter_than_global_wins_and_global_still_applies(kanban_home):
    kb.create_board("proj-a")
    _write_config(kanban_home, {"boards": {"proj-a": {"max_in_progress": 10}}})
    _seed("proj-a", 6)
    spawns: list = []
    _tick("proj-a", spawns, max_in_progress=3)
    assert _running("proj-a") == 3


def test_invalid_board_cap_ignored(kanban_home):
    kb.create_board("proj-a")
    _write_config(kanban_home, {"boards": {"proj-a": {"max_in_progress": 0}}})
    assert kbd.configured_board_max_in_progress("proj-a") is None
    _write_config(kanban_home, {"boards": {"proj-a": {"max_in_progress": 2}}})
    assert kbd.configured_board_max_in_progress("proj-a") == 2
    assert kbd.configured_board_max_in_progress("other") is None
