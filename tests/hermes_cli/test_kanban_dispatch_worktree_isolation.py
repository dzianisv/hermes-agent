"""Contract: a dispatched card on a git-repo board never runs in the shared checkout.

Boards whose ``default_workdir`` is a git repo hand every ``dir``/``worktree``
card that repo path at create time. Dispatch must give each card its own linked
worktree; the main checkout is never the spawn cwd.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.name=T", "-c", "user.email=t@e.x",
         "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACES_ROOT", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    repo = tmp_path / "src" / "vibe"
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    (repo / "README.md").write_text("x\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    kb.create_board("repo-board", default_workdir=str(repo))
    kb.init_db(board="repo-board")
    return repo


def _dispatch(board: str):
    spawned: dict[str, str] = {}

    def spawn(task, workspace, board=None):
        spawned[task.id] = workspace
        return None

    with kbc.connect(board=board) as conn:
        kbd.dispatch_once(conn, spawn_fn=spawn, board=board)
    return spawned


def _row(board: str, tid: str):
    with kbc.connect(board=board) as conn:
        return conn.execute(
            "SELECT workspace_kind, workspace_path, branch_name FROM tasks WHERE id=?", (tid,)
        ).fetchone()


@pytest.mark.parametrize("kind", ["dir", "worktree"])
def test_board_repo_cards_get_distinct_worktrees(env, kind):
    repo = env
    with kbc.connect(board="repo-board") as conn:
        a = kb.create_task(conn, title="a", assignee="default", workspace_kind=kind, board="repo-board")
        b = kb.create_task(conn, title="b", assignee="default", workspace_kind=kind, board="repo-board")
    spawned = _dispatch("repo-board")
    assert set(spawned) == {a, b}
    main = repo.resolve()
    paths = {Path(p).resolve() for p in spawned.values()}
    assert len(paths) == 2
    for tid, ws in spawned.items():
        ws = Path(ws).resolve()
        assert ws != main
        assert Path(_git(ws, "rev-parse", "--path-format=absolute", "--git-common-dir")).resolve() == main / ".git"
        assert Path(_git(ws, "rev-parse", "--path-format=absolute", "--git-dir")).resolve() != main / ".git"
        assert _git(ws, "branch", "--show-current") == f"wt/{tid}"
        row = _row("repo-board", tid)
        assert row["workspace_kind"] == "worktree"
        assert Path(row["workspace_path"]).resolve() == ws
    assert _git(repo, "branch", "--show-current") == "main"


def test_dir_card_lands_under_kanban_workspaces(env):
    repo = env
    with kbc.connect(board="repo-board") as conn:
        t = kb.create_task(conn, title="t", assignee="default", workspace_kind="dir", board="repo-board")
    ws = Path(_dispatch("repo-board")[t]).resolve()
    assert ws == (kb.workspaces_root(board="repo-board") / t / repo.name).resolve()


def test_dir_card_on_repo_subdir_is_isolated(env):
    repo = env
    (repo / "pkg").mkdir()
    with kbc.connect(board="repo-board") as conn:
        t = kb.create_task(conn, title="t", assignee="default", workspace_kind="dir",
                           workspace_path=str(repo / "pkg"), board="repo-board")
    ws = Path(_dispatch("repo-board")[t]).resolve()
    assert not ws.is_relative_to(repo.resolve())


def test_dir_card_on_non_git_dir_unchanged(env, tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with kbc.connect(board="repo-board") as conn:
        t = kb.create_task(conn, title="t", assignee="default", workspace_kind="dir",
                           workspace_path=str(plain), board="repo-board")
    assert Path(_dispatch("repo-board")[t]).resolve() == plain.resolve()
    assert _row("repo-board", t)["workspace_kind"] == "dir"


def test_worktree_card_on_repo_subdir_is_isolated(env):
    repo = env
    (repo / "pkg").mkdir()
    with kbc.connect(board="repo-board") as conn:
        t = kb.create_task(conn, title="t", assignee="default", workspace_kind="worktree",
                           workspace_path=str(repo / "pkg"), board="repo-board")
    ws = Path(_dispatch("repo-board")[t]).resolve()
    assert ws != repo.resolve() and ws != (repo / "pkg").resolve()
    assert _git(ws, "branch", "--show-current") == f"wt/{t}"
