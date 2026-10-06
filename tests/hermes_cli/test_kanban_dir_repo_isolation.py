"""Issue #25: a ``dir`` card pointing at a shared git checkout must not run in it.

Real data (board ``vibebrowser``): card t_c76a8f89 was created with
``workspace_kind='dir'``, ``workspace_path='/Users/engineer/workspace/vibebrowser/vibe'``
— the software-engineer profile's ``terminal.cwd`` and the main checkout of the
repo. ``resolve_workspace`` used ``dir`` paths verbatim, so parallel cards
edited the same tree. A git-repo workspace must always become a per-card
worktree, and the dispatcher must refuse paths another running card uses or a
profile's shared default checkout.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.name=T", "-c", "user.email=t@e",
         "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _make_repo(tmp_path: Path, name: str = "vibe") -> Path:
    repo = tmp_path / "workspace" / "vibebrowser" / name
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    (repo / "README.md").write_text("base\n")
    (repo / "src").mkdir()
    (repo / "src" / "a.txt").write_text("a\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    return repo.resolve()


def _dir_task(conn, repo: Path, title: str) -> str:
    return kb.create_task(conn, title=title, assignee="software-engineer",
                          workspace_kind="dir", workspace_path=str(repo))


def test_dir_card_on_repo_checkout_gets_own_worktree(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    with kbc.connect() as conn:
        a = kb.get_task(conn, _dir_task(conn, repo, "card a"))
        b = kb.get_task(conn, _dir_task(conn, repo, "card b"))
    wa = kbw.resolve_workspace(a)
    wb = kbw.resolve_workspace(b)
    assert wa.resolve() != repo and wb.resolve() != repo
    assert wa != wb
    # Under the managed per-card root: workspaces/<id>/<repo-name>
    assert wa.parts[-2:] == (a.id, "vibe")
    assert _git(wa, "branch", "--show-current") == f"wt/{a.id}"
    # Shared checkout untouched; still on main.
    assert _git(repo, "branch", "--show-current") == "main"
    # Idempotent: the persisted worktree path is reused as-is.
    a.workspace_path = str(wa)
    assert kbw.resolve_workspace(a) == wa


def test_dir_card_on_repo_subdir_maps_into_worktree(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    with kbc.connect() as conn:
        t = kb.get_task(conn, _dir_task(conn, repo / "src", "subdir card"))
    ws = kbw.resolve_workspace(t)
    assert ws.name == "src" and (ws / "a.txt").exists()
    assert repo not in ws.parents


def test_dir_card_outside_git_is_unchanged(kanban_home, tmp_path):
    plain = tmp_path / "notes"
    with kbc.connect() as conn:
        t = kb.get_task(conn, _dir_task(conn, plain, "plain"))
    assert kbw.resolve_workspace(t) == plain


def test_dispatcher_refuses_workspace_of_other_running_card(kanban_home, tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    with kbc.connect() as conn:
        other = _dir_task(conn, shared, "running one")
        conn.execute("UPDATE tasks SET status='running' WHERE id=?", (other,))
        conn.commit()
        mine = _dir_task(conn, shared, "mine")
        reason = kbd._workspace_conflict(conn, mine, shared, "software-engineer")
    assert reason and other in reason


def test_dispatcher_refuses_profile_default_checkout(kanban_home, tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.setattr(kbd, "_profile_default_cwd", lambda name: shared)
    with kbc.connect() as conn:
        mine = _dir_task(conn, shared, "mine")
        reason = kbd._workspace_conflict(conn, mine, shared, "software-engineer")
        ok = kbd._workspace_conflict(conn, mine, tmp_path / "elsewhere", "software-engineer")
    assert reason and "default" in reason
    assert ok is None
