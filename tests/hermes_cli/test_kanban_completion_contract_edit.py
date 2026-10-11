"""`hermes kanban edit --completion-contract`: re-point after a repo rename."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task(contract="old-owner/OldRepo", **kw):
    with kbc.connect_closing() as conn:
        return kb.create_task(conn, title="t", completion_contract=contract, **kw)


def _get(tid):
    with kbc.connect_closing() as conn:
        return kb.get_task(conn, tid), kb.list_comments(conn, tid), kb.list_events(conn, tid)


def test_repoint_records_event_and_comment(kanban_home):
    tid = _task()
    out = kc.run_slash(f"edit {tid} --completion-contract NewOrg/NewRepo --reason renamed")
    assert "old-owner/OldRepo -> NewOrg/NewRepo" in out
    task, comments, events = _get(tid)
    assert task.completion_contract == "NewOrg/NewRepo"
    assert any("old-owner/OldRepo -> NewOrg/NewRepo" in c.body and "renamed" in c.body for c in comments)
    ev = [e for e in events if e.kind == "completion_contract_changed"]
    assert ev and ev[-1].payload["old"] == "old-owner/OldRepo" and ev[-1].payload["new"] == "NewOrg/NewRepo"


def test_repoint_accepts_pr_url_and_local_only(kanban_home):
    tid = _task()
    kc.run_slash(f"edit {tid} --completion-contract https://github.com/NewOrg/NewRepo/pull/5")
    assert _get(tid)[0].completion_contract == "https://github.com/NewOrg/NewRepo/pull/5"
    kc.run_slash(f"edit {tid} --completion-contract local-only")
    assert _get(tid)[0].completion_contract == "local-only"


def test_repoint_rejects_invalid_contract(kanban_home):
    tid = _task()
    out = kc.run_slash(f"edit {tid} --completion-contract 'not a repo'")
    assert "completion_contract must be" in out
    assert _get(tid)[0].completion_contract == "old-owner/OldRepo"


def test_repoint_refuses_done_task(kanban_home):
    tid = _task("local-only")
    with kbc.connect_closing() as conn:
        assert kb.complete_task(conn, tid, result="ok")
        assert kb.set_completion_contract(conn, tid, "NewOrg/NewRepo", author="x") is None


def test_repoint_noop_does_not_comment(kanban_home):
    tid = _task("NewOrg/NewRepo")
    with kbc.connect_closing() as conn:
        res = kb.set_completion_contract(conn, tid, "NewOrg/NewRepo", author="x")
    assert res == {"old": "NewOrg/NewRepo", "new": "NewOrg/NewRepo", "changed": False}
    assert _get(tid)[1] == []


def test_repoint_from_workspace_git_remote(kanban_home, tmp_path):
    repo = tmp_path / "ws"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin",
                    "git@github.com:NewOrg/NewRepo.git"], check=True)
    tid = _task(workspace_kind="dir", workspace_path=str(repo))
    out = kc.run_slash(f"edit {tid} --completion-contract workspace")
    assert "-> NewOrg/NewRepo" in out
    assert _get(tid)[0].completion_contract == "NewOrg/NewRepo"


def test_repoint_cannot_combine_with_other_edits(kanban_home):
    tid = _task()
    out = kc.run_slash(f"edit {tid} --completion-contract NewOrg/NewRepo --title x")
    assert "cannot be combined" in out
