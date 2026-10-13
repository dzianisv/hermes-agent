"""A review worker keeps the terminal tool so it can run ``git diff`` itself.

The plain review prompt ("work kanban task <id>") relies on the reviewer
inspecting the workspace on its own; nothing on the review path may strip
the ``terminal`` toolset from the worker's ``--toolsets`` pin.
"""

from __future__ import annotations

from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli.kanban_db import Task


def _review_task() -> Task:
    return Task(
        id="t_rev1", title="review it", body="ACCEPTANCE: tests pass", assignee="reviewer",
        status="running", priority=0, created_by=None, created_at=1, started_at=None,
        completed_at=None, workspace_kind="scratch", workspace_path=None,
        claim_lock=None, claim_expires=None, tenant=None,
    )


def test_review_worker_profile_resolves_terminal_toolset(tmp_path):
    home = tmp_path / "profiles" / "reviewer"
    home.mkdir(parents=True)
    toolsets = dispatch._resolve_worker_cli_toolsets(str(home))
    assert toolsets and "terminal" in toolsets


def test_review_worker_argv_is_plain_prompt_with_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "_resolve_hermes_argv", lambda: ["hermes"])
    home = tmp_path / "profiles" / "reviewer"
    home.mkdir(parents=True)
    argv = dispatch._worker_argv(_review_task(), "reviewer", str(home))
    assert "terminal" in argv[argv.index("--toolsets") + 1].split(",")
    assert argv[-2:] == ["-q", "work kanban task t_rev1"]
