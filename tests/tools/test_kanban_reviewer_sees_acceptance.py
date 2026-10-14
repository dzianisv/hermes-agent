"""Row #11: the reviewer's first prompt is just `work kanban task {id}`, but its
system prompt (KANBAN_GUIDANCE) makes kanban_show() the first step, and
kanban_show returns the card body with DESIGN/ACCEPTANCE. No brief needed."""
from __future__ import annotations

import json

from agent.prompt_builder import KANBAN_GUIDANCE


def test_guidance_orients_with_kanban_show_first():
    assert "1. **Orient.** Call `kanban_show()` first" in KANBAN_GUIDANCE


def test_reviewer_kanban_show_returns_acceptance(monkeypatch, tmp_path):
    from pathlib import Path
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "verifier")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    body = "Fix X.\nDESIGN: use the existing retry ladder\nACCEPTANCE: endpoint returns 200"
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="card", body=body, assignee="verifier")
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    from tools import kanban_tools as kt
    out = json.loads(kt._handle_show({}))
    assert "ACCEPTANCE: endpoint returns 200" in out["task"]["body"]
    assert "ACCEPTANCE: endpoint returns 200" in out["worker_context"]
    assert "DESIGN: use the existing retry ladder" in out["worker_context"]
