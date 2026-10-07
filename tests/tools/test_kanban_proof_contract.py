"""kanban_complete on a card with a PROOF: line requires the pasted proof run.

Exercised through the real tool registry against a real temp HERMES_HOME
kanban DB, the same path a dispatcher-owned worker takes.
"""
import json

import pytest

PROOF_BODY = "Fix the widget.\n\nPROOF: `curl -sf https://example.test/health`\n"


def _worker_card(monkeypatch, tmp_path, body):
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="widget", body=body, assignee="test-worker")
        assert kb.claim_task(conn, tid) is not None
        run_id = kb._current_run_id(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


def _complete(tid, metadata=None):
    import tools.kanban_tools  # noqa: F401  (registers handlers)
    from tools.registry import registry
    args = {"task_id": tid, "summary": "review passed, shipped"}
    if metadata is not None:
        args["metadata"] = metadata
    return json.loads(registry.dispatch("kanban_complete", args))


def _status(tid):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        return kb.get_task(conn, tid).status
    finally:
        conn.close()


GOOD = {"command": "$ curl -sf  https://example.test/health",
        "output": '{"ok": true}', "ran_at": "2026-10-07T12:00:00-07:00"}


@pytest.mark.parametrize("metadata, needle", [
    (None, "metadata.proof is missing"),
    ({"published_pr": "https://github.com/o/r/pull/1"}, "metadata.proof is missing"),
    ({"proof": {**GOOD, "output": "   "}}, "output"),
    ({"proof": {k: v for k, v in GOOD.items() if k != "ran_at"}}, "ran_at"),
    ({"proof": {**GOOD, "command": "pytest -q"}}, "does not match"),
])
def test_proof_card_refused_without_valid_proof(monkeypatch, tmp_path, metadata, needle):
    tid = _worker_card(monkeypatch, tmp_path, PROOF_BODY)
    out = _complete(tid, metadata)
    assert needle in out["error"]
    assert "BLOCKER:EXTERNAL" in out["error"]
    assert "curl -sf https://example.test/health" in out["error"]
    assert _status(tid) == "running"


def test_proof_card_completes_with_matching_proof(monkeypatch, tmp_path):
    tid = _worker_card(monkeypatch, tmp_path, PROOF_BODY)
    out = _complete(tid, {"proof": GOOD})
    assert out.get("ok") is True, out
    assert _status(tid) == "done"


def test_card_without_proof_line_unchanged(monkeypatch, tmp_path):
    tid = _worker_card(monkeypatch, tmp_path, "Just fix it. Proof of concept welcome.")
    out = _complete(tid)
    assert out.get("ok") is True, out
    assert _status(tid) == "done"
