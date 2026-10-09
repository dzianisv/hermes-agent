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


def _events(tid, kind):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        return [e for e in kb.list_events(conn, tid) if e.kind == kind]
    finally:
        conn.close()


def test_proof_cmd_failure_refused_with_harness_exit(monkeypatch, tmp_path):
    tid = _worker_card(monkeypatch, tmp_path, "Fix it.\n\nPROOF-CMD: false\n")
    out = _complete(tid, {"proof": {"command": "false", "output": "PASS (lie)", "ran_at": "x"}})
    assert "exited 1" in out["error"], out
    assert _status(tid) == "running"
    ev = _events(tid, "proof_run")
    assert ev and ev[-1].payload["proof_run"][0]["exit_code"] == 1
    assert ev[-1].payload["proof_run"][0]["runner"] == "harness"
    assert ev[-1].payload["passed"] is False


def test_proof_cmd_pass_completes_with_recorded_output(monkeypatch, tmp_path):
    tid = _worker_card(monkeypatch, tmp_path, "Fix it.\n\nPROOF-CMD: `echo PASS`\n")
    out = _complete(tid)
    assert out.get("ok") is True, out
    assert _status(tid) == "done"
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        rec = kb.latest_run(conn, tid).metadata["proof_run"][0]
    finally:
        conn.close()
    assert rec["exit_code"] == 0 and rec["output_tail"].strip() == "PASS"


def test_proof_cmd_timeout_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_KANBAN_PROOF_TIMEOUT", "1")
    tid = _worker_card(monkeypatch, tmp_path, "PROOF-CMD: sleep 5\n")
    out = _complete(tid)
    assert "timed out" in out["error"]
    assert _status(tid) == "running"


def test_async_proof_parks_in_review_then_rerun_completes(monkeypatch, tmp_path):
    flag = tmp_path / "live"
    tid = _worker_card(monkeypatch, tmp_path, f"Deploy.\n\nPROOF-CMD-ASYNC: test -f {flag}\n")
    out = _complete(tid)
    assert out.get("proof_async_pending") is True, out
    assert _status(tid) == "review"
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_proof_contract import rerun_pending_proofs
    conn = kbc.connect()
    try:
        assert rerun_pending_proofs(conn)[0]["passed"] is False
    finally:
        conn.close()
    assert _status(tid) == "review"
    flag.write_text("up")
    conn = kbc.connect()
    try:
        assert rerun_pending_proofs(conn)[0]["passed"] is True
        assert rerun_pending_proofs(conn) == []
    finally:
        conn.close()
    assert _status(tid) == "done"


def test_create_warns_on_impl_card_without_proof(monkeypatch, tmp_path):
    _worker_card(monkeypatch, tmp_path, "seed")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    import tools.kanban_tools  # noqa: F401
    from tools.registry import registry
    out = json.loads(registry.dispatch("kanban_create", {
        "title": "Fix login bug", "body": "do it", "assignee": "w"}))
    assert out["ok"] is True and "PROOF-CMD" in out["warnings"][0]
    out = json.loads(registry.dispatch("kanban_create", {
        "title": "Fix login bug", "body": "PROOF-CMD: true", "assignee": "w",
        "allow_duplicate": True}))
    assert "warnings" not in out
