"""Comments hold lasting content; progress goes to the run, stage to a field.

Contract tests against a real temp HERMES_HOME and kanban DB, driven through
the CLI (``run_slash``) and the tool registry.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME",
                "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return h


def _create(title="card", **kw):
    with kbc.connect_closing() as conn:
        return kb.create_task(conn, title=title, **kw)


def _show(tid):
    return json.loads(kc.run_slash(f"show {tid} --json"))


def _tool(name, args):
    import tools.kanban_tools  # noqa: F401  (registers handlers)
    from tools.registry import registry
    return json.loads(registry.dispatch(name, args))


def _claimed_worker(monkeypatch):
    tid = _create(assignee="test-worker")
    with kbc.connect_closing() as conn:
        assert kb.claim_task(conn, tid) is not None
        run_id = kb._current_run_id(conn, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid, run_id


def test_stage_cli_sets_field_and_event_without_comment(home):
    tid = _create()
    kc.run_slash(f"stage {tid} review --note 'PR open at abc123'")
    out = _show(tid)
    assert out["task"]["current_step_key"] == "review"
    assert out["comments"] == []
    ev = [e for e in out["events"] if e["kind"] == "stage_set"]
    assert ev and ev[-1]["payload"]["to"] == "review"
    assert ev[-1]["payload"]["note"] == "PR open at abc123"
    assert "stage:     review" in kc.run_slash(f"show {tid}")
    with kbc.connect_closing() as conn:
        assert "Stage:    review" in kb.build_worker_context(conn, tid)


@pytest.mark.parametrize("body", [
    "SCHEDULED: EM: paused for priority #3",
    "Specified — updated body and promoted to todo.",
    "Unblocked: parents done",
    "UNBLOCK: parents done, dispatch now",
])
def test_status_comments_become_events(home, body):
    tid = _create()
    kc.run_slash(f"comment {tid} {shlex.quote(body)}")
    out = _show(tid)
    assert out["comments"] == []
    notes = [e for e in out["events"] if e["kind"] == "status_note"]
    assert notes and notes[-1]["payload"]["note"] == body
    assert "status_note" in kc.run_slash(f"show {tid}")


def test_stage_comment_becomes_stage_update(home):
    tid = _create()
    kc.run_slash(f"comment {tid} " + json.dumps("STAGE: development. pi delegate running"))
    out = _show(tid)
    assert out["comments"] == []
    assert out["task"]["current_step_key"] == "development"
    ev = [e for e in out["events"] if e["kind"] == "stage_set"][-1]
    assert ev["payload"]["note"] == "pi delegate running"


@pytest.mark.parametrize("body", [
    "STAGE: review\nR1: the lock is not held across the write",
    "SCHEDULED: waiting\nBLOCKER:DEP t_abc",
    "Specified — body\nDESIGN: one transaction per delete",
    "Unblocked\nEM DECISION: ship behind a flag",
    "STATUS: x\nACCEPTANCE: prod delete leaves no rows",
    "PROOF: curl -sf https://x/health",
    "NEEDS-DESIGN: split the lease table",
    "SCOPE: only the k8s path",
])
def test_lasting_markers_always_stored(home, body):
    tid = _create()
    kc.run_slash(f"comment {tid} {shlex.quote(body)}")
    assert [c["body"] for c in _show(tid)["comments"]] == [body]


def test_identical_comment_deduped_per_author(home):
    tid = _create()
    with kbc.connect_closing() as conn:
        a = kb.add_comment(conn, tid, "alice", "Root cause: lease not released")
        b = kb.add_comment(conn, tid, "alice", "Root cause: lease not released")
        c = kb.add_comment(conn, tid, "bob", "Root cause: lease not released")
    assert a == b and c != a
    assert len(_show(tid)["comments"]) == 2


def test_routing_disabled_by_config_stores_everything(home):
    (home / "config.yaml").write_text("kanban:\n  comment_routing: false\n")
    tid = _create()
    with kbc.connect_closing() as conn:
        kb.add_comment(conn, tid, "alice", "SCHEDULED: later")
        kb.add_comment(conn, tid, "alice", "SCHEDULED: later")
    assert len(_show(tid)["comments"]) == 2


def test_tool_comment_reports_routing(home, monkeypatch):
    tid, _ = _claimed_worker(monkeypatch)
    out = _tool("kanban_comment", {"task_id": tid, "body": "SCHEDULED: wait for slot"})
    assert out["ok"] and out["routed"] == "status" and out["comment_id"] is None
    out = _tool("kanban_comment", {"task_id": tid, "body": "EM DECISION: keep it"})
    assert out["comment_id"]
    assert len(_show(tid)["comments"]) == 1


def test_heartbeat_progress_on_run_and_in_next_worker_context(home, monkeypatch):
    with kbc.connect_closing() as conn:
        pass
    (home / "config.yaml").write_text("kanban:\n  progress_history_max: 3\n")
    tid, run_id = _claimed_worker(monkeypatch)
    for i in range(5):
        assert _tool("kanban_heartbeat", {"note": f"step {i}"})["ok"]
    assert _tool("kanban_heartbeat", {"note": "rebasing", "stage": "development"})["stage"] == "development"
    out = _show(tid)
    assert out["comments"] == []
    assert out["task"]["current_step_key"] == "development"
    run = [r for r in out["runs"] if r["id"] == run_id][0]
    meta = run["metadata"]
    assert meta["progress_note"] == "rebasing"
    assert [h["note"] for h in meta["progress_history"]] == ["step 3", "step 4", "rebasing"]
    # Run ends; the next worker sees the latest progress note of the prior run.
    with kbc.connect_closing() as conn:
        assert kb.block_task(conn, tid, reason="need input")
        ctx = kb.build_worker_context(conn, tid)
    assert "_latest progress_" in ctx and "rebasing" in ctx
    assert "progress_history" not in ctx
    assert "progress: rebasing" in kc.run_slash(f"show {tid}")
