"""Notion harness #13: product-outcome cards need a live-validation receipt.

Body shape taken from the real board card t_ec823387 (Telegram alert path),
which went ``done`` on review although its acceptance demanded live proof.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

REPO = "den/vibe"
MERGED = "a" * 40
LATER = "b" * 40
RUN_URL = f"https://github.com/{REPO}/actions/runs/777"
PR_URL = f"https://github.com/{REPO}/pull/42"
BODY = (
    "The repo-wide Telegram alert credentials resolve to a bot that is BLOCKED by the recipient.\n"
    "2. Prove delivery end to end from one GitHub workflow consumer, not a green run.\n"
    "PROOF: integration-telegram.yml must pass on the merged commit\n"
)
CFG = {"kanban": {"live_receipt": {"workflows": ["integration-telegram.yml"]}}}


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    import hermes_cli.config as config
    monkeypatch.setattr(config, "load_config", lambda *a, **k: CFG)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _gh(run=None, merged_at="2026-10-01T10:00:00Z", compare="ahead"):
    run = {"conclusion": "success", "status": "completed", "headSha": MERGED,
           "workflowName": "Integration Telegram", "path": ".github/workflows/integration-telegram.yml",
           "createdAt": "2026-10-01T10:05:00Z", "url": RUN_URL, "event": "workflow_dispatch", **(run or {})}
    calls = []

    def fake(args):
        calls.append(args)
        if args[:2] == ["pr", "view"]:
            return {"state": "MERGED", "mergedAt": merged_at, "mergeCommit": {"oid": MERGED}}
        if args[:2] == ["run", "view"]:
            return run
        if args[0] == "api" and "/compare/" in args[1]:
            return {"status": compare}
        raise AssertionError(args)
    fake.calls = calls
    return fake


def _task(conn, body=BODY):
    tid = kb.create_task(conn, title="Telegram alert path delivers", assignee="software-engineer", body=body)
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    return tid


def _state(conn, tid):
    r = conn.execute("SELECT status, assignee FROM tasks WHERE id=?", (tid,)).fetchone()
    return r["status"], r["assignee"]


def _complete(conn, tid, **md):
    return kb.complete_task(conn, tid, summary="PR approved and merged", metadata=md or None, force=True)


@pytest.fixture
def gh(monkeypatch):
    from hermes_cli import kanban_live_receipt as lr
    holder = {"fn": _gh()}
    monkeypatch.setattr(lr, "GH_RUNNER", lambda args: holder["fn"](args))
    return holder


def test_review_pass_without_receipt_is_rejected(conn, gh):
    tid = _task(conn)
    with pytest.raises(ValueError, match="live-validation receipt"):
        _complete(conn, tid, published_pr=PR_URL)
    assert _state(conn, tid) == ("running", "software-engineer")


@pytest.mark.parametrize("run,compare,match", [
    ({"conclusion": "skipped"}, "ahead", "skipped"),
    ({"conclusion": "failure"}, "ahead", "conclusion=failure"),
    ({"createdAt": "2026-10-01T09:00:00Z"}, "ahead", "stale"),
    ({"path": ".github/workflows/lint.yml", "workflowName": "Lint"}, "ahead", "wrong target"),
    ({"headSha": LATER}, "behind", "does not contain merged"),
])
def test_bad_receipts_rejected_owner_kept(conn, gh, run, compare, match):
    gh["fn"] = _gh(run=run, compare=compare)
    tid = _task(conn)
    with pytest.raises(ValueError, match=match):
        _complete(conn, tid, published_pr=PR_URL, live_receipt=RUN_URL)
    assert _state(conn, tid) == ("running", "software-engineer")
    kinds = [r[0] for r in conn.execute("SELECT kind FROM task_events WHERE task_id=?", (tid,))]
    assert "completion_blocked_live_receipt" in kinds and "completed" not in kinds


def test_wrong_repo_receipt_rejected(conn, gh):
    tid = _task(conn)
    with pytest.raises(ValueError, match="wrong target"):
        _complete(conn, tid, published_pr=PR_URL, live_receipt="https://github.com/other/repo/actions/runs/1")


def test_valid_receipt_completes(conn, gh):
    tid = _task(conn)
    assert _complete(conn, tid, published_pr=PR_URL, live_receipt=RUN_URL)
    assert _state(conn, tid)[0] == "done"
    kinds = [r[0] for r in conn.execute("SELECT kind FROM task_events WHERE task_id=?", (tid,))]
    assert "live_receipt_verified" in kinds


def test_descendant_head_accepted(conn, gh):
    gh["fn"] = _gh(run={"headSha": LATER}, compare="ahead")
    tid = _task(conn)
    assert _complete(conn, tid, published_pr=PR_URL, live_receipt=RUN_URL)


def test_off_unless_configured(conn, gh, monkeypatch):
    import hermes_cli.config as config
    monkeypatch.setattr(config, "load_config", lambda *a, **k: {})
    gh["fn"] = lambda args: (_ for _ in ()).throw(AssertionError("gh must not run"))
    tid = _task(conn)
    assert _complete(conn, tid, published_pr=PR_URL)


def test_unmarked_card_not_gated(conn, gh):
    gh["fn"] = lambda args: (_ for _ in ()).throw(AssertionError("gh must not run"))
    tid = _task(conn, body="plain refactor, no live proof needed")
    assert _complete(conn, tid)


def test_project_marked_card_gated(conn, gh, monkeypatch):
    import hermes_cli.config as config
    tid = _task(conn, body="no proof line")
    conn.execute("UPDATE tasks SET project_id='p_live' WHERE id=?", (tid,))
    monkeypatch.setattr(config, "load_config", lambda *a, **k: {"kanban": {"live_receipt": {
        "workflows": ["integration-telegram.yml"], "projects": ["p_live"]}}})
    with pytest.raises(ValueError, match="live-validation receipt"):
        _complete(conn, tid, published_pr=PR_URL)
