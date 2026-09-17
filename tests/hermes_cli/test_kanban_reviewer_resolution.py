"""Omitted-reviewer resolution: resolve to a distinct reviewer, or fail closed.

Defect this pins down: ``kanban_db.request_review`` treated ``reviewer`` as
optional and, when omitted, moved the task to ``review`` **without changing
``assignee``**. The builder stayed assigned, so the review dispatcher claimed
the card for the same profile that wrote the code — silent self-review.

The contract here:

* Explicit ``reviewer=`` always wins (caller override, unchanged).
* Omitted reviewer resolves from ``kanban.default_reviewer`` in config.
* A missing / unknown / same-as-implementer default **fails closed**: no
  transition to ``review``, the live claim is untouched, and the caller gets a
  precise actionable reason. Never a silent fallback to the implementer.
* Every surface (domain fn, worker tool, CLI, dashboard) uses the same
  resolver.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _make_profile(kanban_home: Path, name: str) -> None:
    (kanban_home / "profiles" / name).mkdir(parents=True, exist_ok=True)


def _set_default_reviewer(
    monkeypatch: pytest.MonkeyPatch, value: object,
) -> None:
    cfg = {"kanban": {"default_reviewer": value}} if value is not None else {}
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: cfg)
    monkeypatch.setattr(config_mod, "load_config", lambda: cfg)


def _events(conn, tid, kind):
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
        (tid,),
    ).fetchall()
    return [
        json.loads(r["payload"]) if r["payload"] else None
        for r in rows if r["kind"] == kind
    ]


# ---------------------------------------------------------------------------
# 1. Omitted reviewer resolves to the configured default and is assigned
# ---------------------------------------------------------------------------


def test_omitted_reviewer_resolves_to_configured_default(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_profile(kanban_home, "reviewer")
    _set_default_reviewer(monkeypatch, "reviewer")

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="builder")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        ok, reason = kb.request_review(
            conn, tid,
            summary="done",
            expected_run_id=claimed.current_run_id,
            with_reason=True,
        )
        assert (ok, reason) == (True, None)

        task = kb.get_task(conn, tid)
        assert task.status == "review"
        assert task.assignee == "reviewer", "reviewer must own the review lane"

        payload = _events(conn, tid, "review_requested")[-1]
        assert payload["implementer"] == "builder"
        assert payload["reviewer"] == "reviewer"

        # The real dispatch path selects the reviewer, not the builder.
        review_run = kb.claim_review_task(conn, tid)
        assert review_run is not None
        assert kb.get_task(conn, tid).assignee == "reviewer"


# ---------------------------------------------------------------------------
# 2. Explicit reviewer wins over the configured default
# ---------------------------------------------------------------------------


def test_explicit_reviewer_overrides_the_default(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_profile(kanban_home, "reviewer")
    _set_default_reviewer(monkeypatch, "reviewer")

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="builder")
        claimed = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="done", reviewer="lead-reviewer",
            expected_run_id=claimed.current_run_id,
        ) is True
        assert kb.get_task(conn, tid).assignee == "lead-reviewer"
        assert _events(conn, tid, "review_requested")[-1]["reviewer"] == (
            "lead-reviewer"
        )


# ---------------------------------------------------------------------------
# 3. Fail-closed: missing / unknown / self default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "default_value, needle",
    [
        (None, "kanban.default_reviewer"),
        ("", "kanban.default_reviewer"),
        ("   ", "kanban.default_reviewer"),
        ("ghost-profile", "does not exist"),
        ("builder", "same profile as the implementer"),
    ],
)
def test_unresolvable_default_fails_closed(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    default_value,
    needle: str,
) -> None:
    _set_default_reviewer(monkeypatch, default_value)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="builder")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        ok, reason = kb.request_review(
            conn, tid,
            summary="done",
            expected_run_id=claimed.current_run_id,
            with_reason=True,
        )
        assert ok is False
        assert reason is not None and needle in reason

        row = conn.execute(
            "SELECT status, assignee, claim_lock, worker_pid, current_run_id "
            "FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        # No transition, no self-review, live claim intact.
        assert row["status"] == "running"
        assert row["assignee"] == "builder"
        assert row["claim_lock"] is not None
        assert row["current_run_id"] == claimed.current_run_id
        assert _events(conn, tid, "review_requested") == []


def test_fail_closed_default_does_not_end_the_run(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused review request must leave the run open so the worker can
    retry with an explicit reviewer instead of being stranded."""
    _set_default_reviewer(monkeypatch, "")
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="builder")
        claimed = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="done",
            expected_run_id=claimed.current_run_id,
        ) is False
        run = conn.execute(
            "SELECT status, outcome FROM task_runs WHERE task_id = ? "
            "ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()
        assert run["status"] == "running"
        assert run["outcome"] is None

        # Retrying with an explicit reviewer still works on the same run.
        assert kb.request_review(
            conn, tid, summary="done", reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        ) is True
        assert kb.get_task(conn, tid).assignee == "reviewer"


# ---------------------------------------------------------------------------
# 4. The worker tool surface carries the same behaviour
# ---------------------------------------------------------------------------


def test_worker_tool_resolves_default_and_fails_closed(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as tools

    _make_profile(kanban_home, "reviewer")

    # (a) unresolvable -> tool error, card stays running with the builder.
    _set_default_reviewer(monkeypatch, "")
    with kb.connect() as conn:
        bad = kb.create_task(conn, title="impl", assignee="builder")
        bad_run = kb.claim_task(conn, bad, claimer="builder:1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", bad)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(bad_run.current_run_id))
    out = json.loads(tools._handle_request_review({"summary": "ready"}))
    assert "error" in out
    assert "kanban.default_reviewer" in out["error"]
    with kb.connect() as conn:
        after = kb.get_task(conn, bad)
        assert after.status == "running" and after.assignee == "builder"

    # (b) configured -> tool resolves it without the caller naming anyone.
    _set_default_reviewer(monkeypatch, "reviewer")
    with kb.connect() as conn:
        good = kb.create_task(conn, title="impl", assignee="builder")
        good_run = kb.claim_task(conn, good, claimer="builder:2")
    monkeypatch.setenv("HERMES_KANBAN_TASK", good)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(good_run.current_run_id))
    out = json.loads(tools._handle_request_review({"summary": "ready"}))
    assert "error" not in out, out
    with kb.connect() as conn:
        landed = kb.get_task(conn, good)
        assert landed.status == "review"
        assert landed.assignee == "reviewer"


# ---------------------------------------------------------------------------
# 5. Re-review provenance still wins over the configured default
# ---------------------------------------------------------------------------


def test_changes_requested_provenance_beats_the_default(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_profile(kanban_home, "reviewer")
    _set_default_reviewer(monkeypatch, "reviewer")

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="builder")
        claimed = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="v1", reviewer="lead-reviewer",
            expected_run_id=claimed.current_run_id,
        ) is True
        review = kb.claim_review_task(conn, tid)
        assert kb.request_changes(
            conn, tid, reason="fix it",
            expected_run_id=review.current_run_id,
        ) == (True, "builder")
        again = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="v2", expected_run_id=again.current_run_id,
        ) is True
        assert kb.get_task(conn, tid).assignee == "lead-reviewer"
