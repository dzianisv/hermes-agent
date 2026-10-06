"""``blocker_auth`` is scoped to the latest FAILED run; handoffs clear stale error text.

Incident: a crash stamped ``tasks.last_failure_error`` with the worker's last
output (``curl -H "Authorization: Bearer ***"``). The next run ended
``review_requested``, but ``check_respawn_guard`` only exempted a latest
``crashed`` outcome, so the stale task-scoped text tripped ``blocker_auth`` on
every tick (~2.4h) and the reviewer never spawned. Two contracts pin the fix:

* the blocker regex is trusted only when no run has ended or the latest ended
  run's outcome is a diagnostic one (``_BLOCKER_DIAGNOSTIC_OUTCOMES``);
* ``request_review`` / ``request_changes`` clear ``last_failure_error`` (but
  never ``consecutive_failures``).

The outcome inventory test keeps the diagnostic set honest: every outcome any
run-ending writer can produce must be classified, so a new outcome cannot
silently re-open (or silently skip) the blocker check.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_pr_acceptance_store as kpas

REPO = Path(__file__).resolve().parents[2]

CRASH_TEXT = (
    "Worker exited with code 1. Worker's last output: "
    "'curl -s -H \"Authorization: Bearer ***\" https://api.example.com/v1/items'"
)


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Never shell out to `gh` from a test.
    monkeypatch.setattr(kbd, "_active_pr_guard_applies", lambda _url: (True, None))
    kb.init_db()
    return home


@pytest.fixture
def review_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)
    monkeypatch.setattr(
        cfgmod, "load_config", lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )


def _stamp_crash(conn, tid: str, error: str = CRASH_TEXT) -> None:
    """Book a crash the way ``_reclaim_dead_workers`` + ``_account_crashes`` do:
    restore the phase, close the run ``crashed``, then the real failure recorder."""
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, claim_expires = NULL, "
            "worker_pid = NULL, worker_started_at = NULL WHERE id = ?", (tid,),
        )
        kb._end_run(conn, tid, outcome="crashed", status="crashed", error=error)
    assert kbd._record_task_failure(conn, tid, error=error, outcome="crashed") is False


def _task_row(conn, tid):
    return conn.execute(
        "SELECT status, last_failure_error, consecutive_failures FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()


def _guard_events(conn, tid) -> int:
    return conn.execute(
        "SELECT count(*) FROM task_events WHERE task_id = ? AND kind = 'respawn_guarded'",
        (tid,),
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# A1: the incident end-to-end
# ---------------------------------------------------------------------------


def test_reviewer_spawns_after_crash_then_review_handoff(kanban_home, review_dispatch):
    spawned: list[tuple[str, str]] = []

    def spawn(task, workspace):
        spawned.append((task.id, task.assignee))
        return None

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="crash then review", assignee="worker")
        _stamp_crash(conn, tid)
        assert "Authorization" in _task_row(conn, tid)["last_failure_error"]

        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb.request_review(
            conn, tid, summary="done", reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        )
        assert _task_row(conn, tid)["last_failure_error"] is None
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None

        result = kbd.dispatch_once(conn, spawn_fn=spawn)

        assert (tid, "reviewer") in spawned
        assert [s for s in spawned if s[0] == tid] == [(tid, "reviewer")]
        assert all(t != tid for t, _ in result.respawn_guarded)
        assert _guard_events(conn, tid) == 0
        reviewer_runs = conn.execute(
            "SELECT count(*) FROM task_runs WHERE task_id = ? AND profile = 'reviewer'",
            (tid,),
        ).fetchone()[0]
        assert reviewer_runs == 1


# ---------------------------------------------------------------------------
# A2: the guard scope alone (column NOT cleared)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lane", ["review", "ready"])
def test_handoff_outcome_does_not_trust_stale_error_text(kanban_home, lane):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stale text", assignee="worker")
        _stamp_crash(conn, tid)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET outcome = 'review_requested', status = 'review' "
                "WHERE id = (SELECT id FROM task_runs WHERE task_id = ? "
                "ORDER BY ended_at DESC, id DESC LIMIT 1)", (tid,),
            )
        assert _task_row(conn, tid)["last_failure_error"] == CRASH_TEXT
        # No completed run and no PR comment: recent_success / active_pr cannot apply.
        assert kbd.check_respawn_guard(conn, tid, lane=lane) is None


# ---------------------------------------------------------------------------
# A3: diagnostic outcomes (and no run at all) still park the card
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", sorted(kbd._BLOCKER_DIAGNOSTIC_OUTCOMES))
def test_diagnostic_outcome_keeps_blocker_auth(kanban_home, outcome):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=outcome, assignee="worker")
        assert kb.claim_task(conn, tid) is not None
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready', claim_lock = NULL WHERE id = ?", (tid,))
            # metadata=None: a spawn_failed without ``infrastructure`` is a card failure.
            kb._end_run(conn, tid, outcome=outcome, status=outcome, error="401 auth failed")
            conn.execute(
                "UPDATE tasks SET last_failure_error = '401 auth failed' WHERE id = ?", (tid,),
            )
        assert kbd.check_respawn_guard(conn, tid) == "blocker_auth"


@pytest.mark.parametrize("error", ["401 auth failed", "quota exceeded for this key"])
def test_no_ended_run_keeps_blocker_auth(kanban_home, error):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="never ran", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET last_failure_error = ? WHERE id = ?", (error, tid))
        assert kbd._RESPAWN_BLOCKER_RE.search(error)
        assert kbd.check_respawn_guard(conn, tid) == "blocker_auth"


# ---------------------------------------------------------------------------
# A4: request_changes clears the text, never the streak
# ---------------------------------------------------------------------------


def test_request_changes_clears_error_text_but_keeps_failure_count(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="changes", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="ready", reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        )
        assert kb.claim_review_task(conn, tid) is not None
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET consecutive_failures = 2, last_failure_error = ? WHERE id = ?",
                (CRASH_TEXT, tid),
            )
        ok, detail = kb.request_changes(conn, tid, reason="add tests")
        assert ok is True, detail
        row = _task_row(conn, tid)
        assert row["last_failure_error"] is None
        assert row["consecutive_failures"] == 2


def test_pr_acceptance_refusal_after_handoff_stamps_fresh_error(kanban_home):
    """After a handoff cleared the column, a later refusal still surfaces."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="accept", assignee="worker", completion_contract="acme/repo")
        claimed = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="ready", reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        )
        assert _task_row(conn, tid)["last_failure_error"] is None
        receipt = {"ok": False, "classification": "failing", "detail": "required check red",
                   "recovery": "fix CI and retry"}
        with kb.write_txn(conn):
            assert kpas.record_acceptance(conn, tid, (kpas._snapshot(conn, tid), receipt)) is False
        assert "PR acceptance failing" in _task_row(conn, tid)["last_failure_error"]


# ---------------------------------------------------------------------------
# A5: outcome inventory over every run-ending writer
# ---------------------------------------------------------------------------

_SCANNED = ("hermes_cli/kanban_db_dispatch.py", "hermes_cli/kanban_db.py", "agent/turn_finalizer.py")
_WRITERS = {"_record_task_failure", "_end_run", "_end_or_synthesize_run", "_synthesize_ended_run"}

# Outcomes that must NOT re-enable blocker_auth, one reason each.
EXCLUDED = {
    "crashed": "error text is captured worker stdout, not a diagnosis (#117097)",
    "rate_limited": "own cooldown path runs before blocker_auth",
    "reclaimed": "stale-lock / operator reclaim diagnostic, not a provider blocker",
    "stale": "heartbeat-stale reclaim diagnostic, not a provider blocker",
    "completed": "success, not a failure",
    "blocked": "operator/worker block; blocked cards are not dispatched",
    "review_requested": "handoff, not a failure",
    "changes_requested": "handoff, not a failure",
    "approved": "handoff, not a failure",
    "scheduled": "deferred wake, not a failure",
}

# Non-literal ``outcome=`` arguments, reviewed as pure forwarding sites.
REVIEWED_DYNAMIC = {
    ("hermes_cli/kanban_db_dispatch.py", "_reclaim_dead_workers"),  # _DeadWorker.run_outcome
    ("hermes_cli/kanban_db_dispatch.py", "_record_task_failure"),   # forwards caller's outcome
    ("hermes_cli/kanban_db.py", "_end_or_synthesize_run"),          # forwards caller's outcome
}


def _collect(source: str, label: str):
    """``(literal outcomes, dynamic sites)`` for every writer call in ``source``,
    plus string literals returned by any ``run_outcome``."""
    tree = ast.parse(source)
    literals: set[str] = set()
    dynamic: set[tuple[str, str]] = set()

    def strings(node) -> set[str]:
        return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}

    def visit(node, func: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node.name
            if func == "run_outcome":
                for ret in ast.walk(node):
                    if isinstance(ret, ast.Return) and ret.value is not None:
                        literals.update(strings(ret.value))
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name in _WRITERS:
                arg = next((k.value for k in node.keywords if k.arg == "outcome"), None)
                if arg is None:
                    # All writers take ``outcome`` keyword-only; a positional or
                    # **kwargs form must be reviewed like any dynamic site.
                    dynamic.add((label, func))
                elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    literals.add(arg.value)
                else:
                    dynamic.add((label, func))
        for child in ast.iter_child_nodes(node):
            visit(child, func)

    visit(tree, "<module>")
    return literals, dynamic


def _check(literals, dynamic) -> list[str]:
    problems = [
        f"unclassified outcome {o!r}" for o in sorted(literals)
        if o not in kbd._BLOCKER_DIAGNOSTIC_OUTCOMES and o not in EXCLUDED
    ]
    problems += [f"unreviewed dynamic outcome at {s}" for s in sorted(dynamic - REVIEWED_DYNAMIC)]
    return problems


def test_every_run_outcome_is_classified_for_blocker_auth():
    literals: set[str] = set()
    dynamic: set[tuple[str, str]] = set()
    for rel in _SCANNED:
        lit, dyn = _collect((REPO / rel).read_text(encoding="utf-8"), rel)
        literals |= lit
        dynamic |= dyn
    assert not (kbd._BLOCKER_DIAGNOSTIC_OUTCOMES & set(EXCLUDED))
    # The collector reached the known producers (breaker gave_up, run_outcome).
    assert {"gave_up", "rate_limited", "crashed", "review_requested"} <= literals
    assert _check(literals, dynamic) == []


def test_collector_flags_new_literal_and_unreviewed_dynamic_site():
    source = textwrap.dedent(
        """
        def new_path(conn, tid):
            _kb._end_run(conn, tid, outcome="exploded", status="exploded")

        def forwarder(conn, tid, oc):
            _record_task_failure(conn, tid, "err", outcome=oc)

        class _Dead:
            @property
            def run_outcome(self):
                return "melted" if self.x else "crashed"
        """
    )
    literals, dynamic = _collect(source, "fixture.py")
    problems = _check(literals, dynamic)
    assert "unclassified outcome 'exploded'" in problems
    assert "unclassified outcome 'melted'" in problems
    assert "unreviewed dynamic outcome at ('fixture.py', 'forwarder')" in problems
