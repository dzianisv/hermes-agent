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

The outcome inventory keeps the diagnostic set honest behaviorally: each
run-outcome writer is driven through its real entry point while a test-only
SQLite trigger records the outcomes written, and every recorded outcome must
be classified, so a new outcome cannot silently re-open (or skip) the check.
"""

from __future__ import annotations

import importlib.util
import sys
import time
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


def _stamp_crash_in_review(conn, tid: str, error: str = CRASH_TEXT) -> None:
    """Same booking for a reviewer crash: the review lane restores ``review``."""
    claimed = kb.claim_review_task(conn, tid)
    assert claimed is not None
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'review', claim_lock = NULL, claim_expires = NULL, "
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
        # max_retries: two crashes must not trip the breaker before the verdict.
        tid = kb.create_task(conn, title="changes", assignee="worker", max_retries=5)
        _stamp_crash(conn, tid)
        stamped = _task_row(conn, tid)
        assert "Authorization: Bearer ***" in stamped["last_failure_error"]
        streak = stamped["consecutive_failures"]
        assert streak >= 1

        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb.request_review(
            conn, tid, summary="ready", reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        )
        # The reviewer's own crash before its verdict re-stamps the column.
        _stamp_crash_in_review(conn, tid)
        assert _task_row(conn, tid)["last_failure_error"] == CRASH_TEXT
        streak = _task_row(conn, tid)["consecutive_failures"]

        review_claim = kb.claim_review_task(conn, tid)
        assert review_claim is not None
        ok, detail = kb.request_changes(
            conn, tid, reason="add tests", expected_run_id=review_claim.current_run_id,
        )
        assert ok is True, detail
        row = _task_row(conn, tid)
        assert row["status"] == "ready"
        assert row["last_failure_error"] is None
        assert row["consecutive_failures"] == streak
        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None


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
# A5: behavioral outcome inventory over every run-outcome writer
# ---------------------------------------------------------------------------
#
# A test-only SQLite trigger on the throwaway test DB records every outcome a
# ``task_runs`` row is created or updated with; each writer site is then driven
# through its real product entry point. Production schema is untouched. The
# recorder is a regular table + trigger (not TEMP) because several writers
# (dashboard, turn finalizer) open their own connection and a TEMP trigger only
# fires on the connection that created it.

# Outcomes that must NOT re-enable blocker_auth, one reason each.
_EXCLUDED_OUTCOMES = {
    "crashed": "error text is captured worker stdout, not a diagnosis (#117097)",
    "rate_limited": "own cooldown path runs before blocker_auth",
    "reclaimed": "stale-lock / operator / orphan reclaim, not a provider blocker",
    "stale": "heartbeat-stale reclaim, not a provider blocker",
    "completed": "success, not a failure",
    "blocked": "operator/worker block; blocked cards are not dispatched",
    "review_requested": "handoff, not a failure",
    "changes_requested": "handoff, not a failure",
    "approved": "handoff, not a failure",
    "scheduled": "deferred wake, not a failure",
}

_RECORDER_SQL = """
CREATE TABLE IF NOT EXISTS _outcomes_seen (task_id TEXT, run_id INTEGER, outcome TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS _rec_outcome_insert AFTER INSERT ON task_runs
WHEN NEW.outcome IS NOT NULL
BEGIN INSERT INTO _outcomes_seen VALUES (NEW.task_id, NEW.id, NEW.outcome); END;
CREATE TRIGGER IF NOT EXISTS _rec_outcome_update AFTER UPDATE OF outcome ON task_runs
WHEN NEW.outcome IS NOT NULL
BEGIN INSERT INTO _outcomes_seen VALUES (NEW.task_id, NEW.id, NEW.outcome); END;
"""


def unclassified(seen) -> set[str]:
    return set(seen) - kbd._BLOCKER_DIAGNOSTIC_OUTCOMES - set(_EXCLUDED_OUTCOMES)


def _install_recorder(conn) -> None:
    conn.executescript(_RECORDER_SQL)


def _seen(conn, tid: str | None = None) -> set[str]:
    sql, args = "SELECT outcome FROM _outcomes_seen", ()
    if tid is not None:
        sql, args = sql + " WHERE task_id = ?", (tid,)
    return {r[0] for r in conn.execute(sql, args).fetchall()}


def _dead_worker(conn, monkeypatch, tid: str, exit_code: int) -> None:
    """Claim host-locally, then report the worker pid dead with ``exit_code``."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    assert kb.claim_task(conn, tid) is not None
    pid = 71000 + exit_code
    kbd._set_worker_pid(conn, tid, pid)
    kbd._record_worker_exit(pid, exit_code << 8)
    kbd.detect_crashed_workers(conn)


def _drive_crashed(conn, monkeypatch):
    tid = kb.create_task(conn, title="crash", assignee="worker")
    _dead_worker(conn, monkeypatch, tid, 1)
    return tid


def _drive_rate_limited(conn, monkeypatch):
    tid = kb.create_task(conn, title="rl", assignee="worker")
    _dead_worker(conn, monkeypatch, tid, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    return tid


def _backdate_run(conn, tid: str, seconds: int) -> None:
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET started_at = ? "
            "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (int(time.time()) - seconds, tid),
        )


def _drive_timed_out_runtime(conn, monkeypatch):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    tid = kb.create_task(conn, title="slow", assignee="worker", max_runtime_seconds=1)
    assert kb.claim_task(conn, tid) is not None
    kbd._set_worker_pid(conn, tid, 72001)
    _backdate_run(conn, tid, 30)
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda *_a: None)
    return tid


def _drive_timed_out_budget(conn, monkeypatch):
    import logging

    from agent import turn_finalizer

    tid = kb.create_task(conn, title="budget", assignee="worker")
    assert kb.claim_task(conn, tid) is not None
    turn_finalizer._record_kanban_budget_exhausted(tid, 90, 90, logging.getLogger("test"))
    return tid


def _drive_stale(conn, monkeypatch):
    tid = kb.create_task(conn, title="stale", assignee="worker")
    assert kb.claim_task(conn, tid) is not None
    _backdate_run(conn, tid, 7200)
    assert tid in kbd.detect_stale_running(conn, stale_timeout_seconds=60)
    return tid


def _failing_spawn(task, workspace):
    raise RuntimeError("spawn exploded")


def _spawnable(monkeypatch) -> None:
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)


def _drive_spawn_failed(conn, monkeypatch):
    _spawnable(monkeypatch)
    tid = kb.create_task(conn, title="spawn", assignee="worker", max_retries=5)
    kbd.dispatch_once(conn, spawn_fn=_failing_spawn)
    return tid


def _drive_gave_up(conn, monkeypatch):
    _spawnable(monkeypatch)
    tid = kb.create_task(conn, title="breaker", assignee="worker", max_retries=1)
    result = kbd.dispatch_once(conn, spawn_fn=_failing_spawn)
    assert tid in result.auto_blocked
    return tid


def _drive_reclaimed_stale_claim(conn, monkeypatch):
    tid = kb.create_task(conn, title="ttl", assignee="worker")
    assert kb.claim_task(conn, tid, ttl_seconds=1) is not None
    # Claim TTL elapsed: the only input release_stale_claims reads.
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET claim_expires = 1 WHERE id = ?", (tid,))
    assert kb.release_stale_claims(conn, signal_fn=lambda *_a: None) == 1
    return tid


def _drive_reclaimed_operator(conn, monkeypatch):
    tid = kb.create_task(conn, title="manual", assignee="worker")
    assert kb.claim_task(conn, tid) is not None
    assert kb.reclaim_task(conn, tid, reason="operator", signal_fn=lambda *_a: None)
    return tid


def _drive_reclaimed_orphan(conn, monkeypatch):
    tid = kb.create_task(conn, title="orphan", assignee="worker")
    assert kb.claim_task(conn, tid) is not None
    # Broken claim bookkeeping (crash mid-claim / DB restore) is the input.
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET claim_lock = NULL WHERE id = ?", (tid,))
    assert tid in kbd.reconcile_orphaned_running(conn)
    return tid


def _drive_reclaimed_parent_reopen(conn, monkeypatch):
    parent = kb.create_task(conn, title="parent", assignee="planner")
    assert kb.complete_task(conn, parent, result="done")
    child = kb.create_task(conn, title="child", assignee="worker", parents=[parent])
    assert kb.claim_task(conn, child) is not None
    ok, _ = kb.reopen_done_task(conn, parent, actor="operator")
    assert ok
    return child


def _drive_reclaimed_archive(conn, monkeypatch):
    tid = kb.create_task(conn, title="archive", assignee="worker")
    assert kb.claim_task(conn, tid) is not None
    assert kb.archive_task(conn, tid, signal_fn=lambda *_a: None)
    return tid


def _drive_reclaimed_dangling(conn, monkeypatch):
    tid = kb.create_task(conn, title="dangling", assignee="worker")
    assert kb.claim_task(conn, tid) is not None
    # Leaked open run: phase reset to ready while current_run_id still points
    # at an open run (the invariant violation _reclaim_dangling_run recovers).
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready', claim_lock = NULL WHERE id = ?", (tid,))
    assert kb.claim_task(conn, tid) is not None
    return tid


def _drive_reclaimed_backfill_cas(conn, monkeypatch):
    """``_backfill_legacy_inflight_runs`` CAS-failure branch.

    Legacy input: a ``running`` task with no ``current_run_id``. The race (a
    claimer installing ``current_run_id`` between the backfill's INSERT and its
    CAS UPDATE) is reproduced deterministically by a test-only TEMP trigger on
    this connection that, right after the backfill inserts its run row, points
    the task at a different run -- exactly what the racing claimer would do.
    """
    tid = kb.create_task(conn, title="legacy", assignee="worker")
    racer = kb.create_task(conn, title="racer", assignee="worker")
    assert kb.claim_task(conn, racer) is not None
    racer_run = conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (racer,)).fetchone()[0]
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'running', claim_lock = 'legacy:1', "
            "claim_expires = 9999999999, current_run_id = NULL WHERE id = ?", (tid,),
        )
    conn.execute(
        f"CREATE TEMP TRIGGER _race_claim AFTER INSERT ON main.task_runs "
        f"WHEN NEW.task_id = '{tid}' AND NEW.status = 'running' "
        f"BEGIN UPDATE tasks SET current_run_id = {int(racer_run)} WHERE id = NEW.task_id; END"
    )
    try:
        kbc._backfill_legacy_inflight_runs(conn)
    finally:
        conn.execute("DROP TRIGGER IF EXISTS _race_claim")
    return tid


def _load_dashboard_client():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_outcome_inventory",
        REPO / "plugins" / "kanban" / "dashboard" / "plugin_api.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    app = fastapi.FastAPI()
    app.include_router(mod.router, prefix="/api/plugins/kanban")
    return TestClient(app)


def _drive_reclaimed_dashboard(conn, monkeypatch):
    client = _load_dashboard_client()
    tid = kb.create_task(conn, title="drag", assignee="worker")
    assert kb.claim_task(conn, tid) is not None
    r = client.patch(f"/api/plugins/kanban/tasks/{tid}", json={"status": "ready"})
    assert r.status_code == 200, r.text
    return tid


def _drive_completed_run(conn, monkeypatch):
    tid = kb.create_task(conn, title="done", assignee="worker")
    claimed = kb.claim_task(conn, tid)
    assert kb.complete_task(conn, tid, result="ok", expected_run_id=claimed.current_run_id)
    return tid


def _drive_completed_synth(conn, monkeypatch):
    tid = kb.create_task(conn, title="manual done", assignee="worker")
    assert kb.complete_task(conn, tid, result="ok", summary="manual")
    return tid


def _drive_completed_edit(conn, monkeypatch):
    """``hermes kanban edit --result``: backfill on a done card with no completed
    run (legacy pre-runs row: status ``done``, no task_runs) synthesizes one."""
    tid = kb.create_task(conn, title="edit", assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (tid,))
    assert kb.edit_task(conn, tid, result="backfilled", summary="backfilled")
    return tid


def _drive_completed_swarm(conn, monkeypatch):
    from hermes_cli.kanban_swarm import SwarmWorkerSpec, create_swarm

    created = create_swarm(
        conn, goal="inventory", workers=[SwarmWorkerSpec(profile="w", title="A", body="A")],
        verifier_assignee="verifier", synthesizer_assignee="synth",
    )
    return created.root_id


def _drive_blocked(conn, monkeypatch):
    tid = kb.create_task(conn, title="block", assignee="worker")
    claimed = kb.claim_task(conn, tid)
    assert kb.block_task(conn, tid, reason="needs input", expected_run_id=claimed.current_run_id)
    return tid


def _review_ready(conn, title: str) -> str:
    tid = kb.create_task(conn, title=title, assignee="worker")
    claimed = kb.claim_task(conn, tid)
    assert kb.request_review(
        conn, tid, summary="ready", reviewer="reviewer", expected_run_id=claimed.current_run_id,
    )
    return tid


def _drive_review_requested(conn, monkeypatch):
    return _review_ready(conn, "review")


def _drive_changes_requested(conn, monkeypatch):
    tid = _review_ready(conn, "changes")
    claim = kb.claim_review_task(conn, tid)
    ok, detail = kb.request_changes(conn, tid, reason="fix", expected_run_id=claim.current_run_id)
    assert ok, detail
    return tid


def _drive_approved(conn, monkeypatch):
    tid = _review_ready(conn, "approve")
    claim = kb.claim_review_task(conn, tid)
    ok, detail = kb.approve_for_merge(conn, tid, summary="lgtm", expected_run_id=claim.current_run_id)
    assert ok, detail
    return tid


def _drive_scheduled(conn, monkeypatch):
    tid = kb.create_task(conn, title="later", assignee="worker")
    claimed = kb.claim_task(conn, tid)
    assert kb.schedule_task(conn, tid, reason="wait for CI", expected_run_id=claimed.current_run_id)
    return tid


# site -> (driver, outcome the site must produce)
_WRITER_SITES = {
    "detect_crashed_workers:crashed": (_drive_crashed, "crashed"),
    "detect_crashed_workers:rate_limited": (_drive_rate_limited, "rate_limited"),
    "enforce_max_runtime": (_drive_timed_out_runtime, "timed_out"),
    "turn_finalizer:budget_exhausted": (_drive_timed_out_budget, "timed_out"),
    "detect_stale_running": (_drive_stale, "stale"),
    "dispatch_once:spawn_failed": (_drive_spawn_failed, "spawn_failed"),
    "dispatch_once:breaker_gave_up": (_drive_gave_up, "gave_up"),
    "release_stale_claims": (_drive_reclaimed_stale_claim, "reclaimed"),
    "reclaim_task": (_drive_reclaimed_operator, "reclaimed"),
    "reconcile_orphaned_running": (_drive_reclaimed_orphan, "reclaimed"),
    "invalidate_descendants_for_parent_reopen": (_drive_reclaimed_parent_reopen, "reclaimed"),
    "archive_task": (_drive_reclaimed_archive, "reclaimed"),
    "_reclaim_dangling_run": (_drive_reclaimed_dangling, "reclaimed"),
    "kanban_db_connect:_backfill_legacy_inflight_runs": (_drive_reclaimed_backfill_cas, "reclaimed"),
    "dashboard:_set_status_direct": (_drive_reclaimed_dashboard, "reclaimed"),
    "complete_task:end_run": (_drive_completed_run, "completed"),
    "complete_task:synthesize": (_drive_completed_synth, "completed"),
    "edit_task:synthesize": (_drive_completed_edit, "completed"),
    "kanban_swarm:_activate_root_inline": (_drive_completed_swarm, "completed"),
    "block_task": (_drive_blocked, "blocked"),
    "request_review": (_drive_review_requested, "review_requested"),
    "request_changes": (_drive_changes_requested, "changes_requested"),
    "approve_for_merge": (_drive_approved, "approved"),
    "schedule_task": (_drive_scheduled, "scheduled"),
}


def test_classification_sets_are_disjoint():
    assert not (kbd._BLOCKER_DIAGNOSTIC_OUTCOMES & set(_EXCLUDED_OUTCOMES))


@pytest.mark.parametrize("site", sorted(_WRITER_SITES))
def test_every_writer_outcome_is_classified(kanban_home, monkeypatch, site):
    driver, expected = _WRITER_SITES[site]
    with kbc.connect() as conn:
        _install_recorder(conn)
        tid = driver(conn, monkeypatch)
        produced = _seen(conn, tid)
        # The named writer actually wrote: a removed/rerouted site is noticed.
        assert expected in produced, f"{site} produced {sorted(produced)}"
        assert unclassified(_seen(conn)) == set()


def test_inventory_covers_every_classified_outcome():
    """Every classified outcome has a driven writer (stale entries are noticed)."""
    driven = {outcome for _driver, outcome in _WRITER_SITES.values()}
    assert driven == kbd._BLOCKER_DIAGNOSTIC_OUTCOMES | set(_EXCLUDED_OUTCOMES)


def test_recorder_flags_a_new_outcome(kanban_home):
    with kbc.connect() as conn:
        _install_recorder(conn)
        tid = kb.create_task(conn, title="novel", assignee="worker")
        assert kb.claim_task(conn, tid) is not None
        with kb.write_txn(conn):
            kb._end_run(conn, tid, outcome="brand_new_outcome", status="brand_new_outcome")
        assert unclassified(_seen(conn)) == {"brand_new_outcome"}
