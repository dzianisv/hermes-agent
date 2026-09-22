"""Regression suite for the "superseded-but-alive worker" incident (t_6a6ac2d3).

Timeline, reconstructed from the live board (ground truth):

* 15:39:58 the worker called ``block_task`` — its run went terminal
  (``outcome='blocked'``) and ``tasks.current_run_id`` became NULL.
* 15:40:38 the EM ran ``unblock`` — the card went back to ``ready``.
* The worker process for that run was STILL ALIVE and kept working until
  20:25. Every ``kanban_heartbeat`` it made came back with the generic
  ``could not heartbeat t_6a6ac2d3 (unknown id or not running)``, which is
  indistinguishable from "you typo'd the id".
* The dispatcher emitted ``respawn_guarded {"reason":"active_pr"}`` every
  ~60s for 4h45m and never respawned the card: rule 4 releases only on a
  handoff event, and an operator ``unblocked`` is not one.

Two of the three defects live here (the third, health telemetry, is in
``test_kanban_health_respawn_guarded.py``):

1. the heartbeat has no way to say "your run was superseded, exit";
2. the respawn guard's per-rule bypasses deadlock the card.

The worker-reaping half of the original fix is NOT ported: this tree already
ships ``reap_terminal_workers`` (``kanban_db_dispatch.py``), which retains
``worker_pid`` / ``worker_started_at`` / ``claim_lock`` on the closed run row
and covers the same hole with a stronger PID-reuse guard. See
``test_kanban_terminal_worker_reaper.py``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


PR_URL = "https://github.com/NousResearch/hermes-agent/pull/4774"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _backdate_comments(conn, task_id: str, ts: int) -> None:
    """Move every comment on ``task_id`` to ``ts`` (event ordering control)."""
    conn.execute(
        "UPDATE task_comments SET created_at = ? WHERE task_id = ?", (ts, task_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# 1. The incident, end to end: block -> unblock -> the next tick MUST spawn
# ---------------------------------------------------------------------------


def test_unblocked_card_with_open_pr_respawns(kanban_home, all_assignees_spawnable):
    """The exact t_6a6ac2d3 sequence must produce a spawn, not a guard."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="ship the thing", assignee="a")
        kb.add_comment(conn, tid, "worker", f"Opened PR: {PR_URL}")
        conn.commit()

        # The worker claims, then blocks: the run goes terminal and
        # current_run_id becomes NULL.
        assert kb.claim_task(conn, tid) is not None
        run_id = kb.latest_run(conn, tid).id
        assert kb.block_task(conn, tid, reason="needs a decision",
                             expected_run_id=run_id)
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.get_task(conn, tid).current_run_id is None

        # Before the unblock the guard is doing its job: a PR comment with no
        # deliberate re-queue behind it.
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"

        # The EM unblocks — NEW information (their decision) that only reaches
        # a worker through a fresh spawn's worker_context.
        assert kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"

        assert kbd.check_respawn_guard(conn, tid) is None, (
            "an operator unblock AFTER the PR comment is a deliberate re-queue "
            "and must release the active_pr guard"
        )

        res = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 99001)
        assert [t[0] for t in res.spawned] == [tid], (
            f"dispatch must spawn the unblocked card; got spawned={res.spawned} "
            f"respawn_guarded={res.respawn_guarded}"
        )
        assert res.respawn_guarded == []


# ---------------------------------------------------------------------------
# 2. Heartbeat directives
# ---------------------------------------------------------------------------


def test_heartbeat_from_superseded_run_returns_exit_directive(
    kanban_home, monkeypatch,
):
    """The superseded worker must be told to stop, with the real ids."""
    import tools.kanban_tools as kt

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="ship the thing", assignee="a")
        assert kb.claim_task(conn, tid) is not None
        run_id = kb.latest_run(conn, tid).id
        assert kb.block_task(conn, tid, reason="decide please",
                             expected_run_id=run_id)
        assert kb.unblock_task(conn, tid)
        assert kb.claim_task(conn, tid) is not None
        new_run_id = kb.latest_run(conn, tid).id
        assert new_run_id != run_id

        # DB layer: structured, and still falsy for legacy bool callers.
        hb = kbd.heartbeat_worker(conn, tid, expected_run_id=run_id)
        assert not hb
        assert hb.superseded and not hb.unknown_task
        assert hb.expected_run_id == run_id
        assert hb.current_run_id == new_run_id
        assert hb.task_status == "running"

    # Tool layer: the message the worker actually reads.
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    out = kt._handle_heartbeat({"task_id": tid})

    assert "superseded" in out
    assert "EXIT IMMEDIATELY" in out
    assert "fresh dispatch" in out
    # Names the status and BOTH run ids — the generic wording is the defect.
    assert "running" in out
    assert str(run_id) in out
    assert str(new_run_id) in out
    assert "unknown id or not running" not in out, (
        "the generic message is exactly what let the t_6a6ac2d3 worker keep "
        "going for 4h45m"
    )


def test_heartbeat_with_unknown_task_id_still_reports_unknown(kanban_home):
    """A genuinely bogus id must NOT be reported as a supersede."""
    import tools.kanban_tools as kt

    with kbc.connect_closing() as conn:
        hb = kbd.heartbeat_worker(conn, "t_does_not_exist")
        assert not hb
        assert hb.unknown_task and not hb.superseded

    out = kt._handle_heartbeat({"task_id": "t_does_not_exist"})
    assert "unknown id" in out
    assert "superseded" not in out
    assert "EXIT IMMEDIATELY" not in out


def test_heartbeat_on_current_run_still_succeeds(kanban_home, monkeypatch):
    """No regression for the healthy path (DB layer and tool layer)."""
    import tools.kanban_tools as kt

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="ship the thing", assignee="a")
        assert kb.claim_task(conn, tid) is not None
        run_id = kb.latest_run(conn, tid).id

        hb = kbd.heartbeat_worker(conn, tid, note="alive", expected_run_id=run_id)
        assert hb  # truthy — legacy callers keep working
        assert hb.ok and not hb.superseded and not hb.unknown_task
        assert hb.current_run_id == run_id
        assert kb.get_task(conn, tid).last_heartbeat_at is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    out = kt._handle_heartbeat({"task_id": tid})
    assert '"ok": true' in out
    assert "superseded" not in out


# ---------------------------------------------------------------------------
# 3. The bypass is DERIVED once, not a list per rule
# ---------------------------------------------------------------------------


def test_requeue_after_the_pr_comment_bypasses_but_before_does_not(kanban_home):
    """The predicate compares against EACH rule's own evidence timestamp."""
    with kbc.connect_closing() as conn:
        # Case A: unblock BEFORE the PR comment -> the guard must HOLD.
        # (Otherwise any historical unblock would permanently disarm the rule —
        # the failure mode a single global timestamp would create.)
        early = kb.create_task(conn, title="early unblock", assignee="a")
        assert kb.block_task(conn, early, reason="x")
        assert kb.unblock_task(conn, early)
        kb.add_comment(conn, early, "worker", f"Opened PR: {PR_URL}")
        # Put the comment strictly after the unblock event.
        _backdate_comments(conn, early, int(time.time()) + 5)
        assert kbd.check_respawn_guard(conn, early) == "active_pr", (
            "an unblock that predates the PR comment must not bypass"
        )

        # Case B: unblock AFTER the PR comment -> the guard must RELEASE.
        late = kb.create_task(conn, title="late unblock", assignee="a")
        kb.add_comment(conn, late, "worker", f"Opened PR: {PR_URL}")
        _backdate_comments(conn, late, int(time.time()) - 60)
        assert kbd.check_respawn_guard(conn, late) == "active_pr"
        assert kb.block_task(conn, late, reason="x")
        assert kb.unblock_task(conn, late)
        assert kbd.check_respawn_guard(conn, late) is None


@pytest.mark.parametrize(
    "kind", ["status", "promoted", "unblocked", "reclaimed", "review_reopened"],
)
def test_every_requeue_kind_bypasses_every_timestamped_rule(kanban_home, kind):
    """One predicate, applied uniformly — not a list per rule.

    The same event stream must release BOTH timestamped rules
    (``recent_success`` on a completed run, ``active_pr`` on a PR comment).
    Rule 4 shipped with its own private release path that no operator re-queue
    could reach; this asserts the shared derivation instead.
    """
    with kbc.connect_closing() as conn:
        # active_pr: PR comment, then the re-queue event.
        pr_task = kb.create_task(conn, title="pr", assignee="a")
        kb.add_comment(conn, pr_task, "worker", f"Opened PR: {PR_URL}")
        _backdate_comments(conn, pr_task, int(time.time()) - 60)
        assert kbd.check_respawn_guard(conn, pr_task) == "active_pr"

        # recent_success: a completed run, then the re-queue event.
        done_task = kb.create_task(conn, title="done", assignee="a")
        assert kb.claim_task(conn, done_task) is not None
        assert kb.complete_task(conn, done_task, summary="done")
        conn.execute(
            "UPDATE task_runs SET ended_at = ? WHERE task_id = ?",
            (int(time.time()) - 60, done_task),
        )
        conn.commit()
        assert kbd.check_respawn_guard(conn, done_task) == "recent_success"

        for tid in (pr_task, done_task):
            with kbc.write_txn(conn):
                kb._append_event(conn, tid, kind, {"synthetic": True})
            assert kbd.check_respawn_guard(conn, tid) is None, (
                f"a {kind!r} event after the evidence is a deliberate re-queue "
                f"and must release every timestamped rule"
            )


def test_recent_success_bypass_still_falls_through_to_active_pr(kanban_home):
    """A bypassed rule must not short-circuit the rules BELOW it.

    Rule 3's inline bypass used to fall through to rule 4. Moving the bypass
    into the wrapper must keep that: a card re-queued after its completion but
    carrying a NEWER PR comment is still guarded by ``active_pr``.
    """
    now = int(time.time())
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="done then PR", assignee="a")
        assert kb.claim_task(conn, tid) is not None
        assert kb.complete_task(conn, tid, summary="done")
        conn.execute(
            "UPDATE task_runs SET ended_at = ? WHERE task_id = ?", (now - 300, tid),
        )
        # Re-queue after the completion, then a PR comment AFTER the re-queue.
        with kbc.write_txn(conn):
            kb._append_event(conn, tid, "unblocked", None)
        conn.execute(
            "UPDATE task_events SET created_at = ? WHERE task_id = ? AND kind = 'unblocked'",
            (now - 200, tid),
        )
        kb.add_comment(conn, tid, "worker", f"Opened PR: {PR_URL}")
        _backdate_comments(conn, tid, now - 100)

        assert kbd.check_respawn_guard(conn, tid) == "active_pr", (
            "the re-queue predates the PR comment, so rule 4 must still hold "
            "even though rule 3 was bypassed"
        )


def test_rate_limit_cooldown_is_not_bypassable(kanban_home):
    """Explicit non-regression: the cooldown is a timer, not stale evidence.

    Re-probing a quota bucket already proven empty just hammers it, so
    ``rate_limit_cooldown`` reports no evidence timestamp and the shared bypass
    cannot fire on it.
    """
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="throttled", assignee="a")
        assert kb.claim_task(conn, tid) is not None
        with kbc.write_txn(conn):
            kb._end_run(conn, tid, outcome="rate_limited", status="rate_limited",
                        error="429 quota exceeded")
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        conn.commit()
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"

        with kbc.write_txn(conn):
            kb._append_event(conn, tid, "unblocked", None)
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown", (
            "a re-queue must not re-probe a quota bucket mid-cooldown"
        )


def test_infrastructure_cooldown_is_not_bypassable(kanban_home):
    """Same reasoning for the host-refusal cooldown: it is a timer on a HOST
    condition, and touching the card cannot make the host accept a spawn."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="host refused", assignee="a")
        assert kb.claim_task(conn, tid) is not None
        with kbc.write_txn(conn):
            kb._end_run(conn, tid, outcome="spawn_failed", status="spawn_failed",
                        error="no restart-safe scope", metadata={"infrastructure": True})
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        conn.commit()
        assert kbd.check_respawn_guard(conn, tid) == "infrastructure_cooldown"

        with kbc.write_txn(conn):
            kb._append_event(conn, tid, "unblocked", None)
        assert kbd.check_respawn_guard(conn, tid) == "infrastructure_cooldown"


# ---------------------------------------------------------------------------
# 4. Upstream rule-4 semantics this change must NOT weaken
# ---------------------------------------------------------------------------


def test_noop_reassign_after_pr_comment_still_guards(kanban_home):
    """``_is_handoff_event``'s no-op-reassign rule must survive the refactor.

    A dev→dev re-assign is not a handoff: lifting ``active_pr`` for the very
    implementer that opened the PR is how duplicate PRs get opened (#111910).
    ``assigned`` is deliberately absent from ``_REQUEUE_EVENT_KINDS``, so the
    new shared bypass must not release it either.
    """
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="pr open", assignee="a")
        kb.add_comment(conn, tid, "worker", f"Opened PR: {PR_URL}")
        _backdate_comments(conn, tid, int(time.time()) - 60)
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"

        with kbc.write_txn(conn):
            kb._append_event(conn, tid, "assigned", {"from": "a", "assignee": "a"})
        assert kbd.check_respawn_guard(conn, tid) == "active_pr", (
            "a no-op re-assign is not a handoff and must not release active_pr"
        )

        # A REAL handoff still releases it — the rule is intact, not disabled.
        with kbc.write_txn(conn):
            kb._append_event(conn, tid, "assigned", {"from": "a", "assignee": "b"})
        assert kbd.check_respawn_guard(conn, tid) is None


def test_open_pr_comment_with_no_requeue_still_guards(kanban_home):
    """Negative control: the guard must not widen into 'always respawn'."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="ship the thing", assignee="a")
        kb.add_comment(conn, tid, "worker", f"Opened PR: {PR_URL}")
        conn.commit()
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
