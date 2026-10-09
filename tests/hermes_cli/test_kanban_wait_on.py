"""Harness #16: a card waiting on an external condition is not re-claimed until
that condition changes; 2 consecutive no-progress runs send it to re-spec.

Real temp DB, real functions. Observation is injected as a plain function
(status_fn / head_fn) that reads a dict the test controls.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_wait_on as kwo
from tests.hermes_cli._kanban_modules import KanbanModules

kb = KanbanModules()


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    c = kb.connect()
    yield c
    c.close()


class World:
    """External state the status function reads; tests mutate it."""

    def __init__(self):
        self.fp: dict[str, str] = {}
        self.calls = 0

    def status(self, handle, _conn):
        self.calls += 1
        return self.fp.get(str(handle))


def _running(conn, title="fix CI", assignee="software-engineer"):
    tid = kb.create_task(conn, title=title, assignee=assignee)
    claimed = kb.claim_task(conn, tid)
    return tid, claimed.current_run_id


def test_parse_handles():
    assert str(kwo.parse_handle("gh-run:o/r:123")) == "gh-run:o/r:123"
    assert kwo.parse_handle("pr:o/r#5@ABCDEF1").sha == "abcdef1"
    assert kwo.parse_handle("card:t_abc123").kind == "card"
    with pytest.raises(ValueError):
        kwo.parse_handle("ci red, retry later")


def test_parked_card_not_claimed_until_handle_changes(conn):
    w = World()
    h = "gh-run:o/r:42"
    w.fp[h] = "completed|failure|1|aaa"
    tid, run = _running(conn)
    assert kwo.wait_on_task(conn, tid, h, expected_run_id=run, status_fn=w.status)
    assert kb.get_task(conn, tid).status == "scheduled"

    # Unchanged handle: many ticks, never promoted.
    for i in range(5):
        assert kwo.poll_wait_on(conn, status_fn=w.status, now=int(time.time()) + 1000 * (i + 1),
                                poll_seconds=0) == []
    assert kb.get_task(conn, tid).status == "scheduled"
    assert kb.claim_task(conn, tid) is None

    # Re-run attempt 2 goes green: promoted once, handle cleared.
    w.fp[h] = "completed|success|2|aaa"
    assert kwo.poll_wait_on(conn, status_fn=w.status, now=int(time.time()) + 9000, poll_seconds=0) == [tid]
    t = kb.get_task(conn, tid)
    assert t.status == "ready"
    row = conn.execute("SELECT wait_on, wait_on_fingerprint FROM tasks WHERE id=?", (tid,)).fetchone()
    assert row["wait_on"] is None and row["wait_on_fingerprint"] is None
    ev = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='wait_on_changed'",
                      (tid,)).fetchone()
    assert json.loads(ev["payload"])["passing"] is True


def test_poll_is_rate_limited(conn):
    w = World()
    h = "pr:o/r#7"
    w.fp[h] = "sha1|OPEN|ci=failure"
    tid, run = _running(conn)
    kwo.wait_on_task(conn, tid, h, expected_run_id=run, status_fn=w.status)
    w.calls = 0
    now = int(time.time())
    kwo.poll_wait_on(conn, status_fn=w.status, now=now + 10, poll_seconds=120)
    assert w.calls == 0  # parked just now -> not due
    kwo.poll_wait_on(conn, status_fn=w.status, now=now + 200, poll_seconds=120)
    kwo.poll_wait_on(conn, status_fn=w.status, now=now + 210, poll_seconds=120)
    assert w.calls == 1  # second tick within the window skipped


def test_max_polls_per_tick(conn):
    w = World()
    ids = []
    for i in range(4):
        tid, run = _running(conn, title=f"c{i}")
        kwo.wait_on_task(conn, tid, f"gh-run:o/r:{i}", expected_run_id=run, status_fn=w.status)
        ids.append(tid)
    w.calls = 0
    kwo.poll_wait_on(conn, status_fn=w.status, now=int(time.time()) + 999, poll_seconds=0, max_polls=2)
    assert w.calls == 2


def test_unobservable_handle_stays_parked(conn):
    w = World()  # status returns None (gh down)
    tid, run = _running(conn)
    kwo.wait_on_task(conn, tid, "gh-run:o/r:1", expected_run_id=run, status_fn=w.status)
    assert kwo.poll_wait_on(conn, status_fn=w.status, now=int(time.time()) + 999, poll_seconds=0) == []
    assert kb.get_task(conn, tid).status == "scheduled"


def test_card_handle_uses_board_state(conn):
    dep = kb.create_task(conn, title="dep", assignee="software-engineer")
    tid, run = _running(conn, title="waits on dep")
    kwo.wait_on_task(conn, tid, f"card:{dep}", expected_run_id=run)
    assert kwo.poll_wait_on(conn, now=int(time.time()) + 999, poll_seconds=0) == []
    kb.complete_task(conn, dep, result="done")
    assert kwo.poll_wait_on(conn, now=int(time.time()) + 999, poll_seconds=0) == [tid]


def test_default_fingerprints_and_passing():
    assert kwo.fingerprint_run({"status": "completed", "conclusion": "failure", "attempt": 1, "headSha": "a"}) \
        == "completed|failure|1|a"
    fp = kwo.fingerprint_pr({"headRefOid": "ABC", "state": "OPEN", "statusCheckRollup": [
        {"name": "b", "conclusion": "SUCCESS"}, {"context": "a", "state": "FAILURE"}]})
    assert fp == "abc|OPEN|a=failure,b=success"
    assert not kwo.is_passing(fp)
    assert kwo.is_passing("completed|success|2|a")
    assert kwo.is_passing("done|123")


def _blocked_run(conn, tid, head):
    claimed = kb.claim_task(conn, tid)
    kwo.record_run_head(conn, claimed.current_run_id, head)
    assert kb.schedule_task(conn, tid, reason="CI still red",
                            expected_run_id=claimed.current_run_id)
    kb.unblock_task(conn, tid)


def test_two_no_progress_runs_go_to_architect(conn):
    tid = kb.create_task(conn, title="impl", assignee="software-engineer")
    head = {"v": "c1"}
    head_fn = lambda _ws: head["v"]  # noqa: E731
    _blocked_run(conn, tid, "c1")
    assert kwo.no_progress_streak(conn, tid, "c1") == 1
    assert kwo.no_progress_guard(conn, tid, "software-engineer", None, head_fn=head_fn,
                                 architect="architect-critic-fable", limit=2) is None
    _blocked_run(conn, tid, "c1")
    assert kwo.no_progress_guard(conn, tid, "software-engineer", None, head_fn=head_fn,
                                 architect="architect-critic-fable", limit=2) == "no_progress_respec"
    t = kb.get_task(conn, tid)
    assert t.assignee == "architect-critic-fable"
    assert any("[no-progress]" in c.body for c in kb.list_comments(conn, tid))
    # Streak resets after re-spec: the next implementer run starts clean.
    assert kwo.no_progress_streak(conn, tid, "c1") == 0


def test_new_commit_is_progress(conn):
    tid = kb.create_task(conn, title="impl", assignee="software-engineer")
    _blocked_run(conn, tid, "c1")
    _blocked_run(conn, tid, "c2")  # HEAD moved c2 -> c3 during this run
    assert kwo.no_progress_streak(conn, tid, "c3") == 0
    assert kwo.no_progress_streak(conn, tid, "c2") == 1


def test_status_change_is_progress(conn):
    tid = kb.create_task(conn, title="impl", assignee="software-engineer")
    _blocked_run(conn, tid, "c1")
    claimed = kb.claim_task(conn, tid)
    kwo.record_run_head(conn, claimed.current_run_id, "c1")
    assert kb.request_review(conn, tid, summary="ready", expected_run_id=claimed.current_run_id)
    assert kwo.no_progress_streak(conn, tid, "c1") == 0


def test_dispatcher_skips_parked_and_respecs(conn, monkeypatch):
    """End to end through dispatch_once: a parked card is never spawned; two
    no-progress runs re-route the next spawn to the architect."""
    monkeypatch.setattr(kb, "_design_phase_cfg", lambda: None)
    monkeypatch.setattr(kb, "_profile_exists_fn", lambda: None)
    monkeypatch.setattr(kwo, "respec_assignee", lambda: "architect-critic-fable")
    monkeypatch.setattr(kwo, "default_head", lambda _ws: None)
    monkeypatch.setattr(kwo, "default_status", lambda h, c: "same")
    spawned = []
    tid, run = _running(conn)
    kwo.wait_on_task(conn, tid, "gh-run:o/r:9", expected_run_id=run, status_fn=lambda h, c: "same")
    for _ in range(3):
        kb.dispatch_once(conn, spawn_fn=lambda t, ws, b=None: spawned.append(t.id) or None)
    assert tid not in spawned

    other = kb.create_task(conn, title="loop", assignee="software-engineer")
    for _ in range(2):
        c = kb.claim_task(conn, other)
        kb.schedule_task(conn, other, reason="still red", expected_run_id=c.current_run_id)
        kb.unblock_task(conn, other)
    res = kb.dispatch_once(conn, spawn_fn=lambda t, ws, b=None: spawned.append((t.id, t.assignee)) or None)
    assert (other, "no_progress_respec") in res.respawn_guarded
    assert kb.get_task(conn, other).assignee == "architect-critic-fable"
