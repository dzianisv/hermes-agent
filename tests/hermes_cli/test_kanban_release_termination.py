"""Third-party claim releases must contain the worker they orphan.

REAL subprocesses, REAL lifecycle functions, an isolated temp board. No
mocks: the whole measured defect was that a release NULLs
``claim_lock``/``worker_pid`` while a live process keeps writing into the
card's workspace, and only a live writer can prove that.

Every case here drives one of the six third-party release variants
(``block_task`` plain / ``kind="dependency"`` / loop-breaker to ``triage``,
``reclaim_task``, ``reassign_task``, ``schedule_task``) through the single
shared containment primitive ``_terminate_released_worker``:

1. a third-party release never leaves the card claimable beside a live writer,
2. after VERIFIED termination there are ZERO writes stamped after the release,
3. when termination FAILS the card is HELD (claim re-asserted, not claimable)
   and the containment failure is recorded as an event,
4. a SELF transition (the worker's own ``expected_run_id``) never signals the
   caller's own process,
5. the worker's own child dies with it.

Process helpers are imported from ``test_kanban_worker_tree_termination``
rather than duplicated.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from tests.hermes_cli.test_kanban_worker_tree_termination import (
    _alive,
    _descendants_of,
    _wait_gone,
)


pytestmark = [
    pytest.mark.skipif(
        not hasattr(os, "killpg"), reason="POSIX process groups required"
    ),
    # Real worker trees, real signals; descendants get reparented to init the
    # moment the worker dies, which is outside the live-system guard's subtree.
    pytest.mark.live_system_guard_bypass,
]


# ---------------------------------------------------------------------------
# The writer: a real worker that leaves durable, timestamped evidence
# ---------------------------------------------------------------------------

_WRITER_SRC = r"""
import os
import subprocess
import sys
import time

evidence = sys.argv[1]
# A child of our own: termination must take the whole tree, not just us.
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
sys.stdout.write("%d %d\n" % (os.getpid(), child.pid))
sys.stdout.flush()
while True:
    with open(evidence, "a") as fh:
        fh.write("%d %.6f\n" % (os.getpid(), time.time()))
        fh.flush()
    time.sleep(0.2)
"""


class _Writer:
    """A live worker process writing ``<pid> <time.time()>`` every ~0.2s."""

    def __init__(self, tmp_path: Path, name: str) -> None:
        script = tmp_path / "writer_script.py"
        if not script.exists():
            script.write_text(_WRITER_SRC)
        self.evidence = tmp_path / f"{name}.evidence"
        self.evidence.write_text("")
        self.proc = subprocess.Popen(
            [sys.executable, str(script), str(self.evidence)],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert self.proc.stdout is not None
        pid, child_pid = self.proc.stdout.readline().split()
        self.pid = int(pid)
        self.child_pid = int(child_pid)
        assert self.pid == self.proc.pid
        assert os.getsid(self.pid) == self.pid, "worker must lead its session"
        # Precondition: it really is writing before we release anything.
        deadline = time.time() + 10.0
        while time.time() < deadline and not self._lines():
            time.sleep(0.05)
        assert self._lines(), "writer produced no evidence before the release"
        assert _alive(self.child_pid)

    def _lines(self) -> list[str]:
        return [
            ln for ln in self.evidence.read_text().splitlines() if ln.strip()
        ]

    def writes_after(self, t0: float) -> int:
        return sum(1 for ln in self._lines() if float(ln.split()[1]) > t0)

    def total_writes(self) -> int:
        return len(self._lines())

    def alive(self) -> bool:
        # Popen keeps the child reapable; a zombie is NOT a live writer, so
        # poll() first and only then fall back to the pid probe.
        return self.proc.poll() is None and _alive(self.pid)

    def cleanup(self) -> None:
        for pid in (self.child_pid, self.pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            pass
        for pid in (self.pid, self.child_pid):
            assert _wait_gone(pid, timeout=5), f"teardown failed to reap {pid}"


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home: Path):
    c = kb.connect()
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def writers(tmp_path: Path):
    made: list[_Writer] = []

    def _make(name: str) -> _Writer:
        w = _Writer(tmp_path, name)
        made.append(w)
        return w

    try:
        yield _make
    finally:
        for w in made:
            w.cleanup()


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------


def _claimed_card_with_writer(conn, writer: _Writer, title: str) -> tuple[str, int]:
    """Create + claim a card and bind ``writer`` to it as its worker."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    kb.recompute_ready(conn)
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None and claimed.status == "running"
    kb._set_worker_pid(conn, tid, writer.pid)
    row = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    assert row["current_run_id"] is not None
    return tid, int(row["current_run_id"])


def _prime_loop_breaker(conn, title: str) -> str:
    """Drive a real block/unblock cycle so the NEXT block routes to triage."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    kb.recompute_ready(conn)
    assert kb.block_task(conn, tid, reason="first hold") is True
    row = conn.execute(
        "SELECT status, block_recurrences FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    assert row["status"] == "blocked"
    assert int(row["block_recurrences"]) == kb.BLOCK_RECURRENCE_LIMIT - 1
    assert kb.unblock_task(conn, tid) is True
    return tid


def _state(conn, tid: str) -> dict:
    row = conn.execute(
        "SELECT status, claim_lock, claim_expires, worker_pid "
        "FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    return dict(row)


def _claimable(conn, tid: str) -> bool:
    """Both claim_task and claim_review_task require ``claim_lock IS NULL``."""
    return _state(conn, tid)["claim_lock"] is None


def _events(conn, tid: str, kind: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id", (tid, kind),
    ).fetchall()
    return [json.loads(r["payload"]) if r["payload"] else {} for r in rows]


# ---------------------------------------------------------------------------
# The six third-party release variants
# ---------------------------------------------------------------------------


def _setup_plain(conn, writer):
    return _claimed_card_with_writer(conn, writer, "block plain")


def _setup_triage(conn, writer):
    tid = _prime_loop_breaker(conn, "loop breaker")
    kb.recompute_ready(conn)
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None and claimed.status == "running"
    kb._set_worker_pid(conn, tid, writer.pid)
    row = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    return tid, int(row["current_run_id"])


def _release_block_plain(conn, tid, **kw):
    return kb.block_task(conn, tid, reason="third party hold", **kw)


def _release_block_dependency(conn, tid, **kw):
    return kb.block_task(
        conn, tid, reason="waiting on parent", kind="dependency", **kw
    )


def _release_block_triage(conn, tid, **kw):
    return kb.block_task(conn, tid, reason="first hold", **kw)


def _release_reclaim(conn, tid, **kw):
    return kb.reclaim_task(conn, tid, reason="operator abort", **kw)


def _release_reassign(conn, tid, **kw):
    return kb.reassign_task(
        conn, tid, "other-profile", reclaim_first=True,
        reason="operator reassign", **kw
    )


def _release_schedule(conn, tid, **kw):
    return kb.schedule_task(conn, tid, reason="timed park", **kw)


# (id, setup, release, expected landing status)
VARIANTS = [
    ("block_plain", _setup_plain, _release_block_plain, "blocked"),
    ("block_dependency", _setup_plain, _release_block_dependency, "todo"),
    ("block_triage_loop", _setup_triage, _release_block_triage, "triage"),
    ("reclaim", _setup_plain, _release_reclaim, "ready"),
    ("reassign", _setup_plain, _release_reassign, "ready"),
    ("schedule", _setup_plain, _release_schedule, "scheduled"),
]

_IDS = [v[0] for v in VARIANTS]


@pytest.fixture(params=VARIANTS, ids=_IDS)
def variant(request):
    name, setup, release, landing = request.param
    return dict(name=name, setup=setup, release=release, landing=landing)


# ---------------------------------------------------------------------------
# 1 + 2. Verified termination: no live writer beside a released card, and
#        ZERO writes stamped after the release.
# ---------------------------------------------------------------------------


def test_third_party_release_terminates_the_worker_and_stops_the_writes(
    conn, writers, variant,
) -> None:
    writer = writers(variant["name"])
    tid, _run_id = variant["setup"](conn, writer)

    before = writer.total_writes()
    assert before > 0, "precondition: the writer must already be writing"

    # Third party: no expected_run_id, so the caller did not prove ownership.
    assert variant["release"](conn, tid) is True
    released_at = time.time()

    state = _state(conn, tid)
    assert state["status"] == variant["landing"], state

    # (1) never terminal-or-claimable beside a live writer.
    if _claimable(conn, tid):
        assert not writer.alive(), (
            f"{variant['name']}: card is claimable in status "
            f"{state['status']} while writer {writer.pid} is still alive"
        )

    # (2) verified termination => zero writes stamped after the release.
    assert _wait_gone(writer.pid, timeout=10), "worker tree must be terminated"
    time.sleep(1.5)  # >= 7 write intervals
    after = writer.writes_after(released_at)
    assert after == 0, (
        f"{variant['name']}: {after} write(s) landed AFTER the release "
        f"(total writes {writer.total_writes()}, before {before})"
    )


# ---------------------------------------------------------------------------
# 5. Process tree: the worker's own child dies too.
# ---------------------------------------------------------------------------


def test_third_party_release_reaps_the_workers_child(
    conn, writers, variant,
) -> None:
    writer = writers(variant["name"] + "_tree")
    tid, _run_id = variant["setup"](conn, writer)

    descendants = _descendants_of(writer.pid)
    assert writer.child_pid in descendants, (
        f"precondition not established: child {writer.child_pid} not among "
        f"descendants {descendants}"
    )

    assert variant["release"](conn, tid) is True

    assert _wait_gone(writer.pid, timeout=10), "worker must die"
    assert _wait_gone(writer.child_pid, timeout=10), (
        f"{variant['name']}: the worker's child {writer.child_pid} leaked"
    )


# ---------------------------------------------------------------------------
# 3. Survival: termination sabotaged to fail => the card is HELD.
# ---------------------------------------------------------------------------


def _noop_signal(*_args, **_kwargs) -> None:
    """A signal_fn that does nothing: attempted + host-local + NOT terminated."""
    return None


def test_release_beside_a_surviving_worker_holds_the_card(
    conn, writers, variant,
) -> None:
    writer = writers(variant["name"] + "_survive")
    tid, _run_id = variant["setup"](conn, writer)

    ok = variant["release"](conn, tid, signal_fn=_noop_signal)
    released_at = time.time()

    # reclaim/reassign report the failed containment by refusing; block and
    # schedule still land the transition but must hold the claim.
    if variant["name"] in ("reclaim", "reassign"):
        assert ok is False, (
            f"{variant['name']} must refuse when containment failed"
        )
    else:
        assert ok is True

    time.sleep(1.0)
    assert writer.alive(), (
        "precondition not established: the sabotaged termination actually "
        "killed the writer, so this is not a survival case"
    )
    assert writer.writes_after(released_at) > 0, (
        "precondition not established: the surviving writer stopped writing"
    )

    state = _state(conn, tid)
    assert state["claim_lock"] is not None, (
        f"{variant['name']}: card left CLAIMABLE in status {state['status']} "
        f"beside surviving writer {writer.pid}"
    )
    assert state["worker_pid"] == writer.pid
    assert state["claim_expires"] is not None
    assert not _claimable(conn, tid)

    # The containment failure is recorded, with the termination dict inline.
    held = _events(conn, tid, "reclaim_deferred")
    failed = _events(conn, tid, "release_containment_failed")
    records = held + failed
    assert records, (
        f"{variant['name']}: containment failure was not recorded as an event"
    )
    last = records[-1]
    assert last.get("termination_attempted") is True
    assert last.get("host_local") is True
    assert last.get("terminated") is False


def test_a_held_card_cannot_be_reclaimed_into_a_second_worker(
    conn, writers, variant,
) -> None:
    """The hold is what stops the duplicate-writer loop."""
    writer = writers(variant["name"] + "_dup")
    tid, _run_id = variant["setup"](conn, writer)
    variant["release"](conn, tid, signal_fn=_noop_signal)

    assert _state(conn, tid)["claim_lock"] is not None
    kb.recompute_ready(conn)
    assert kb.claim_task(conn, tid) is None, (
        f"{variant['name']}: a SECOND worker could claim a held card"
    )
    assert kb.claim_review_task(conn, tid) is None


# ---------------------------------------------------------------------------
# 4. Self-transition safety: a worker releasing its OWN run is never signalled.
#    This is the single largest regression risk: every worker blocks itself.
# ---------------------------------------------------------------------------


def test_self_release_never_signals_the_callers_own_process(
    conn, writers, variant,
) -> None:
    writer = writers(variant["name"] + "_self")
    tid, run_id = variant["setup"](conn, writer)

    before = writer.total_writes()
    variant["release"](conn, tid, expected_run_id=run_id)
    released_at = time.time()

    time.sleep(1.0)
    assert writer.alive(), (
        f"{variant['name']}: a SELF release killed the caller's own worker "
        f"{writer.pid}"
    )
    assert _alive(writer.child_pid), "the caller's child was killed too"
    assert writer.writes_after(released_at) > 0, (
        f"{variant['name']}: the caller's own worker stopped writing after a "
        f"self release (total {writer.total_writes()}, before {before})"
    )


def test_self_release_is_not_signalled_even_when_signalling_would_work(
    conn, writers,
) -> None:
    """The guard is the run id, not the reachability of the process.

    Injecting a signal_fn that records everything proves nothing was even
    attempted, rather than inferring it from the worker's survival.
    """
    writer = writers("self_no_attempt")
    tid, run_id = _claimed_card_with_writer(conn, writer, "self block")
    seen: list[tuple] = []

    assert kb.block_task(
        conn, tid, reason="worker blocks itself",
        expected_run_id=run_id,
        signal_fn=lambda p, s: seen.append((p, s)),
    ) is True

    assert seen == [], f"a self block attempted to signal: {seen}"
    assert writer.alive()


# ---------------------------------------------------------------------------
# schedule_task: proof its behaviour is UNCHANGED by the refactor onto the
# shared primitive.
# ---------------------------------------------------------------------------


def test_schedule_park_still_terminates_and_records_source_status(
    conn, writers,
) -> None:
    writer = writers("schedule_unchanged")
    tid, _run_id = _claimed_card_with_writer(conn, writer, "park me")

    assert kb.schedule_task(conn, tid, reason="waiting on CI") is True
    state = _state(conn, tid)
    assert state["status"] == "scheduled"
    assert state["claim_lock"] is None
    assert _wait_gone(writer.pid, timeout=10)

    payloads = _events(conn, tid, "scheduled")
    assert payloads and payloads[-1]["source_status"] == "running"
    assert payloads[-1]["reason"] == "waiting on CI"


def test_schedule_self_park_is_still_never_signalled(conn, writers) -> None:
    writer = writers("schedule_self_unchanged")
    tid, run_id = _claimed_card_with_writer(conn, writer, "park my own run")
    seen: list[tuple] = []

    assert kb.schedule_task(
        conn, tid, reason="self park", expected_run_id=run_id,
        signal_fn=lambda p, s: seen.append((p, s)),
    ) is True
    assert seen == []
    assert writer.alive()


# ---------------------------------------------------------------------------
# block_recurrences / the loop breaker must be byte-identical.
# ---------------------------------------------------------------------------


def test_block_recurrence_counter_and_loop_breaker_are_unchanged(conn) -> None:
    """No worker involved: pure routing/counter behaviour."""
    tid = kb.create_task(conn, title="loop", assignee="worker")
    kb.recompute_ready(conn)

    assert kb.block_task(conn, tid, reason="a", kind="needs_input") is True
    row = _state(conn, tid)
    assert row["status"] == "blocked"
    assert kb.get_task(conn, tid).block_recurrences == 1

    assert kb.unblock_task(conn, tid) is True
    assert kb.block_task(conn, tid, reason="a", kind="needs_input") is True
    assert _state(conn, tid)["status"] == "triage"
    assert kb.get_task(conn, tid).block_recurrences == kb.BLOCK_RECURRENCE_LIMIT
    payload = _events(conn, tid, "block_loop_detected")[-1]
    assert payload["recurrences"] == kb.BLOCK_RECURRENCE_LIMIT
    assert payload["limit"] == kb.BLOCK_RECURRENCE_LIMIT
    assert payload["kind"] == "needs_input"


def test_different_cause_resets_the_recurrence_counter(conn) -> None:
    tid = kb.create_task(conn, title="mixed", assignee="worker")
    kb.recompute_ready(conn)
    assert kb.block_task(conn, tid, reason="a", kind="needs_input") is True
    assert kb.unblock_task(conn, tid) is True
    # Different cause => counter restarts at 1, so this must NOT hit triage.
    assert kb.block_task(conn, tid, reason="b", kind="capability") is True
    assert _state(conn, tid)["status"] == "blocked"
    assert kb.get_task(conn, tid).block_recurrences == 1


def test_dependency_block_still_routes_to_todo_without_a_worker(conn) -> None:
    tid = kb.create_task(conn, title="waiter", assignee="worker")
    kb.recompute_ready(conn)
    assert _state(conn, tid)["status"] == "ready"
    assert kb.block_task(
        conn, tid, reason="waiting", kind="dependency",
    ) is True
    assert _state(conn, tid)["status"] == "todo"
    assert _events(conn, tid, "dependency_wait")[-1]["kind"] == "dependency"
