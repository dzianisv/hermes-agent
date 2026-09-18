"""The dashboard's direct status write must contain the worker it orphans.

``plugins/kanban/dashboard/plugin_api._set_status_direct`` is the drag-drop
status write behind the Kanban dashboard. Moving a card OFF ``running``
conditionally NULLs ``claim_lock``/``claim_expires``/``worker_pid``
(``CASE WHEN ? = 'running' THEN ... ELSE NULL END``) while the worker process
is still live — the same defect class as the in-module third-party releases,
one module away.

It is routed through ``kanban_db._terminate_released_worker`` today, but
until this file nothing DETECTED it stopping being routed: emptying the
post-commit drain loop left the whole repo suite byte-identically green.

REAL subprocesses, REAL lifecycle functions, an isolated temp board — the
harness (`_Writer`, the board helpers, the fixtures) is imported from
``test_kanban_release_termination`` rather than re-invented, because only a
live writer can prove the defect.
"""

from __future__ import annotations

import time

import pytest

from hermes_cli import kanban_db as kb
from plugins.kanban.dashboard.plugin_api import _set_status_direct
from tests.hermes_cli.test_kanban_worker_tree_termination import _wait_gone
from tests.hermes_cli.test_kanban_release_termination import (  # noqa: F401
    _claimable,
    _claimed_card_with_writer,
    _events,
    _noop_signal,
    _state,
    conn,
    kanban_home,
    pytestmark,
    writers,
)


@pytest.fixture
def sabotaged_signal(monkeypatch: pytest.MonkeyPatch):
    """Make termination FAIL without changing anything else.

    The real primitive still runs — ownership guard, hold, event recording —
    only the signal delivery is a no-op, exactly the sabotage the in-module
    survival cases use via ``signal_fn=_noop_signal``. ``_set_status_direct``
    does not take a ``signal_fn``, so it is injected at the primitive.
    """
    real = kb._terminate_released_worker

    def _patched(*args, **kwargs):
        kwargs.setdefault("signal_fn", _noop_signal)
        return real(*args, **kwargs)

    monkeypatch.setattr(kb, "_terminate_released_worker", _patched)
    return _patched


# ---------------------------------------------------------------------------
# (a) a surviving worker => the card is HELD, never left claimable
# ---------------------------------------------------------------------------


def test_dashboard_status_move_holds_the_card_when_the_worker_survives(
    conn, writers, sabotaged_signal,
) -> None:
    writer = writers("dashboard_survive")
    tid, _run_id = _claimed_card_with_writer(conn, writer, "drag me off running")

    assert _set_status_direct(conn, tid, "ready") is True
    released_at = time.time()

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
        f"dashboard status move left the card CLAIMABLE in status "
        f"{state['status']} beside surviving writer {writer.pid}"
    )
    assert state["worker_pid"] == writer.pid
    assert state["claim_expires"] is not None
    assert not _claimable(conn, tid)

    # The hold is what stops a SECOND worker joining the live one.
    kb.recompute_ready(conn)
    assert kb.claim_task(conn, tid) is None, (
        "a SECOND worker could claim a card held beside a live writer"
    )
    assert kb.claim_review_task(conn, tid) is None

    # The containment failure is recorded, not silent.
    records = _events(conn, tid, "reclaim_deferred") + _events(
        conn, tid, "release_containment_failed"
    )
    assert records, "containment failure was not recorded as an event"
    last = records[-1]
    assert last.get("termination_attempted") is True
    assert last.get("host_local") is True
    assert last.get("terminated") is False


# ---------------------------------------------------------------------------
# (b) verified termination => zero writes land after the release
# ---------------------------------------------------------------------------


def test_dashboard_status_move_terminates_the_worker_and_stops_the_writes(
    conn, writers,
) -> None:
    writer = writers("dashboard_terminate")
    tid, _run_id = _claimed_card_with_writer(conn, writer, "yank me back")

    before = writer.total_writes()
    assert before > 0, "precondition: the writer must already be writing"

    assert _set_status_direct(conn, tid, "ready") is True
    released_at = time.time()

    assert _state(conn, tid)["status"] in {"ready", "todo", "review"}, _state(
        conn, tid
    )

    assert _wait_gone(writer.pid, timeout=10), (
        "the released worker tree must be terminated"
    )
    time.sleep(1.5)  # >= 7 write intervals
    after = writer.writes_after(released_at)
    assert after == 0, (
        f"{after} write(s) from the released worker landed AFTER the "
        f"dashboard status move (total {writer.total_writes()}, "
        f"before {before})"
    )


def test_dashboard_status_move_reaps_the_workers_child(conn, writers) -> None:
    """Containment takes the whole owned tree, not just the worker pid."""
    writer = writers("dashboard_tree")
    tid, _run_id = _claimed_card_with_writer(conn, writer, "tree teardown")

    assert _set_status_direct(conn, tid, "ready") is True

    assert _wait_gone(writer.pid, timeout=10), "worker must die"
    assert _wait_gone(writer.child_pid, timeout=10), (
        f"the worker's child {writer.child_pid} leaked"
    )
