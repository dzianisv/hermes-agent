"""Worker termination must take out the worker's DESCENDANTS too.

Dispatcher workers are spawned with ``start_new_session=True``, so each
worker leads its own session/process GROUP. Signalling only the worker pid
leaves its children (the observed ``gtimeout`` -> ``pi`` orphan chain)
running against a task that has already been handed to a new worker.

These tests cover both termination paths (reclaim and max-runtime timeout)
with REAL subprocesses, plus the load-bearing safety property: we only ever
group-signal a pid that is ITSELF the group leader — a non-leader pid gets
a single-pid signal, because its group belongs to somebody else.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


pytestmark = pytest.mark.skipif(
    not hasattr(os, "killpg"), reason="POSIX process groups required"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _alive(pid: int) -> bool:
    return kb._pid_alive(pid)


def _wait_gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


class _Tree:
    """A real worker process group: leader shell + a sleeping grandchild."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            ["/bin/sh", "-c", "sleep 300 & echo $! ; sleep 300"],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert self.proc.stdout is not None
        self.child_pid = int(self.proc.stdout.readline().strip())
        self.pid = self.proc.pid
        # Precondition: the worker really is its own group leader, and the
        # grandchild inherited that group without leading it.
        assert os.getpgid(self.pid) == self.pid
        assert os.getpgid(self.child_pid) == self.pid
        assert _alive(self.pid) and _alive(self.child_pid)

    def cleanup(self) -> None:
        for pid in (self.child_pid, self.pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass


@pytest.fixture
def tree():
    t = _Tree()
    try:
        yield t
    finally:
        t.cleanup()


@pytest.fixture
def bystander():
    """A process spawned OUTSIDE the worker's group. Must always survive."""
    proc = subprocess.Popen(["sleep", "300"])
    try:
        yield proc
    finally:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# _signal_worker_tree: routing + the safety property (injected fakes)
# ---------------------------------------------------------------------------


def test_group_leader_is_group_signalled_not_pid_signalled() -> None:
    kills: list[tuple[int, int]] = []
    pgkills: list[tuple[int, int]] = []
    kb._signal_worker_tree(
        4242, signal.SIGTERM,
        kill=lambda p, s: kills.append((p, s)),
        killpg=lambda p, s: pgkills.append((p, s)),
        getpgid=lambda p: p,  # leader: pgid == pid
    )
    assert pgkills == [(4242, signal.SIGTERM)]
    assert kills == []


def test_non_leader_pid_is_signalled_alone_and_its_group_is_spared() -> None:
    """Load-bearing safety property.

    A pid whose group leader is somebody ELSE must never be group-signalled:
    that group is not ours and killpg would take out unrelated processes.
    """
    kills: list[tuple[int, int]] = []
    pgkills: list[tuple[int, int]] = []
    kb._signal_worker_tree(
        4242, signal.SIGTERM,
        kill=lambda p, s: kills.append((p, s)),
        killpg=lambda p, s: pgkills.append((p, s)),
        getpgid=lambda _p: 999,  # foreign group leader
    )
    assert kills == [(4242, signal.SIGTERM)]
    assert pgkills == [], "must not killpg a group we do not own"


def test_pgid_is_derived_not_assumed_equal_to_pid() -> None:
    """getpgid is consulted; ``pgid == pid`` is never assumed."""
    seen: list[int] = []
    kb._signal_worker_tree(
        77, signal.SIGKILL,
        kill=lambda *_a: None,
        killpg=lambda *_a: None,
        getpgid=lambda p: seen.append(p) or p,
    )
    assert seen == [77]


@pytest.mark.parametrize("exc", [ProcessLookupError(), OSError(), AttributeError()])
def test_falls_back_to_pid_signal_when_getpgid_fails(exc) -> None:
    kills: list[tuple[int, int]] = []
    pgkills: list[tuple[int, int]] = []

    def _boom(_p):
        raise exc

    kb._signal_worker_tree(
        11, signal.SIGTERM,
        kill=lambda p, s: kills.append((p, s)),
        killpg=lambda p, s: pgkills.append((p, s)),
        getpgid=_boom,
    )
    assert kills == [(11, signal.SIGTERM)]
    assert pgkills == []


def test_falls_back_to_pid_signal_when_killpg_is_unavailable() -> None:
    """Windows has neither getpgid nor killpg."""
    kills: list[tuple[int, int]] = []
    kb._signal_worker_tree(
        12, signal.SIGTERM,
        kill=lambda p, s: kills.append((p, s)),
        killpg=None,
        getpgid=None,
    )
    assert kills == [(12, signal.SIGTERM)]


# ---------------------------------------------------------------------------
# Real processes: the safety property, live
# ---------------------------------------------------------------------------


def test_real_non_leader_pid_does_not_take_down_its_group(tree: _Tree) -> None:
    """Signalling the grandchild (a non-leader) kills only the grandchild."""
    kb._signal_worker_tree(tree.child_pid, signal.SIGKILL)
    assert _wait_gone(tree.child_pid)
    time.sleep(0.3)
    assert _alive(tree.pid), "the group leader must survive a non-leader signal"


# ---------------------------------------------------------------------------
# Reclaim path (_terminate_reclaimed_worker)
# ---------------------------------------------------------------------------


def test_reclaim_termination_kills_the_whole_worker_tree(
    tree: _Tree, bystander: subprocess.Popen
) -> None:
    info = kb._terminate_reclaimed_worker(tree.pid, kb._claimer_id())
    assert info["host_local"] is True
    assert info["termination_attempted"] is True

    assert _wait_gone(tree.pid), "worker must die"
    assert _wait_gone(tree.child_pid), (
        "descendant leaked: the orphan chain survived the reclaim"
    )
    assert info["terminated"] is True
    assert bystander.poll() is None, "unrelated process must survive a reclaim"


def test_reclaim_termination_ignores_non_host_local_claims(
    tree: _Tree,
) -> None:
    info = kb._terminate_reclaimed_worker(tree.pid, "some-other-host:1234")
    assert info["host_local"] is False
    assert info["termination_attempted"] is False
    time.sleep(0.3)
    assert _alive(tree.pid) and _alive(tree.child_pid)


def test_reclaim_termination_reports_terminated_for_a_dead_pid(
    tree: _Tree,
) -> None:
    tree.cleanup()
    assert _wait_gone(tree.pid)
    info = kb._terminate_reclaimed_worker(tree.pid, kb._claimer_id())
    assert info["termination_attempted"] is True
    assert info["terminated"] is True


# ---------------------------------------------------------------------------
# Timeout path (enforce_max_runtime)
# ---------------------------------------------------------------------------


def test_max_runtime_termination_kills_the_whole_worker_tree(
    kanban_home: Path, tree: _Tree, bystander: subprocess.Popen
) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="overrunning job", assignee="worker",
            max_runtime_seconds=1,
        )
        assert kb.claim_task(conn, tid) is not None
        kb._set_worker_pid(conn, tid, tree.pid)
        old = int(time.time()) - 300
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (old, tid))
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (old, tid),
            )

        assert tid in kb.enforce_max_runtime(conn)

    assert _wait_gone(tree.pid), "worker must die on timeout"
    assert _wait_gone(tree.child_pid), (
        "descendant leaked: the orphan chain survived the timeout reap"
    )
    assert bystander.poll() is None, "unrelated process must survive a timeout reap"


def test_injected_signal_fn_intercepts_the_group_signal(
    kanban_home: Path,
) -> None:
    """The existing ``signal_fn`` test hook still sees every signal.

    Regression guard: if the group path bypassed the hook, a test using a
    live pid would fire a real killpg at the test runner's own group.
    """
    seen: list[tuple[int, int]] = []
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="hooked", assignee="worker", max_runtime_seconds=1,
        )
        assert kb.claim_task(conn, tid) is not None
        kb._set_worker_pid(conn, tid, os.getpid())
        old = int(time.time()) - 300
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (old, tid))
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (old, tid),
            )
        orig_alive = kb._pid_alive
        kb._pid_alive = lambda _pid: False
        try:
            assert tid in kb.enforce_max_runtime(
                conn, signal_fn=lambda p, s: seen.append((p, s)),
            )
        finally:
            kb._pid_alive = orig_alive

    assert seen, "signal_fn must still be called"
    assert all(s == signal.SIGTERM for _p, s in seen)
    assert all(p in (os.getpid(), os.getpgid(os.getpid())) for p, _s in seen)
