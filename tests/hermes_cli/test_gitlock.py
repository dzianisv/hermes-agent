"""Tests for hermes_cli.gitlock — stale git lock recovery + ancestry probe.

These cover the two failure modes that produced the false "update available"
notification and the hard ``update --check`` failure after a crashed fetch on
a shallow clone:

1. A stale ``.git/shallow.lock`` makes every later ``git fetch`` fail with
   "File exists" unless cleared.
2. On a shallow clone the update check compares tip SHAs, so local
   cherry-picks on top of the remote tip look like "update available" even
   though HEAD already contains the remote tip.

The module's safety rules are also pinned: a *young* lock or a *running git
process* must never be cleared.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli.gitlock import (
    LOCK_NAMES,
    STALE_LOCK_MIN_AGE_SECONDS,
    clear_stale_git_locks,
    describe_lock_file,
    describe_repo_lock,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real, tiny git repo with two commits (no network)."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    (root / "a.txt").write_text("one\n")
    subprocess.run(["git", "add", "a.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "first"], cwd=root, check=True)
    (root / "b.txt").write_text("two\n")
    subprocess.run(["git", "add", "b.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "second"], cwd=root, check=True)
    return root


def _touch(path: Path, age_seconds: float) -> None:
    """Create (or truncate) a file and backdate its mtime."""
    path.touch()
    old = time.time() - age_seconds
    os.utime(path, (old, old))


@pytest.fixture
def no_git_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the process guard: pretend no git process is running.

    The removal tests exercise the sweep itself, not the guard.  Without
    this pin they are flaky on CI: the parallel per-file test runner is
    almost always running a real ``git`` subprocess somewhere, the
    ``pgrep -x git`` probe hits, and the sweep (correctly) refuses to
    remove anything — failing the assertion for reasons unrelated to the
    code under test.  The guard's own behavior is pinned separately in
    :func:`test_clear_skips_sweep_while_git_running`.
    """
    import hermes_cli.gitlock as gitlock

    monkeypatch.setattr(gitlock, "_git_proc_running", lambda *args: False)


def test_clear_removes_stale_shallow_lock(repo: Path, no_git_running: None) -> None:
    _touch(repo / ".git" / "shallow.lock", STALE_LOCK_MIN_AGE_SECONDS + 60)
    removed = clear_stale_git_locks(repo)
    assert str(repo / ".git" / "shallow.lock") in removed
    assert not (repo / ".git" / "shallow.lock").exists()


def test_clear_removes_all_stale_lock_kinds(repo: Path, no_git_running: None) -> None:
    for name in LOCK_NAMES:
        _touch(repo / ".git" / name, STALE_LOCK_MIN_AGE_SECONDS + 60)
    removed = clear_stale_git_locks(repo)
    assert len(removed) == len(LOCK_NAMES)
    for name in LOCK_NAMES:
        assert not (repo / ".git" / name).exists()


def test_clear_skips_sweep_while_git_running(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A running git process must block the sweep even for stale locks."""
    import hermes_cli.gitlock as gitlock

    monkeypatch.setattr(gitlock, "_git_proc_running", lambda *args: True)
    _touch(repo / ".git" / "shallow.lock", STALE_LOCK_MIN_AGE_SECONDS + 60)
    removed = clear_stale_git_locks(repo)
    assert removed == []
    assert (repo / ".git" / "shallow.lock").exists()


def test_clear_keeps_young_lock(repo: Path, no_git_running: None) -> None:
    _touch(repo / ".git" / "shallow.lock", 1)  # 1 second old — presumably live
    removed = clear_stale_git_locks(repo)
    assert removed == []
    assert (repo / ".git" / "shallow.lock").exists()


def test_clear_noop_on_non_repo(tmp_path: Path) -> None:
    bare = tmp_path / "not-a-repo"
    bare.mkdir()
    assert clear_stale_git_locks(bare) == []


def test_clear_noop_with_no_locks(repo: Path) -> None:
    assert clear_stale_git_locks(repo) == []


def test_describe_repo_lock_is_silent_on_a_clean_repo(repo: Path) -> None:
    assert describe_repo_lock(repo) is None


def test_describe_repo_lock_names_the_dead_lock_with_age_and_probes(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The incident's lock was 0 bytes and ~6.5 hours old with no holder."""
    import hermes_cli.gitlock as gitlock

    monkeypatch.setattr(gitlock, "_git_pids", lambda: ())
    _touch(repo / ".git" / "index.lock", 6 * 3600 + 31 * 60)

    message = describe_repo_lock(repo)

    assert message is not None
    assert "stale index.lock" in message
    assert "6h 31m" in message
    assert "no live process holds it" in message
    assert "Safe to remove" in message
    assert str(repo / ".git" / "index.lock") in message
    assert "lsof" in message and "pgrep -x git" in message


def test_describe_lock_file_never_claims_safety_without_a_probe(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No /proc fd view and no lsof: the report must send the reader to check by hand."""
    import hermes_cli.gitlock as gitlock

    monkeypatch.setattr(gitlock, "_proc_fd_holders", lambda _path: None)
    monkeypatch.setattr(gitlock, "_lsof_holders", lambda _path: None)
    monkeypatch.setattr(gitlock, "_git_pids", lambda: ())
    lock = repo / ".git" / "index.lock"
    lock.touch()

    message = describe_lock_file(lock)

    assert message is not None
    assert "No liveness probe is available" in message
    assert "Safe to remove" not in message


@pytest.mark.platforms("linux")
def test_clear_refuses_a_lock_a_live_process_holds_open(repo: Path, no_git_running: None) -> None:
    """The sweep may not remove a lock whose inode a live process still has open."""
    import subprocess as sp
    import sys as _sys

    import hermes_cli.gitlock as gitlock

    lock = repo / ".git" / "index.lock"
    _touch(lock, STALE_LOCK_MIN_AGE_SECONDS + 60)
    holder = sp.Popen(
        [_sys.executable, "-c", "import sys, time; f = open(sys.argv[1], 'rb'); time.sleep(60)", str(lock)],
    )
    try:
        deadline = time.time() + 10
        while time.time() < deadline and not gitlock._proc_fd_holders(lock):
            time.sleep(0.05)
        if not gitlock._proc_fd_holders(lock):
            pytest.skip("no /proc fd visibility on this host")

        assert clear_stale_git_locks(repo) == []
        assert lock.exists()
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_git_proc_running_is_scoped_to_the_repo(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A long-lived git daemon elsewhere must not read as a writer of this repo.

    ``git credential-cache--daemon`` keeps the comm ``git`` for days; a machine-wide probe made the
    sweep (and the lock report) skip every time it was running.
    """
    import hermes_cli.gitlock as gitlock

    monkeypatch.setattr(gitlock, "_git_pids", lambda: (4242,))
    monkeypatch.setattr(gitlock, "_proc_cwd", lambda _pid: "/somewhere/else")

    assert gitlock._git_proc_running(repo) is False
    # Unscoped callers keep the conservative machine-wide answer.
    assert gitlock._git_proc_running() is True

    monkeypatch.setattr(gitlock, "_proc_cwd", lambda _pid: str(repo))
    assert gitlock._git_proc_running(repo) is True


def test_sweep_clears_a_stale_lock_while_an_unrelated_git_process_runs(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cleaner must still work when some other checkout's git (or a daemon) is alive."""
    import hermes_cli.gitlock as gitlock

    monkeypatch.setattr(gitlock, "_git_pids", lambda: (4242,))
    monkeypatch.setattr(gitlock, "_proc_cwd", lambda _pid: "/somewhere/else")
    _touch(repo / ".git" / "index.lock", STALE_LOCK_MIN_AGE_SECONDS + 60)

    removed = clear_stale_git_locks(repo)

    assert str(repo / ".git" / "index.lock") in removed
    assert not (repo / ".git" / "index.lock").exists()

