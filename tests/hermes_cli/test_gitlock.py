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


def test_proc_cwd_is_none_for_a_pid_the_kernel_cannot_resolve() -> None:
    """``_proc_cwd`` must report "unknown", not a fabricated path (the F1 defect).

    ``os.path.realpath("/proc/<dead>/cwd")`` returns the input path instead of raising, so the
    conservative ``cwd is None`` branch in ``_git_pids_working_in`` never ran and every pid on a
    host without /proc was classified as "not working in this repo".
    """
    import hermes_cli.gitlock as gitlock

    absent = max(
        (int(p.name) for p in Path("/proc").iterdir() if p.name.isdigit()), default=1
    ) + 1000 if Path("/proc").is_dir() else 999999
    assert gitlock._proc_cwd(absent) is None


@pytest.mark.platforms("linux")
def test_proc_cwd_reports_a_deleted_cwd_as_unknown(tmp_path: Path) -> None:
    """A cwd the kernel marks "(deleted)" must read as unknown, not as some other directory."""
    import hermes_cli.gitlock as gitlock

    gone = tmp_path / "gone"
    gone.mkdir()
    previous = os.getcwd()
    try:
        os.chdir(gone)
        gone.rmdir()
        assert gitlock._proc_cwd(os.getpid()) is None
    finally:
        os.chdir(previous)


def test_git_proc_running_stays_conservative_when_the_cwd_is_unknown(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No /proc (macOS/Windows) or an unreadable pid must keep the sweep gated, not open it."""
    import hermes_cli.gitlock as gitlock

    monkeypatch.setattr(gitlock, "_git_pids", lambda: (4242,))
    monkeypatch.setattr(gitlock, "_proc_cwd", lambda _pid: None)

    assert gitlock._git_proc_running(repo) is True


def test_tasklist_csv_yields_the_git_pids() -> None:
    """``tasklist /FO CSV`` quotes every field, so a whitespace tokeniser finds no PID at all."""
    import hermes_cli.gitlock as gitlock

    output = (
        '"Image Name","PID","Session Name","Session#","Mem Usage"\r\n'
        '"git.exe","1234","Console","1","12,345 K"\r\n'
        '"git.exe","5678","Services","0","1,234 K"\r\n'
        '"notepad.exe","9999","Console","1","9,999 K"\r\n'
    )

    assert gitlock._pids_from_tasklist_csv(output) == (1234, 5678)
    assert gitlock._pids_from_tasklist_csv('INFO: No tasks are running which match the specified criteria.\r\n') == ()


@pytest.mark.platforms("posix")
def test_sweep_refuses_a_held_lock_without_a_proc_fd_view(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """macOS shape: no /proc fd view. The sweep must still ask lsof before unlinking.

    On that platform ``_proc_fd_scan_available()`` is False and ``_open_holders`` falls back to
    ``lsof -t``. Without the fallback the sweep had no holder check at all there and removed a
    lock a live process held open (the F1 data-loss case).
    """
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
        while time.time() < deadline and holder.poll() is None:
            if gitlock._proc_fd_holders(lock):
                break
            if not Path("/proc/self/fd").is_dir():
                time.sleep(0.3)  # no fd view here: give the child time to open the file
                break
            time.sleep(0.05)
        assert holder.poll() is None, "holder process died before the sweep ran"

        monkeypatch.setattr(gitlock, "_proc_fd_scan_available", lambda: False)
        monkeypatch.setattr(gitlock, "_lsof_holders",
                            lambda path: (holder.pid,) if os.path.realpath(path) == os.path.realpath(lock) else ())
        monkeypatch.setattr(gitlock, "_git_pids", lambda: ())

        assert gitlock._open_holders([lock]) == {lock: (holder.pid,)}
        assert clear_stale_git_locks(repo) == []
        assert lock.exists()
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_sweep_still_clears_a_stale_lock_without_a_proc_fd_view(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lsof fallback must not turn into a permanent skip: no holder -> the lock is removed."""
    import hermes_cli.gitlock as gitlock

    monkeypatch.setattr(gitlock, "_proc_fd_scan_available", lambda: False)
    monkeypatch.setattr(gitlock, "_lsof_holders", lambda _path: ())
    monkeypatch.setattr(gitlock, "_git_pids", lambda: ())
    _touch(repo / ".git" / "index.lock", STALE_LOCK_MIN_AGE_SECONDS + 60)

    removed = clear_stale_git_locks(repo)

    assert str(repo / ".git" / "index.lock") in removed
    assert not (repo / ".git" / "index.lock").exists()


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

