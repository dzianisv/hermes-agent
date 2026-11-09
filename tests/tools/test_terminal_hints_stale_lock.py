"""A stale git lock must be reported as a dead lock — never as a merge conflict.

2026-10-21 (measured): a crashed git left a 0-byte ``.git/index.lock`` in the live runtime
checkout. Every later write failed with

    error: Unable to write index.
    Automatic merge failed; fix conflicts and then commit the result.

and the terminal hint told the operator to "resolve the conflicted files" — files that did not
conflict. The merge also left ``MERGE_HEAD`` behind, and ``git merge --abort`` failed on the same
lock. These tests pin the incident's real git output and the two safety rules: the hint names the
lock with its age and the liveness probe that cleared it, and a removal is never proposed while a
live process holds the lock.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

import hermes_cli.gitlock as gitlock
from tools.terminal_hints import annotate_failure


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real repo where a --no-ff merge is clean except for a planted index.lock."""
    root = tmp_path / "repo"
    root.mkdir()
    git = ["git", "-C", str(root)]
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True)
    subprocess.run([*git, "config", "user.email", "test@example.com"], check=True)
    subprocess.run([*git, "config", "user.name", "Test"], check=True)
    (root / "a.txt").write_text("base\n")
    subprocess.run([*git, "add", "a.txt"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "base"], check=True)
    subprocess.run([*git, "checkout", "-q", "-b", "fix/reviewed"], check=True)
    (root / "b.txt").write_text("fix\n")
    subprocess.run([*git, "add", "b.txt"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "reviewed fix"], check=True)
    subprocess.run([*git, "checkout", "-q", "main"], check=True)
    return root


@pytest.fixture
def quiet_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the machine-wide probes: the parallel test runner is itself running git.

    Without this the "safe to remove" branch is unreachable in CI for a reason unrelated to the
    code under test (see the same rationale in tests/hermes_cli/test_gitlock.py).
    """
    monkeypatch.setattr(gitlock, "_git_pids", lambda: ())


def _blocked_merge(repo: Path) -> tuple[str, int, str]:
    """Run the merge the incident ran, with an abandoned index.lock in the way."""
    lock = repo / ".git" / "index.lock"
    lock.touch()
    command = f"git -C {repo} merge --no-ff fix/reviewed"
    proc = subprocess.run(
        ["git", "-C", str(repo), "merge", "--no-ff", "fix/reviewed"],
        capture_output=True, text=True, check=False,
    )
    return command, proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def test_blocked_merge_names_the_lock_not_a_conflict(repo: Path, quiet_probes: None) -> None:
    command, returncode, output = _blocked_merge(repo)
    # The pre-change failure text, captured from real git: a write error wrapped in conflict wording.
    assert returncode != 0
    assert "Unable to write index" in output
    assert "fix conflicts" in output

    hint = annotate_failure(command, returncode, output, cwd=str(repo))

    assert hint is not None
    assert "index.lock" in hint
    assert "age " in hint
    assert "no live process holds it" in hint
    assert str(repo / ".git" / "index.lock") in hint
    assert "Safe to remove" in hint
    # The misleading instruction that cost hours: never present when no file conflicts.
    assert "Resolve the conflicted files" not in hint


def test_git_dash_c_target_wins_over_an_unrelated_cwd(repo: Path, quiet_probes: None, tmp_path: Path) -> None:
    """The incident's command shape was `git -C <repo> merge ...`; cwd was not the repo."""
    command, returncode, output = _blocked_merge(repo)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    hint = annotate_failure(command, returncode, output, cwd=str(elsewhere))

    assert hint is not None and str(repo / ".git" / "index.lock") in hint


def test_without_repo_context_the_hint_never_says_fix_conflicts(repo: Path) -> None:
    command, returncode, output = _blocked_merge(repo)

    hint = annotate_failure(command, returncode, output)

    assert hint is not None
    assert "lock" in hint.lower()
    assert "Resolve the conflicted files" not in hint
    # Advice still requires proving no live holder before any removal.
    assert "lsof" in hint and "pgrep" in hint


@pytest.mark.platforms("linux")
def test_live_holder_is_never_proposed_for_removal(repo: Path, quiet_probes: None) -> None:
    """A lock held open by a live process must never be called removable."""
    lock = repo / ".git" / "index.lock"
    lock.touch()
    holder = subprocess.Popen(
        [sys.executable, "-c", "import sys, time; f = open(sys.argv[1], 'rb'); time.sleep(60)", str(lock)],
    )
    try:
        deadline = time.time() + 10
        while time.time() < deadline and not gitlock._proc_fd_holders(lock):
            time.sleep(0.05)
        if not gitlock._proc_fd_holders(lock):
            pytest.skip("no /proc fd visibility on this host")

        output = "error: Unable to write index.\nAutomatic merge failed; fix conflicts and then commit the result.\n"
        hint = annotate_failure(f"git -C {repo} merge --no-ff fix/reviewed", 1, output, cwd=str(repo))

        assert hint is not None
        assert "HELD by live PID(s) %d" % holder.pid in hint
        assert "Do NOT remove it" in hint
        assert "Safe to remove" not in hint
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_informational_git_exit_note_does_not_swallow_the_diagnosis(repo: Path, quiet_probes: None) -> None:
    """A blocked merge exits 1, and "git exit 1 is often normal" used to suppress every hint.

    Measured in a live session: the tool returned git's misleading output with no hint at all,
    because the informational note short-circuits annotate_failure.
    """
    from tools.terminal_tool_result import _failure_hint

    command, returncode, output = _blocked_merge(repo)
    note = "Non-zero exit (often normal — e.g. 'git diff' returns 1 when files differ)"

    hint = _failure_hint(command, returncode, output, note, cwd=str(repo))

    assert hint is not None and "index.lock" in hint
    # Other rc=1 git failures keep the informational note and gain nothing.
    assert _failure_hint("git diff a b", 1, "diff --git a/a b/b", note, cwd=str(repo)) is None
