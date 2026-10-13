"""The incremental re-review diff ran ``git -C <scratch workspace> diff <sha>..HEAD``.
Scratch workspaces live under ~/.hermes/kanban/workspaces/<task>, and ~/.hermes is itself a
git repo, so git resolved the HERMES repo, not the PR repo: 91/95 re-review briefs in the
live logs read "Invalid revision range <sha>..HEAD; fall back to a full review"
(t_e00cedf8 rounds 2-13). Every round was therefore a fresh full review that surfaced new
findings — the real driver of the 13-round loop the round cap (#27) papers over.

The diff must run in a repo that actually contains the reviewed sha, and compare it
against the sha the implementer handed in (review_requested), not whatever HEAD is.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from hermes_cli import kanban_review_brief as krb


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def _init(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@t")
    _git(path, "config", "user.name", "t")


def _commit(path: Path, name: str, text: str) -> str:
    (path / name).write_text(text)
    _git(path, "add", name)
    _git(path, "commit", "-qm", name)
    return _git(path, "rev-parse", "HEAD")


def test_diff_found_in_nested_repo_when_workspace_is_inside_another_repo(tmp_path):
    outer = tmp_path / "hermes"            # ~/.hermes (a git repo)
    _init(outer)
    _commit(outer, "config.yaml", "x")
    ws = outer / "kanban" / "workspaces" / "t_x"   # scratch workspace, not a repo
    pr = ws / "review-5497"                          # the PR checkout the worker made
    _init(pr)
    reviewed = _commit(pr, "a.ts", "one\n")
    _commit(pr, "a.ts", "one\ntwo\n")

    diff, why = krb._git_diff_since(str(ws), reviewed)
    assert diff is not None, why
    assert "+two" in diff


def test_diff_targets_handed_in_head_not_checkout_head(tmp_path):
    pr = tmp_path / "pr"
    _init(pr)
    reviewed = _commit(pr, "a.ts", "one\n")
    handed = _commit(pr, "a.ts", "one\ntwo\n")
    _commit(pr, "a.ts", "one\ntwo\nunrelated-later\n")
    diff, why = krb._git_diff_since(str(pr), reviewed, head=handed)
    assert diff is not None, why
    assert "+two" in diff and "unrelated-later" not in diff


def test_sha_unknown_everywhere_still_reports_unavailable(tmp_path):
    pr = tmp_path / "pr"
    _init(pr)
    _commit(pr, "a.ts", "one\n")
    diff, why = krb._git_diff_since(str(pr), "0" * 40)
    assert diff is None and why
