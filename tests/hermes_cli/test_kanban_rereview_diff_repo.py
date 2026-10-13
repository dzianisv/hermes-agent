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



import json

import pytest

from tests.hermes_cli._kanban_modules import KanbanModules

kb = KanbanModules()


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kb, "_active_pr_guard_applies", lambda _url: (True, None))
    kb.init_db()
    c = kb.connect()
    yield c
    c.close()


def _review_round(conn, tid, head):
    claimed = kb.claim_task(conn, tid)
    kb.add_comment(conn, tid, author="impl", body="Opened https://github.com/e/r/pull/1 for review.")
    assert kb.request_review(conn, tid, summary="PR ready", reviewer="reviewer",
                             metadata={"head_sha": head}, expected_run_id=claimed.current_run_id)


def test_request_review_records_handed_in_head(conn):
    tid = kb.create_task(conn, title="t", assignee="impl")
    _review_round(conn, tid, "c" * 40)
    payload = json.loads(kb._latest_event(conn, tid, "review_requested")["payload"])
    assert payload.get("head_sha") == "c" * 40


def test_rereview_brief_passes_handed_in_head_to_diff(conn, tmp_path):
    """Round 2+: diff reviewed_sha..<head the implementer handed in>, not checkout HEAD."""
    tid = kb.create_task(conn, title="t", assignee="impl")
    _review_round(conn, tid, "a" * 40)
    kb.claim_review_task(conn, tid)
    ok, why = kb.request_changes(conn, tid, reason="fix it", metadata={"reviewed_sha": "a" * 40})
    assert ok, why
    _review_round(conn, tid, "b" * 40)
    seen = {}

    def fake_diff(ws, sha, *, head=None):
        seen.update(sha=sha, head=head)
        return "+x", ""

    out = krb.build_review_brief(conn, kb.get_task(conn, tid), fetch=lambda _l: (None, "x"),
                                 workspace=str(tmp_path), diff_fn=fake_diff)
    assert seen == {"sha": "a" * 40, "head": "b" * 40}, out
    assert "b" * 12 in out
