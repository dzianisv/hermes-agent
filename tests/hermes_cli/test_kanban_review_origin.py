"""request_review DB path: a local-only SHA never enters review.

GitHub 404/422 is a handoff that was never pushed. The refusal is ``push
first`` and the card, its run, and its events stay byte-for-byte as they
were. A 200 moves the card. Any other status, including a failed status
read, fails closed. A handoff with no SHA does not consult the origin API.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_review_origin import (
    API_FAILED,
    MISSING_REPO,
    PUSH_FIRST,
    commit_http_status,
    resolve_review_repo,
    verified_scratch_origin,
)

REPO = "acme/widget"
SHA = "deadbee"


@pytest.fixture
def conn():
    """Tempfile SQLite board, not the process-default kanban path."""
    root = Path(tempfile.mkdtemp(prefix="kanban_review_origin_"))
    db_path = root / "kanban.db"
    kb.init_db(db_path=db_path)
    connection = kbc.connect(db_path=db_path)
    try:
        yield connection
    finally:
        connection.close()


def _seed(conn, *, contract: str = REPO) -> str:
    tid = kb.create_task(
        conn,
        title="ship the handoff",
        assignee="worker",
        completion_contract=contract,
    )
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    assert claimed.status == "running"
    return tid


def _board_state(conn, tid):
    task = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
    runs = conn.execute(
        "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (tid,),
    ).fetchall()
    events = conn.execute(
        "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (tid,),
    ).fetchall()
    return (
        tuple(task),
        [tuple(row) for row in runs],
        [tuple(row) for row in events],
    )


def _patch_status(monkeypatch, status):
    calls = []

    def fake(repo, sha):
        calls.append((repo, sha))
        return status

    monkeypatch.setattr(
        "hermes_cli.kanban_review_origin.commit_http_status", fake,
    )
    return calls


def _review(conn, tid, summary):
    return kb.request_review(conn, tid, summary=summary, with_reason=True)


@pytest.mark.parametrize("status", [404, 422])
def test_local_only_sha_refused_push_first_leaves_state_run_events(
    conn, monkeypatch, status,
):
    tid = _seed(conn)
    before = _board_state(conn, tid)
    calls = _patch_status(monkeypatch, status)

    ok, reason = _review(conn, tid, f"landed {SHA}")

    assert ok is False
    assert reason == PUSH_FIRST
    assert _board_state(conn, tid) == before
    assert calls == [(REPO, SHA)]


def test_published_sha_200_passes(conn, monkeypatch):
    tid = _seed(conn)
    calls = _patch_status(monkeypatch, 200)

    ok, reason = _review(conn, tid, f"landed {SHA}")

    assert (ok, reason) == (True, None)
    task = conn.execute(
        "SELECT status, current_run_id FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    assert task["status"] == "review"
    assert task["current_run_id"] is None
    run = conn.execute(
        "SELECT status, outcome FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    assert run["status"] == "review"
    assert run["outcome"] == "review_requested"
    events = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? AND kind = 'review_requested'",
        (tid,),
    ).fetchall()
    assert len(events) == 1
    assert calls == [(REPO, SHA)]


@pytest.mark.parametrize("status", [None, 500])
def test_origin_api_failure_fails_closed(conn, monkeypatch, status):
    tid = _seed(conn)
    before = _board_state(conn, tid)
    _patch_status(monkeypatch, status)

    ok, reason = _review(conn, tid, f"landed {SHA}")

    assert ok is False
    assert reason == API_FAILED
    assert _board_state(conn, tid) == before


def test_handoff_with_no_sha_passes_without_origin_check(conn, monkeypatch):
    tid = _seed(conn, contract="local-only")
    calls = _patch_status(monkeypatch, 404)

    ok, reason = _review(conn, tid, "implementation complete, no commit named")

    assert (ok, reason) == (True, None)
    assert calls == []
    status = conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (tid,),
    ).fetchone()["status"]
    assert status == "review"


class _Gh:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.mark.parametrize(
    "rc,stdout,stderr,expected",
    [
        (0, "HTTP/2.0 200 OK\r\n", "", 200),
        (1, "HTTP/2.0 200 OK\r\n", "", None),
        (1, "", "gh: Not Found (HTTP 404)\n", 404),
        (1, "", "gh: Unprocessable Entity (HTTP 422)\n", 422),
        (0, "commit exists, status 200\n", "(HTTP 200)\n", None),
    ],
)
def test_commit_http_status_requires_parsed_200_on_zero_exit(
    monkeypatch, rc, stdout, stderr, expected,
):
    """200 only when gh exits 0 and the response line parsed; 404/422 survive a nonzero exit."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return _Gh(rc, stdout, stderr)

    monkeypatch.setattr(
        "hermes_cli.kanban_review_origin.subprocess.run", fake_run,
    )

    assert commit_http_status(REPO, SHA) == expected
    assert calls == [[
        "gh", "api", f"repos/{REPO}/commits/{SHA}", "--hostname", "github.com", "-i",
    ]]


def test_metadata_and_row_result_shas_are_both_checked(conn, monkeypatch):
    """A SHA only in metadata and a SHA only on the stored result are both origin-checked."""
    meta_sha = "abc1234"
    row_sha = "cafebabe"
    tid = _seed(conn)
    conn.execute(
        "UPDATE tasks SET result = ? WHERE id = ?", (f"built {row_sha}", tid),
    )
    conn.commit()
    before = _board_state(conn, tid)
    calls = []

    def fake(repo, sha):
        calls.append((repo, sha))
        return 404 if sha == row_sha else 200

    monkeypatch.setattr(
        "hermes_cli.kanban_review_origin.commit_http_status", fake,
    )

    ok, reason = kb.request_review(
        conn, tid, summary="handoff notes, no sha here",
        metadata={"commit": meta_sha}, with_reason=True,
    )

    assert ok is False
    assert reason == PUSH_FIRST
    assert calls == [(REPO, meta_sha), (REPO, row_sha)]
    assert _board_state(conn, tid) == before


def test_sha_without_resolvable_repo_fails_closed(conn, monkeypatch):
    tid = _seed(conn, contract="local-only")
    before = _board_state(conn, tid)
    calls = _patch_status(monkeypatch, 200)

    ok, reason = _review(conn, tid, f"landed {SHA}")

    assert ok is False
    assert reason == MISSING_REPO
    assert calls == []
    assert _board_state(conn, tid) == before


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True, capture_output=True, text=True,
    )


def test_scratch_resolver_uses_child_origin_not_parent_checkout(
    conn, monkeypatch, tmp_path,
):
    parent = tmp_path / "parent-checkout"
    parent.mkdir()
    _git(parent, "init")
    _git(parent, "remote", "add", "origin", "https://github.com/other/parent.git")
    bare = parent / "bare-scratch"
    bare.mkdir()
    scratch = parent / "scratch"
    child = scratch / "repo"
    child.mkdir(parents=True)
    _git(child, "init")
    _git(child, "remote", "add", "origin", "git@github.com:acme/widget.git")

    assert verified_scratch_origin(str(bare)) is None
    assert verified_scratch_origin(str(scratch)) == REPO
    assert resolve_review_repo("local-only", "scratch", str(scratch)) == REPO
    assert resolve_review_repo("local-only", "dir", str(scratch)) is None

    tid = _seed(conn, contract="local-only")
    conn.execute(
        "UPDATE tasks SET workspace_path = ? WHERE id = ?", (str(scratch), tid),
    )
    conn.commit()
    calls = _patch_status(monkeypatch, 200)

    ok, reason = _review(conn, tid, f"landed {SHA}")

    assert (ok, reason) == (True, None)
    assert calls == [(REPO, SHA)]


@pytest.mark.parametrize(
    "rc,stdout,stderr,expected",
    [
        (0, "HTTP/2.0 200 OK\r\n", "", 200),
        (1, "HTTP/2.0 200 OK\r\n", "", None),
        (1, "HTTP/2.0 404 Not Found\r\n", "", 404),
        (1, "HTTP/2.0 422 Unprocessable Entity\r\n", "", 422),
        (1, "unreadable body\n", "gh: could not read response\n", None),
    ],
)
def test_commit_http_status_completed_process(
    monkeypatch, rc, stdout, stderr, expected,
):
    """subprocess.run returns CompletedProcess: 200 only on rc 0; 404/422 on rc 1; unreadable is None."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(
        "hermes_cli.kanban_review_origin.subprocess.run", fake_run,
    )

    assert commit_http_status(REPO, SHA) == expected
    assert calls == [[
        "gh", "api", f"repos/{REPO}/commits/{SHA}", "--hostname", "github.com", "-i",
    ]]


def test_summary_and_metadata_second_sha_404_refuses(conn, monkeypatch):
    """Every SHA in summary and metadata is checked; a later 404 refuses the handoff."""
    first = "abc1234"
    second = "cafebabe"
    tid = _seed(conn)
    before = _board_state(conn, tid)
    calls = []

    def fake(repo, sha):
        calls.append((repo, sha))
        return 404 if sha == second else 200

    monkeypatch.setattr(
        "hermes_cli.kanban_review_origin.commit_http_status", fake,
    )

    ok, reason = kb.request_review(
        conn, tid, summary=f"landed {first}",
        metadata={"commit": second}, with_reason=True,
    )

    assert ok is False
    assert reason == PUSH_FIRST
    assert calls == [(REPO, first), (REPO, second)]
    assert _board_state(conn, tid) == before


def test_verified_scratch_origin_workspace_repo_github_origin(tmp_path):
    """Scratch parent workspace/repo with a GitHub origin resolves to owner/repo."""
    workspace = tmp_path / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "remote", "add", "origin", "https://github.com/acme/widget.git")

    assert verified_scratch_origin(str(workspace)) == REPO
