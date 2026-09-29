"""Fail closed unless every handoff commit SHA already exists on GitHub.

``request_review`` is the only transition into ``review`` (CLI, tool, dashboard).
A SHA that GitHub has not accepted is a local-only handoff: 404/422 reject with
``push first`` and leave the card untouched. Repo identity comes from the
completion contract, or from the scratch workspace's own ``origin`` when the
contract does not name a repo — never from cwd, prose, or a parent checkout.
Network runs before the write transaction; the caller rechecks the snapshot.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Optional

from hermes_cli.kanban_pr_acceptance import _PR, _REPO

PUSH_FIRST = "push first"
MISSING_REPO = "cannot verify commit origin"
API_FAILED = "commit origin check failed"
STALE_SNAPSHOT = "task changed before review handoff"

_SHA = re.compile(r"(?<![0-9A-Za-z_])([0-9a-fA-F]{7,64})(?![0-9A-Za-z_/\\])")
_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_GH_REMOTE = re.compile(
    r"^(?:https://(?:[^@/\s]+@)?github\.com/|ssh://(?:[^@/\s]+@)?github\.com/"
    r"|git@github\.com:)([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
_ORIGIN_REASONS = frozenset({PUSH_FIRST, MISSING_REPO, API_FAILED})


def is_origin_refusal(reason: Optional[str]) -> bool:
    return reason in _ORIGIN_REASONS


def extract_commit_shas(*parts: Any) -> list[str]:
    """Standalone git SHAs in handoff prose/metadata/result. Not path segments,
    task ids, or UUID pieces."""
    found: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        scrubbed = _UUID.sub(" ", text)
        for match in _SHA.finditer(scrubbed):
            token = match.group(1)
            if not re.search(r"[a-fA-F]", token):
                continue
            key = token.lower()
            if key not in seen:
                seen.add(key)
                found.append(key)

    def walk(value: Any) -> None:
        if isinstance(value, str):
            add(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    for part in parts:
        walk(part)
    return found


def repo_from_contract(contract: Optional[str]) -> Optional[str]:
    """OWNER/REPO or the repo in an exact PR URL. ``None`` when unnamed."""
    if not contract or contract == "local-only":
        return None
    pr = _PR.fullmatch(contract)
    if pr:
        return pr.group(1)
    if _REPO.fullmatch(contract):
        return contract
    return None


def contract_names_repo(contract: Optional[str]) -> bool:
    return bool(contract) and contract != "local-only"


def verified_scratch_origin(workspace_path: Optional[str]) -> Optional[str]:
    """GitHub ``owner/repo`` of this scratch workspace's task remote.

    The workspace may be its own git toplevel, or the parent of exactly one
    immediate child git worktree (``repo/``). That single child must expose
    one GitHub ``origin``. Zero child worktrees, several, or a non-GitHub
    remote are rejected. A parent checkout the path merely sits inside is
    not the task's remote.
    """
    if not workspace_path:
        return None
    path = Path(workspace_path)
    if not path.is_absolute():
        return None
    try:
        resolved = path.resolve()
    except OSError:
        return None
    if not resolved.is_dir():
        return None

    def github_origin(cwd: Path) -> Optional[str]:
        remote = _git(cwd, "remote", "get-url", "origin")
        if not remote or any(ch.isspace() for ch in remote):
            return None
        match = _GH_REMOTE.fullmatch(remote)
        if not match:
            return None
        return f"{match.group(1)}/{match.group(2)}"

    def is_own_toplevel(cwd: Path) -> bool:
        toplevel = _git(cwd, "rev-parse", "--show-toplevel")
        if not toplevel:
            return False
        try:
            return Path(toplevel).resolve() == cwd.resolve()
        except OSError:
            return False

    if is_own_toplevel(resolved):
        return github_origin(resolved)

    worktrees = 0
    origin: Optional[str] = None
    try:
        children = list(resolved.iterdir())
    except OSError:
        return None
    for child in children:
        if child.is_symlink() or not child.is_dir():
            continue
        try:
            child_resolved = child.resolve()
        except OSError:
            return None
        if not is_own_toplevel(child_resolved):
            continue
        worktrees += 1
        origin = github_origin(child_resolved)
    if worktrees != 1:
        return None
    return origin


def resolve_review_repo(
    contract: Optional[str], workspace_kind: Optional[str], workspace_path: Optional[str],
) -> Optional[str]:
    if contract_names_repo(contract):
        return repo_from_contract(contract)
    if workspace_kind == "scratch":
        return verified_scratch_origin(workspace_path)
    return None


def commit_http_status(repo: str, sha: str) -> Optional[int]:
    """HTTP status of ``gh api repos/<owner>/<repo>/commits/<sha>``.

    200 is accepted only when ``gh`` exits 0 and the response status line
    parsed. 404/422 are kept when ``gh`` exits nonzero — those are GitHub's
    missing-object statuses, including the ``(HTTP 404)`` annotation. Any
    other outcome, including an unparseable status or a 200 on a failed
    exit, is ``None``. Stderr is parsed for that annotation only and is
    never logged — it can carry tokens and host details.
    """
    if not _REPO.fullmatch(repo) or not re.fullmatch(r"[0-9a-f]{7,64}", sha):
        return None
    try:
        proc = subprocess.run(
            ["gh", "api", f"repos/{repo}/commits/{sha}", "--hostname", "github.com", "-i"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    response = _response_status(proc.stdout or "")
    if proc.returncode != 0 and response == 200:
        return None
    if proc.returncode != 0:
        if response in (404, 422):
            return response
        annotated = _annotated_status(proc.stderr or "")
        if annotated in (404, 422):
            return annotated
        return None
    return response


def _response_status(stdout: str) -> Optional[int]:
    """Status from the HTTP response line. Not gh's stderr annotation."""
    match = re.search(r"HTTP/\d+(?:\.\d+)?\s+(\d{3})\b", stdout)
    return _int_or_none(match.group(1) if match else None)


def _annotated_status(stderr: str) -> Optional[int]:
    match = re.search(r"\(HTTP\s+(\d{3})\)", stderr)
    return _int_or_none(match.group(1) if match else None)


def _int_or_none(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def classify_statuses(statuses: list[Optional[int]]) -> Optional[str]:
    """``None`` when every SHA is on the remote. Otherwise a refusal reason."""
    if any(code not in (404, 422) and code != 200 for code in statuses):
        return API_FAILED
    if any(code in (404, 422) for code in statuses):
        return PUSH_FIRST
    if statuses and all(code == 200 for code in statuses):
        return None
    return API_FAILED


def verify_shas(repo: Optional[str], shas: list[str]) -> Optional[str]:
    if not shas:
        return None
    if not repo:
        return MISSING_REPO
    return classify_statuses([commit_http_status(repo, sha) for sha in shas])


def _git(cwd: Path, *args: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip() or None
