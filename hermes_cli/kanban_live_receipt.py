"""Live-validation receipt gate for product-outcome cards (Notion harness #13).

"Review passed" is not "works live". When ``kanban.live_receipt.workflows`` is
configured, cards marked as product outcomes (a configured board/project, or a
body line ``PROOF: <workflow>``) can only reach ``done`` with a receipt: the
URL of a GitHub Actions run of a named validation workflow. The receipt is
re-fetched independently via ``gh`` (never trusted from the worker) and must be
``conclusion=success``, run on the merged commit or a descendant of it, and be
created after the merge. Off unless configured.

Config (``config.yaml``)::

    kanban:
      live_receipt:
        workflows: [integration-telegram.yml]   # required to enable
        boards: [production]                    # optional
        projects: [p_abc]                       # optional

``GH_RUNNER`` is the injectable seam: ``(args: list[str]) -> parsed JSON``.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from typing import Any, Callable, Optional

RUN_URL = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/actions/runs/([0-9]+)")
_PR_URL = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")
_PROOF = re.compile(r"^\s*PROOF:\s*(.+?)\s*$", re.MULTILINE)


class LiveReceiptError(ValueError):
    """Completion refused: missing/skipped/stale/wrong-target live receipt.
    Raised before any write; the card keeps its status and owner."""

    def __init__(self, task_id: str, reason: str):
        self.task_id, self.reason = task_id, reason
        super().__init__(f"completion blocked: {task_id} needs a live-validation receipt: {reason}")


def _default_gh(args: list[str]) -> Any:
    out = subprocess.run(["gh", *args], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=30, check=True)
    return json.loads(out.stdout)


GH_RUNNER: Callable[[list[str]], Any] = _default_gh


def _config() -> dict:
    try:
        from hermes_cli.config import load_config
        cfg = ((load_config() or {}).get("kanban") or {}).get("live_receipt") or {}
    except Exception:
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _names(value) -> set[str]:
    if isinstance(value, str):
        value = [value]
    return {str(v).strip() for v in (value or []) if str(v).strip()}


def required_workflows(task, board: Optional[str], cfg: Optional[dict] = None) -> set[str]:
    """Workflows whose receipt the card needs; empty set = gate off for it."""
    cfg = _config() if cfg is None else cfg
    workflows = _names(cfg.get("workflows"))
    if not workflows:
        return set()
    proof = {w for line in _PROOF.findall(task.body or "") for w in workflows if w in line}
    if proof:
        return proof
    if (board and board in _names(cfg.get("boards"))) or (
            task.project_id and task.project_id in _names(cfg.get("projects"))):
        return workflows
    if _PROOF.search(task.body or ""):
        return workflows  # PROOF: line naming no configured workflow still marks the outcome
    return set()


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _find(pattern, *texts) -> Optional[re.Match]:
    for text in texts:
        if isinstance(text, str):
            m = pattern.search(text)
            if m:
                return m
    return None


def verify(task, *, metadata: Optional[dict], summary: Optional[str], result: Optional[str],
           workflows: set[str], gh: Optional[Callable[[list[str]], Any]] = None) -> dict:
    """Return the verified receipt dict, or raise ``LiveReceiptError``."""
    gh = gh or GH_RUNNER
    md = metadata if isinstance(metadata, dict) else {}
    tid = task.id
    wanted = ", ".join(sorted(workflows))
    rm = _find(RUN_URL, md.get("live_receipt"), summary, result)
    if not rm:
        raise LiveReceiptError(tid, f"no GitHub Actions run URL of {wanted} supplied. Run the live "
                               "validation after merge and pass metadata.live_receipt=<run URL>.")
    repo, run_id = rm[1], rm[2]
    pr = _find(_PR_URL, md.get("published_pr"), task.completion_contract, summary, result)
    try:
        if pr:
            if pr[1] != repo:
                raise LiveReceiptError(tid, f"receipt repo {repo} differs from PR repo {pr[1]} (wrong target).")
            info = gh(["pr", "view", pr[2], "-R", pr[1], "--json", "mergeCommit,mergedAt,state"])
            merged_sha = ((info.get("mergeCommit") or {}).get("oid") or "")
            merged_at = info.get("mergedAt")
            if info.get("state") != "MERGED" or not merged_sha or not merged_at:
                raise LiveReceiptError(tid, f"{pr[0]} is not merged; a live receipt must follow the merge.")
        else:
            merged_sha = str(md.get("merged_sha") or "").strip()
            if not re.fullmatch(r"[0-9a-f]{7,40}", merged_sha):
                raise LiveReceiptError(tid, "cannot identify the merged commit; pass metadata.published_pr "
                                       "(merged PR URL) or metadata.merged_sha.")
            commit = gh(["api", f"repos/{repo}/commits/{merged_sha}"])
            merged_sha = commit["sha"]
            merged_at = commit["commit"]["committer"]["date"]
        run = gh(["run", "view", run_id, "-R", repo, "--json",
                  "conclusion,status,headSha,workflowName,path,createdAt,url,event"])
        path = str(run.get("path") or "")
        wf_file = path.rsplit("/", 1)[-1].split("@", 1)[0]
        if not ({wf_file, run.get("workflowName")} & workflows):
            raise LiveReceiptError(tid, f"run {run_id} is workflow {wf_file or run.get('workflowName')!r}, "
                                   f"not {wanted} (wrong target).")
        if run.get("status") != "completed" or run.get("conclusion") != "success":
            raise LiveReceiptError(tid, f"run {run_id} is status={run.get('status')} "
                                   f"conclusion={run.get('conclusion')} (need completed/success; skipped "
                                   "or cancelled runs are not proof).")
        head = run.get("headSha") or ""
        if head != merged_sha:
            cmp = gh(["api", f"repos/{repo}/compare/{merged_sha}...{head}"])
            if cmp.get("status") not in ("identical", "ahead"):
                raise LiveReceiptError(tid, f"run {run_id} tested {head[:12]}, which does not contain merged "
                                       f"commit {merged_sha[:12]} (wrong target).")
        if _ts(run["createdAt"]) < _ts(merged_at):
            raise LiveReceiptError(tid, f"run {run_id} started {run['createdAt']}, before the merge at "
                                   f"{merged_at} (stale). Re-run {wanted} after merge.")
    except LiveReceiptError:
        raise
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        raise LiveReceiptError(tid, f"could not fetch receipt evidence via gh ({type(exc).__name__}); "
                               "check gh auth and retry.") from None
    return {"run_url": rm[0], "workflow": wf_file or run.get("workflowName"), "head_sha": head,
            "merged_sha": merged_sha, "merged_at": merged_at, "run_created_at": run["createdAt"]}


def gate(conn, task_id: str, *, metadata, summary, result, board: Optional[str] = None,
         gh: Optional[Callable[[list[str]], Any]] = None) -> Optional[dict]:
    """Pre-write gate for ``complete_task``. Records an audit event either way."""
    from hermes_cli import kanban_db as kb
    task = kb.get_task(conn, task_id)
    if task is None:
        return None
    if board is None:
        try:
            board = kb.get_current_board()
        except Exception:
            board = None
    workflows = required_workflows(task, board)
    if not workflows:
        return None
    try:
        receipt = verify(task, metadata=metadata, summary=summary, result=result, workflows=workflows, gh=gh)
    except LiveReceiptError as err:
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, "completion_blocked_live_receipt",
                             {"reason": err.reason, "workflows": sorted(workflows)})
        raise
    with kb.write_txn(conn):
        kb._append_event(conn, task_id, "live_receipt_verified", receipt)
    return receipt
