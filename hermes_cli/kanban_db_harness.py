"""EM harness gates for worker blocks and reviewer approvals.

Worker blocks from profiles in ``kanban.block_marker_required_profiles`` are
not blockers unless the reason names a human, an external party, or a card.
A reviewer ``APPROVED`` at a head sha hands the card back for the sanctioned
merge instead of closing it, unless ``metadata.merged`` is set.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Optional

_BLOCKER_MARKER_RE = re.compile(
    r"^\s*BLOCKER:(HUMAN|EXTERNAL|DEP)\s+\S", re.MULTILINE,
)
_APPROVED_RE = re.compile(r"\bAPPROVED\b")
_HEAD_SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b")

_DEFAULT_BLOCK_MARKER_PROFILES = ("software-engineer",)
_BLOCK_REJECTION_LIMIT = 3

BLOCK_REJECTED_NOTE = (
    "EM: not a blocker, continue. A block needs a line "
    "'BLOCKER:HUMAN <decision>' / 'BLOCKER:EXTERNAL <party>' / 'BLOCKER:DEP <card>'. "
    "Waiting on review/tag/CI is not a blocker."
)


def approval_return_comment(sha: str) -> str:
    return (
        f"EM: APPROVED at {sha}. step=merge - run the sanctioned safe-merge gate, "
        "then deploy/accept."
    )


def approval_head_sha(*parts: Optional[str]) -> Optional[str]:
    """Head sha when ``APPROVED`` and a hex sha share the same text; else None."""
    text = "\n".join(part for part in parts if isinstance(part, str) and part)
    if not text or _APPROVED_RE.search(text) is None:
        return None
    match = _HEAD_SHA_RE.search(text)
    return match.group(0) if match else None


def _kb():
    from hermes_cli import kanban_db as kb
    return kb


def _kanban_section() -> dict:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        section = cfg.get("kanban") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def _profile_name(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    canon = (_kb()._canonical_assignee(value) or value).strip().lower()
    return canon or None


def block_marker_required(assignee: Optional[str]) -> bool:
    """True when ``assignee`` is in ``kanban.block_marker_required_profiles``."""
    canon = _profile_name(assignee)
    if canon is None:
        return False
    raw = _kanban_section().get(
        "block_marker_required_profiles", list(_DEFAULT_BLOCK_MARKER_PROFILES),
    )
    if raw is None:
        raw = list(_DEFAULT_BLOCK_MARKER_PROFILES)
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    if not isinstance(raw, (list, tuple)):
        return canon in _DEFAULT_BLOCK_MARKER_PROFILES
    profiles = {name for item in raw if (name := _profile_name(item))}
    return canon in profiles


def approval_returns_to_implementer() -> bool:
    """``kanban.approval_returns_to_implementer`` (default true)."""
    raw = _kanban_section().get("approval_returns_to_implementer", True)
    if raw is None:
        return True
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(raw)


def _prior_block_rejections(conn: sqlite3.Connection, task_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'block_rejected'",
        (task_id,),
    ).fetchone()
    return int(row[0]) if row else 0


def _markerless_block_rejected(
    conn: sqlite3.Connection, task_id: str, *, reason: Optional[str], kind: Optional[str],
    assignee: Optional[str],
) -> bool:
    """True when a worker block must be turned back into ``ready``."""
    if not block_marker_required(assignee):
        return False
    if reason and _BLOCKER_MARKER_RE.search(reason):
        return False
    # An open parent is itself BLOCKER:DEP — do not demand the prose marker.
    if kind == "dependency" and not _kb()._parents_satisfied(conn, task_id):
        return False
    return _prior_block_rejections(conn, task_id) < _BLOCK_REJECTION_LIMIT


def reject_markerless_block(
    conn: sqlite3.Connection, task_id: str, *, reason: Optional[str], kind: Optional[str],
    expected_run_id: Optional[int], assignee: Optional[str],
) -> Optional[bool]:
    """Apply the block-marker gate inside the caller's write txn.

    ``None`` — gate does not apply, caller continues the normal block.
    ``True`` — block rejected, task is ``ready``.
    ``False`` — gate applied but the run-ownership CAS lost.
    """
    if expected_run_id is None:
        return None
    if not _markerless_block_rejected(
        conn, task_id, reason=reason, kind=kind, assignee=assignee,
    ):
        return None
    kb = _kb()
    cur = conn.execute(
        """
        UPDATE tasks
           SET status = 'ready',
               claim_lock = NULL,
               claim_expires = NULL,
               worker_pid = NULL
         WHERE id = ?
           AND status IN ('running', 'ready')
           AND current_run_id = ?
        """,
        (task_id, int(expected_run_id)),
    )
    if cur.rowcount != 1:
        return False
    rejections = _prior_block_rejections(conn, task_id) + 1
    run_id = kb._end_run(conn, task_id, outcome="block_rejected", summary=reason)
    kb._append_event(
        conn, task_id, "block_rejected",
        {"reason": reason, "rejections": rejections}, run_id=run_id,
    )
    kb.add_comment(conn, task_id, "em-harness", BLOCK_REJECTED_NOTE)
    kb._log.info(
        "kanban harness: block_rejected task=%s run=%s rejections=%d",
        task_id, run_id, rejections,
    )
    return True


def return_approved_to_implementer(
    conn: sqlite3.Connection, task_id: str, *, summary: Optional[str], result: Optional[str],
    metadata: Optional[dict], expected_run_id: Optional[int], force: bool = False,
) -> bool:
    """Hand an APPROVED review run back to the implementer. False = fall through."""
    if not approval_returns_to_implementer():
        return False
    meta = metadata if isinstance(metadata, dict) else {}
    if meta.get("merged"):
        return False
    sha = approval_head_sha(summary, result)
    if sha is None:
        return False
    kb = _kb()
    with kb.write_txn(conn):
        task_row = conn.execute(
            "SELECT status, assignee, current_run_id, claim_lock, worker_pid, "
            "worker_started_at FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if task_row is None or task_row["status"] != "running" or not task_row["current_run_id"]:
            return False
        current_run_id = int(task_row["current_run_id"])
        if expected_run_id is not None and current_run_id != int(expected_run_id):
            return False
        if expected_run_id is None and not force and kb._claim_is_live(task_row):
            return False
        claimed = kb._latest_event(conn, task_id, "claimed", current_run_id)
        claimed_payload = kb._json_dict(kb._row_get(claimed, "payload"))
        if claimed_payload.get("source_status") != "review":
            return False
        requested = kb._latest_event(conn, task_id, "review_requested")
        if requested is None:
            return False
        implementer = kb._nonblank_str(
            kb._json_dict(requested["payload"]).get("implementer"),
        )
        if implementer is None:
            return False
        reviewer = kb._canonical_assignee(kb._nonblank_str(task_row["assignee"]))
        new_status = kb._landing_status_after_parents(conn, task_id)
        cur = conn.execute(
            """
            UPDATE tasks
               SET status = ?,
                   assignee = COALESCE(?, assignee),
                   claim_lock = NULL,
                   claim_expires = NULL,
                   worker_pid = NULL,
                   worker_started_at = NULL
             WHERE id = ? AND status = 'running' AND current_run_id = ?
            """,
            (new_status, implementer, task_id, current_run_id),
        )
        if cur.rowcount != 1:
            return False
        run_id = kb._end_run(
            conn, task_id, outcome="review_approved", status=new_status,
            summary=summary if summary is not None else result,
        )
        kb._append_event(
            conn, task_id, "review_approved",
            {
                "implementer": implementer,
                "reviewer": reviewer,
                "step": "merge",
                "head_sha": sha,
            },
            run_id=run_id,
        )
        kb.add_comment(conn, task_id, "em-harness", approval_return_comment(sha))
        kb._log.info(
            "kanban harness: approve->ready task=%s head=%s implementer=%s step=merge",
            task_id, sha, implementer,
        )
    return True
