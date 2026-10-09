"""Wait-on handles and no-progress detection for kanban cards.

R1 — a worker parks a card on an external handle instead of being re-claimed
every tick to re-discover the same red CI:

    gh-run:<owner/repo>:<run_id>        a GitHub Actions run
    pr:<owner/repo>#<number>[@<sha>]    a PR (head sha + check conclusions)
    card:<task_id>                      another kanban card

``wait_on_task`` records the handle and its current fingerprint and parks the
card in ``scheduled`` (never dispatched). ``poll_wait_on`` runs on every
dispatcher tick, re-observes each handle at most once per
``kanban.wait_on_poll_seconds`` (default 120s) and at most
``kanban.wait_on_max_polls`` handles per tick, and promotes the card only when
the fingerprint differs from the baseline.

R2 — a run that ends with no new commit, no status change and no new passing
check is "no progress". Two such runs in a row reassign the card to the
architect for re-spec instead of claiming it a third time.

R3 — ``kanban.review_round_limit`` / ``active_seconds_limit`` (the run caps)
stay as the backstop for loops these signals cannot see.

Observation is injected (``status_fn``/``head_fn``) so tests call the real
functions with a real temp DB; the defaults shell out to ``gh`` / ``git``.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Optional

import sqlite3

from hermes_cli import kanban_db as _kb

StatusFn = Callable[["Handle", sqlite3.Connection], Optional[str]]
HeadFn = Callable[[Optional[str]], Optional[str]]

DEFAULT_POLL_SECONDS = 120
DEFAULT_MAX_POLLS = 10
DEFAULT_NO_PROGRESS_LIMIT = 2

# Outcomes that are themselves a status change on the card (progress).
PROGRESS_OUTCOMES = frozenset({"completed", "review_requested", "approved", "changes_requested"})


@dataclass(frozen=True)
class Handle:
    kind: str           # gh-run | pr | card
    repo: Optional[str]
    ident: str          # run id, PR number, or task id
    sha: Optional[str] = None

    def __str__(self) -> str:
        if self.kind == "card":
            return f"card:{self.ident}"
        if self.kind == "gh-run":
            return f"gh-run:{self.repo}:{self.ident}"
        return f"pr:{self.repo}#{self.ident}" + (f"@{self.sha}" if self.sha else "")


_RX_RUN = re.compile(r"^gh-run:([\w.-]+/[\w.-]+):(\d+)$")
_RX_PR = re.compile(r"^pr:([\w.-]+/[\w.-]+)#(\d+)(?:@([0-9a-fA-F]{7,40}))?$")
_RX_CARD = re.compile(r"^card:(t_[0-9a-zA-Z]+)$")


def parse_handle(text: str) -> Handle:
    s = (text or "").strip()
    if m := _RX_RUN.match(s):
        return Handle("gh-run", m.group(1), m.group(2))
    if m := _RX_PR.match(s):
        return Handle("pr", m.group(1), m.group(2), (m.group(3) or "").lower() or None)
    if m := _RX_CARD.match(s):
        return Handle("card", None, m.group(1))
    raise ValueError(
        f"bad wait_on handle {text!r}; use gh-run:<owner/repo>:<run_id>, "
        "pr:<owner/repo>#<n>[@<sha>] or card:<task_id>")


def _cfg_int(key: str, default: int) -> int:
    try:
        return max(0, int(_kb._kanban_cfg().get(key, default)))
    except (TypeError, ValueError):
        return default


# --- observation -----------------------------------------------------------

def _gh_json(args: list[str]) -> Optional[dict]:
    try:
        out = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        data = json.loads(out.stdout)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def fingerprint_run(data: dict) -> str:
    return "|".join(str(data.get(k) or "") for k in ("status", "conclusion", "attempt", "headSha"))


def fingerprint_pr(data: dict) -> str:
    checks = []
    for c in data.get("statusCheckRollup") or []:
        name = c.get("name") or c.get("context") or ""
        state = c.get("conclusion") or c.get("state") or c.get("status") or ""
        checks.append(f"{name}={state}".lower())
    return "|".join([str(data.get("headRefOid") or "").lower(), str(data.get("state") or ""),
                     ",".join(sorted(checks))])


def default_status(handle: Handle, conn: sqlite3.Connection) -> Optional[str]:
    """Current fingerprint of *handle*; ``None`` = could not observe (stay parked)."""
    if handle.kind == "card":
        row = conn.execute("SELECT status, completed_at FROM tasks WHERE id = ?",
                           (handle.ident,)).fetchone()
        return None if row is None else f"{row['status']}|{row['completed_at'] or ''}"
    if handle.kind == "gh-run":
        data = _gh_json(["run", "view", handle.ident, "-R", handle.repo,
                         "--json", "status,conclusion,attempt,headSha"])
        return None if data is None else fingerprint_run(data)
    data = _gh_json(["pr", "view", handle.ident, "-R", handle.repo,
                     "--json", "headRefOid,state,statusCheckRollup"])
    return None if data is None else fingerprint_pr(data)


def is_passing(fp: Optional[str]) -> bool:
    """A fingerprint that reads as green (used for the R2 'new passing check')."""
    if not fp:
        return False
    low = fp.lower()
    if re.search(r"failure|cancelled|timed_out|action_required|=error|=pending|in_progress|queued", low):
        return False
    return "success" in low or low.startswith("done|")


# --- R1: park and poll -----------------------------------------------------

def wait_on_task(
    conn: sqlite3.Connection, task_id: str, handle: str, *, reason: Optional[str] = None,
    expected_run_id: Optional[int] = None, status_fn: StatusFn = default_status,
) -> bool:
    """Park *task_id* in ``scheduled`` until *handle* changes. The baseline is
    observed now; an unobservable handle stores ``None`` and the first real
    observation counts as the change."""
    h = parse_handle(handle)
    baseline = status_fn(h, conn)
    note = reason or f"waiting on {h}"
    if not _kb.schedule_task(conn, task_id, reason=note, expected_run_id=expected_run_id):
        return False
    with _kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET wait_on = ?, wait_on_fingerprint = ?, wait_on_checked_at = ? WHERE id = ?",
            (str(h), baseline, int(time.time()), task_id))
        _kb._append_event(conn, task_id, "wait_on_set", {"handle": str(h), "fingerprint": baseline})
    return True


def poll_wait_on(
    conn: sqlite3.Connection, *, status_fn: StatusFn = default_status, now: Optional[int] = None,
    poll_seconds: Optional[int] = None, max_polls: Optional[int] = None,
) -> list[str]:
    """Promote every parked card whose handle changed. Rate limited per card
    (``poll_seconds``) and per tick (``max_polls``). Returns promoted ids."""
    now = int(time.time()) if now is None else int(now)
    poll_seconds = _cfg_int("wait_on_poll_seconds", DEFAULT_POLL_SECONDS) if poll_seconds is None else poll_seconds
    max_polls = _cfg_int("wait_on_max_polls", DEFAULT_MAX_POLLS) if max_polls is None else max_polls
    rows = conn.execute(
        "SELECT id, wait_on, wait_on_fingerprint FROM tasks WHERE status = 'scheduled' "
        "AND wait_on IS NOT NULL AND COALESCE(wait_on_checked_at, 0) <= ? "
        "ORDER BY COALESCE(wait_on_checked_at, 0) LIMIT ?",
        (now - poll_seconds, max_polls),
    ).fetchall()
    promoted: list[str] = []
    for row in rows:
        try:
            h = parse_handle(row["wait_on"])
            fp = status_fn(h, conn)
        except Exception:
            fp = None
        if fp is None or fp == row["wait_on_fingerprint"]:
            with _kb.write_txn(conn):
                conn.execute("UPDATE tasks SET wait_on_checked_at = ? WHERE id = ?", (now, row["id"]))
            continue
        with _kb.write_txn(conn):
            landed = _kb._resume_parked_task_locked(
                conn, row["id"], statuses=("scheduled",), event_kind="wait_on_changed", now=now,
                note="invariant recovery on wait_on change",
                extra_payload={"handle": row["wait_on"], "from": row["wait_on_fingerprint"],
                               "to": fp, "passing": is_passing(fp)},
            )
        if landed is not None:
            promoted.append(row["id"])
    return promoted


# --- R2: no-progress -------------------------------------------------------

def default_head(workspace: Optional[str]) -> Optional[str]:
    if not workspace:
        return None
    try:
        out = subprocess.run(["git", "-C", workspace, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def record_run_head(conn: sqlite3.Connection, run_id: Optional[int], head: Optional[str]) -> None:
    """Stamp the workspace HEAD at spawn onto the run (``head_start``)."""
    if not run_id or not head:
        return
    with _kb.write_txn(conn):
        row = conn.execute("SELECT metadata FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        meta = _kb._json_dict(row["metadata"] if row else None)
        meta["head_start"] = head
        conn.execute("UPDATE task_runs SET metadata = ? WHERE id = ?", (json.dumps(meta), run_id))


def no_progress_streak(conn: sqlite3.Connection, task_id: str, current_head: Optional[str]) -> int:
    """Count consecutive trailing ended runs (since the last re-spec) that made
    no progress: outcome is not a forward status change (block/park/crash do
    not count — that is the red-CI hold loop), workspace HEAD unchanged across
    the run, and no wait_on handle turned passing since it started."""
    last_respec = conn.execute(
        "SELECT COALESCE(MAX(created_at), 0) FROM task_events WHERE task_id = ? AND kind = 'no_progress_respec'",
        (task_id,)).fetchone()[0]
    runs = conn.execute(
        "SELECT id, outcome, metadata, started_at, ended_at FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL AND started_at > ? ORDER BY started_at DESC, id DESC",
        (task_id, last_respec)).fetchall()
    passing_at = [int(r["created_at"]) for r in conn.execute(
        "SELECT created_at, payload FROM task_events WHERE task_id = ? AND kind = 'wait_on_changed'",
        (task_id,)) if _kb._json_dict(r["payload"]).get("passing")]
    streak = 0
    head_after = current_head
    for r in runs:
        head_start = _kb._json_dict(r["metadata"]).get("head_start")
        progressed = (
            (r["outcome"] or "") in PROGRESS_OUTCOMES
            or (head_start and head_after and head_start != head_after)
            # A handle that turned green during or after this run is new
            # evidence: runs before it don't count toward the streak.
            or any(t >= r["started_at"] for t in passing_at)
        )
        if progressed:
            break
        streak += 1
        if head_start:
            head_after = head_start
    return streak


def respec_assignee() -> Optional[str]:
    cfg = _kb._kanban_cfg()
    if cfg.get("respec_assignee"):
        return str(cfg["respec_assignee"])
    dp = cfg.get("design_phase") or {}
    return dp.get("architect") or next(iter(dp.get("architects") or []), None)


def no_progress_guard(
    conn: sqlite3.Connection, task_id: str, assignee: str, workspace: Optional[str], *,
    head_fn: HeadFn = default_head, architect: Optional[str] = None, limit: Optional[int] = None,
) -> Optional[str]:
    """Before a claim: after ``limit`` no-progress runs in a row, hand the card
    to the architect for re-spec and return a reason (caller skips the claim)."""
    limit = _cfg_int("no_progress_limit", DEFAULT_NO_PROGRESS_LIMIT) if limit is None else limit
    if not limit:
        return None
    architect = architect or respec_assignee()
    if not architect or assignee == architect:
        return None
    streak = no_progress_streak(conn, task_id, head_fn(workspace))
    if streak < limit:
        return None
    note = (f"BLOCKER:DEP architect — [no-progress] {streak} runs in a row ended with no new commit, "
            "no status change and no new passing check. Re-spec THIS card (what is actually blocking, "
            "what wait_on handle to park on, what PROOF ends it), then reassign it to the implementer.")
    with _kb.write_txn(conn):
        _kb._append_event(conn, task_id, "no_progress_respec",
                          {"streak": streak, "from": assignee, "to": architect})
    _kb.assign_task(conn, task_id, architect)
    _kb.add_comment(conn, task_id, author="kanban-dispatcher", body=note)
    return "no_progress_respec"
