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
architect for re-spec instead of claiming it a third time. A run whose failure is
environmental (the agent never started, or the host could not place/feed it) is
NEUTRAL: it neither counts nor resets the streak, and repeated identical
environmental failures are recorded once as a host incident instead of a re-spec.

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

# Evidence that a run failed on the HOST, not on the card: the launcher could not
# exec the pinned interpreter, the install has no committed dependency environment,
# the card's workspace is not readable by the worker, or the harness itself could
# not be imported. None of these say anything about the card, so they must neither
# count as "the implementer made no progress" nor be answered with a re-spec (live:
# 6 environmental crashes re-spec'd a healthy card at 12:10 on 2026-11-09).
#
# The markers are the launcher's / harness's own lines, never a bare kernel phrase
# and never a third-party import. ``kanban_worker_exit.LAUNCHER_EXEC_FAILURE_MARKER``
# records the regression that taught this: a broad match absorbed a REAL worker
# crash (a worker that ran and then hit its own ``FileNotFoundError``) and retried
# it forever. A hand-maintained list is only safe while it stays this narrow; the
# generation-aware half of the rule is :func:`run_started_agent`, which needs no
# list at all and covers a new host failure mode.
_ENV_FAULT_MARKERS = (
    "Failed to find executable",                # launcher: pinned interpreter absent
    "executable not found",
    "no dependency environment is committed",   # needs ``hermes pm repair``
    "No module named hermes_cli",               # cwd-shadowed harness import
    "No module named 'hermes_cli",
)
# The claim/spawn path prefixes every workspace-resolution failure with ``workspace:``.
_ENV_FAULT_PREFIXES = ("workspace:",)
_ENV_FAULT_RE = re.compile(
    r"\[Errno 13\] Permission denied|not inside a git repo|not a git (?:repository|repo|work ?tree)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Handle:
    kind: str           # gh-run | pr | card
    repo: Optional[str]
    ident: str          # run id, PR number, or task id
    sha: Optional[str] = None

    def __str__(self) -> str:
        if self.kind == "card":
            return f"{self.kind}:{self.ident}"
        if self.kind == "gh-run":
            return f"gh-run:{self.repo}:{self.ident}"
        return f"pr:{self.repo}#{self.ident}" + (f"@{self.sha}" if self.sha else "")


_RX_RUN = re.compile(r"^gh-run:([\w.-]+/[\w.-]+):(\d+)$")
_RX_PR = re.compile(r"^pr:([\w.-]+/[\w.-]+)#(\d+)(?:@([0-9a-fA-F]{7,40}))?$")
_RX_CARD = re.compile(r"^(card):(t_[0-9a-zA-Z]+)$")


def parse_handle(text: str) -> Handle:
    s = (text or "").strip()
    if m := _RX_RUN.match(s):
        return Handle("gh-run", m.group(1), m.group(2))
    if m := _RX_PR.match(s):
        return Handle("pr", m.group(1), m.group(2), (m.group(3) or "").lower() or None)
    if m := _RX_CARD.match(s):
        return Handle(m.group(1), None, m.group(2))
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
        out = subprocess.run(["gh", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
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
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5)
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


def _row_get(row: sqlite3.Row, key: str, default=None):
    """Tolerant column read: the streak query selects fewer columns than the
    incident query, and a missing column must not break dispatch."""
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


def _run_text(row: sqlite3.Row) -> str:
    """Everything a closed run recorded about WHY it failed, as one string: the
    reaper's error plus the captured worker output and exit-reason tail."""
    meta = _kb._json_dict(_row_get(row, "metadata"))
    parts = [_row_get(row, "error") or "", str(meta.get("worker_output") or "")]
    exit_reason = meta.get("exit_reason")
    if isinstance(exit_reason, dict):
        parts.append(str(exit_reason.get("stderr_tail") or ""))
    return "\n".join(parts)


def run_is_environmental(row: sqlite3.Row) -> bool:
    """True when this run failed on the HOST before the card was worked.

    A run that merely ended without progress — a normal ``blocked`` / ``crashed``
    run with real agent output — stays the card's to fix and returns False.
    """
    meta = _kb._json_dict(row["metadata"])
    if (row["outcome"] or "") == "spawn_failed":
        return True
    if meta.get("spawn_fault") or meta.get("env_fault"):
        return True
    try:
        from hermes_cli.kanban_worker_exit import is_spawn_fault

        if is_spawn_fault(meta.get("exit_reason"), _run_text(row)):
            return True
    except Exception:  # pragma: no cover - defensive
        pass
    for line in _run_text(row).splitlines():
        line = line.strip()
        if not line or len(line) > 400:
            continue
        if line.startswith(_ENV_FAULT_PREFIXES):
            return True
        if any(marker in line for marker in _ENV_FAULT_MARKERS):
            return True
        if _ENV_FAULT_RE.search(line):
            return True
    return False


def _environmental_signature(row: sqlite3.Row) -> str:
    """The matching line, whitespace-collapsed: two host faults with the same
    signature are ONE incident, not two independent card failures."""
    for line in _run_text(row).splitlines():
        line = line.strip()
        if not line or len(line) > 400:
            continue
        if (line.startswith(_ENV_FAULT_PREFIXES)
                or any(marker in line for marker in _ENV_FAULT_MARKERS)
                or _ENV_FAULT_RE.search(line)):
            return " ".join(line.split())[:200]
    return "environmental"


def run_started_agent(
    conn: sqlite3.Connection, run_id: int, meta: Optional[dict] = None,
) -> bool:
    """Did the worker actually run? The session id is stamped only AFTER the agent
    builds (``kanban_worker_session``) and a heartbeat / progress note needs a live
    worker loop, so a run with neither never reached the card — the host's problem,
    not the card's. This is the generation-aware half of the rule: it needs no
    hand-maintained fault list, so a NEW host failure mode is still neutral."""
    meta = meta if isinstance(meta, dict) else _kb._json_dict(meta)
    if str(meta.get("worker_session_id") or "").strip():
        return True
    if meta.get("progress_note") or meta.get("progress_at"):
        return True
    row = conn.execute(
        "SELECT 1 FROM task_events WHERE run_id = ? AND kind = 'heartbeat' LIMIT 1",
        (int(run_id),)).fetchone()
    return row is not None


def environmental_streak(
    conn: sqlite3.Connection, task_id: str,
) -> tuple[int, str, list[int]]:
    """Trailing run of environmental failures sharing ONE signature, with that
    signature and the run ids. Counted since the last re-spec / recorded incident
    so an already-handled incident is never announced twice."""
    since = conn.execute(
        "SELECT COALESCE(MAX(created_at), 0) FROM task_events "
        "WHERE task_id = ? AND kind IN ('no_progress_respec', 'host_incident')",
        (task_id,)).fetchone()[0]
    rows = conn.execute(
        "SELECT id, outcome, error, metadata, started_at FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL AND started_at > ? "
        "ORDER BY started_at DESC, id DESC LIMIT 20",
        (task_id, since)).fetchall()
    count, signature, ids = 0, "", []
    for row in rows:
        if not run_is_environmental(row):
            break
        sig = _environmental_signature(row)
        if count and sig != signature:
            break
        signature = signature or sig
        count += 1
        ids.append(int(row["id"]))
    return count, signature, ids


def record_host_incident(
    conn: sqlite3.Connection, task_id: str, limit: int,
) -> Optional[dict]:
    """Repeated identical environmental failures are a HOST incident, not a card
    defect: record ONE ``host_incident`` event plus one operator comment naming the
    signature, and withhold the re-spec. Deduped per signature because the
    dispatcher calls this on every tick while the host stays broken."""
    count, signature, ids = environmental_streak(conn, task_id)
    if not signature or count < max(1, int(limit)):
        return None
    previous = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'host_incident' "
        "ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    if _kb._json_dict(previous["payload"] if previous else None).get("signature") == signature:
        return None
    payload = {"signature": signature, "count": count, "runs": ids}
    with _kb.write_txn(conn):
        _kb._append_event(conn, task_id, "host_incident", payload)
    _kb.add_comment(
        conn, task_id, author="kanban-dispatcher",
        body=(f"BLOCKER:HARNESS infra — {count} consecutive runs failed on the HOST before "
              f"the agent started: {signature}. Not a card defect: no re-spec, no retry "
              f"budget spent. Fix the host (interpreter / dependency environment / workspace "
              f"permissions), then this card can be claimed again."))
    return payload


def reset_no_progress(
    conn: sqlite3.Connection, task_id: str, *, reason: Optional[str] = None,
    author: str = "operator",
) -> bool:
    """Operator escape from a mis-targeted re-spec (``hermes kanban respec-reset``):
    a ``no_progress_respec`` event at/after every run resets the streak to 0 exactly
    as a genuine re-spec does — supported, so nobody hand-writes the board DB again."""
    if _kb.get_task(conn, task_id) is None:
        return False
    with _kb.write_txn(conn):
        _kb._append_event(conn, task_id, "no_progress_respec",
                          {"streak": 0, "reset": True, "author": author,
                           "reason": reason or "operator reset"})
    return True


def no_progress_streak(conn: sqlite3.Connection, task_id: str, current_head: Optional[str]) -> int:
    """Count consecutive trailing ended runs (since the last re-spec) that made
    no progress: outcome is not a forward status change (block/park/crash do
    not count — that is the red-CI hold loop), workspace HEAD unchanged across
    the run, and no wait_on handle turned passing since it started."""
    last_respec = conn.execute(
        "SELECT COALESCE(MAX(created_at), 0) FROM task_events WHERE task_id = ? AND kind = 'no_progress_respec'",
        (task_id,)).fetchone()[0]
    runs = conn.execute(
        "SELECT id, outcome, error, metadata, started_at, ended_at FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL AND started_at > ? ORDER BY started_at DESC, id DESC",
        (task_id, last_respec)).fetchall()
    passing_at = [int(r["created_at"]) for r in conn.execute(
        "SELECT created_at, payload FROM task_events WHERE task_id = ? AND kind = 'wait_on_changed'",
        (task_id,)) if _kb._json_dict(r["payload"]).get("passing")]
    streak = 0
    head_after = current_head
    for r in runs:
        meta = _kb._json_dict(r["metadata"])
        head_start = meta.get("head_start")
        progressed = (
            (r["outcome"] or "") in PROGRESS_OUTCOMES
            or (head_start and head_after and head_start != head_after)
            # A handle that turned green during or after this run is new
            # evidence: runs before it don't count toward the streak.
            or any(t >= r["started_at"] for t in passing_at)
        )
        if progressed:
            break
        # An infrastructure crash says nothing about the card: it must not count
        # toward the streak (that is the mis-targeted re-spec this guards against)
        # and must not reset it either (a genuine no-progress run before it stays
        # counted). Same reasoning as ``rate_limited`` in the protocol-violation
        # streak: neutral evidence is skipped, never scored. A progress ending is
        # judged above, before this gate — a run cannot lose its progress by
        # having no session stamp.
        if run_is_environmental(r) or not run_started_agent(conn, int(r["id"]), meta):
            continue
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
    if record_host_incident(conn, task_id, limit) is not None:
        # The trailing runs never reached the card; a re-spec would be aimed at the
        # wrong target — exactly the incident this fixes.
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
