"""Configurable workflow stages for kanban tasks (``kanban.stages``).

A stage is ``{key, owner, status}``. Moving a task to a stage records
``current_step_key``, hands the task to the stage owner, and puts it in the
stage's status so the normal dispatcher lanes pick it up (``ready`` for the
implementation lane, ``review`` for the review lane). The dispatcher's WIP
caps (``kanban.max_in_progress`` / ``_per_profile``) still apply: a stage move
only queues the task, it never spawns a worker itself.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Optional

# Statuses a stage may target. ``running``/``archived``/``blocked`` are owned by
# the dispatcher / operators and are not valid stage targets.
STAGE_STATUSES = ("triage", "todo", "scheduled", "ready", "review", "done")


@dataclass(frozen=True)
class Stage:
    key: str
    owner: Optional[str]
    status: str


def _parse(raw: Any) -> list[Stage]:
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError("kanban.stages must be a list of {key, owner, status}")
    out: list[Stage] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or not str(item.get("key") or "").strip():
            raise ValueError(f"kanban.stages[{i}] needs a non-empty 'key'")
        key = str(item["key"]).strip()
        if key in seen:
            raise ValueError(f"kanban.stages: duplicate key {key!r}")
        seen.add(key)
        owner = item.get("owner")
        owner = None if owner in (None, "", "none", "None") else str(owner).strip()
        status = str(item.get("status") or "ready").strip()
        if status not in STAGE_STATUSES:
            raise ValueError(
                f"kanban.stages[{key}]: status {status!r} not in {list(STAGE_STATUSES)}")
        out.append(Stage(key, owner, status))
    return out


def load_stages(raw: Any = None) -> list[Stage]:
    """Stages from ``raw`` or the active config's ``kanban.stages``."""
    if raw is None:
        try:
            from hermes_cli.config import load_config
            raw = ((load_config() or {}).get("kanban") or {}).get("stages")
        except Exception:
            raw = None
    return _parse(raw)


def get_stage(key: str, stages: Optional[list[Stage]] = None) -> Stage:
    stages = load_stages() if stages is None else stages
    if not stages:
        raise ValueError("no kanban.stages configured "
                         "(hermes config set kanban.stages '[{key: ..., owner: ..., status: ...}]')")
    for s in stages:
        if s.key == key:
            return s
    raise ValueError(f"unknown stage {key!r}; valid stages: {', '.join(s.key for s in stages)}")


def next_stage(current: Optional[str], stages: Optional[list[Stage]] = None) -> Stage:
    stages = load_stages() if stages is None else stages
    if not stages:
        get_stage("", stages)  # raises the "not configured" error
    if not current:
        return stages[0]
    keys = [s.key for s in stages]
    if current not in keys:
        raise ValueError(f"task is at unknown stage {current!r}; set one explicitly")
    idx = keys.index(current)
    if idx + 1 >= len(stages):
        raise ValueError(f"task is already at the last stage {current!r}")
    return stages[idx + 1]


def set_step_key(conn: sqlite3.Connection, task_id: str, key: Optional[str]) -> bool:
    """Record ``current_step_key`` only (no handoff). ``key`` is validated."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import write_txn

    if key is not None:
        get_stage(key)
    with write_txn(conn):
        row = conn.execute("SELECT current_step_key FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            return False
        conn.execute("UPDATE tasks SET current_step_key = ? WHERE id = ?", (key, task_id))
        kb._append_event(conn, task_id, "stage_set", {"from": row["current_step_key"], "to": key})
    kb.notify_task_updated(conn, task_id, ("current_step_key",))
    return True


def move_to_stage(
    conn: sqlite3.Connection, task_id: str, key: str, *, note: Optional[str] = None,
    author: str = "user", keep_status: bool = False,
) -> dict:
    """Move ``task_id`` to stage ``key``: set the step key, hand off to the owner,
    set the stage status, comment ``STAGE: <key>`` (+ note) and log a
    ``stage_changed`` event. ``keep_status`` records stage+owner but leaves the
    status untouched (e.g. to stage a parked triage/scheduled card)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import write_txn

    stage = get_stage(key)
    owner = kb._canonical_assignee(stage.owner)
    with write_txn(conn):
        row = conn.execute(
            "SELECT status, claim_lock, assignee, current_step_key FROM tasks WHERE id = ?",
            (task_id,)).fetchone()
        if not row:
            raise ValueError(f"unknown task {task_id}")
        if row["status"] == "archived":
            raise ValueError(f"cannot change stage of archived task {task_id}")
        if row["status"] == "running" and row["claim_lock"] is not None:
            raise RuntimeError(f"cannot change stage of {task_id}: currently running (claimed). "
                               "Wait for the worker to hand off or reclaim it first.")
        status = row["status"]
        if not keep_status and stage.status != "done":
            status = stage.status
            # Dispatchable lanes stay dependency-gated exactly like create/recompute_ready.
            if status in ("ready", "review") and not kb._parents_satisfied(conn, task_id):
                status = "todo"
        sets = ["current_step_key = ?", "assignee = ?", "status = ?"]
        params: list[Any] = [stage.key, owner, status]
        if owner != row["assignee"]:
            sets += ["consecutive_failures = 0", "last_failure_error = NULL"]
        if status != row["status"] and status in ("ready", "todo", "review", "triage"):
            sets.append("scheduled_wake_at = NULL")
        conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", (*params, task_id))
        payload = {"from": row["current_step_key"], "to": stage.key,
                   "assignee": owner, "from_assignee": row["assignee"],
                   "status": status, "from_status": row["status"]}
        kb._append_event(conn, task_id, "stage_changed", payload)
        if owner != row["assignee"]:
            kb._append_event(conn, task_id, "assigned", {"assignee": owner, "from": row["assignee"]})
        body = f"STAGE: {stage.key}" + (f"\n{note.strip()}" if note and note.strip() else "")
        kb.add_comment(conn, task_id, author, body)
    if stage.status == "done" and not keep_status:
        if not kb.complete_task(conn, task_id, result=note or f"stage {stage.key}"):
            raise RuntimeError(f"stage {stage.key!r} recorded but {task_id} could not be completed "
                               f"from status {row['status']!r}")
        payload["status"] = "done"
    kb.notify_task_updated(conn, task_id, ("current_step_key", "assignee", "status"))
    payload["at"] = int(time.time())
    return payload
