"""Task graph initialization and atomic decomposition persistence."""
from __future__ import annotations

import sqlite3
import time
from typing import Any, Optional

def inherit_creator_origin(
    conn: sqlite3.Connection, task_id: str, creator_task_id: Optional[str], *,
    created_at: int,
) -> None:
    """Copy durable origin inside creation's transaction, never adding dependencies."""
    if not creator_task_id:
        return
    from hermes_cli.kanban_db import _inherit_notify_subs

    conn.execute(
        "UPDATE tasks SET session_id = COALESCE(session_id, "
        "(SELECT session_id FROM tasks WHERE id = ?)) WHERE id = ?",
        (creator_task_id, task_id),
    )
    _inherit_notify_subs(conn, task_id, (creator_task_id,), created_at=created_at)


def initial_task_state(
    conn: sqlite3.Connection, parents: tuple[str, ...], initial_status: str,
    triage: bool, tenant: Optional[str],
) -> tuple[str, Optional[str]]:
    """Resolve state and tenant under the creator's write transaction.

    Parent order breaks ties in this soft namespace; explicit tenant wins.
    Validate parents even for parked tasks so links never dangle.
    """
    rows = {}
    if parents:
        rows = {row["id"]: row for row in conn.execute(
            "SELECT id, status, tenant FROM tasks WHERE id IN "
            "(" + ",".join("?" * len(parents)) + ")", parents,
        )}
        missing = [pid for pid in parents if pid not in rows]
        if missing:
            raise ValueError(f"unknown parent task(s): {', '.join(missing)}")
        if tenant is None:
            tenant = next((rows[pid]["tenant"] for pid in parents if rows[pid]["tenant"]), None)
    if initial_status == "blocked":
        return "blocked", tenant
    if triage:
        return "triage", tenant
    if any(row["status"] != "done" for row in rows.values()):
        return "todo", tenant
    return "ready", tenant


def _validate_children_graph(children: list) -> None:
    """DB-free shape check + Kahn's cycle check on the sibling graph (a cycle
    would deadlock every involved child in ``todo`` forever)."""
    for idx, child in enumerate(children):
        if not isinstance(child, dict):
            raise ValueError(f"child[{idx}] is not a dict")
        title = child.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"child[{idx}].title is required")
        parents_idx = child.get("parents") or []
        if not isinstance(parents_idx, list):
            raise ValueError(f"child[{idx}].parents must be a list")
        for p in parents_idx:
            if not isinstance(p, int) or p < 0 or p >= len(children):
                raise ValueError(f"child[{idx}].parents[{p}] is not a valid index into children")
            if p == idx:
                raise ValueError(f"child[{idx}] cannot list itself as a parent")

    in_deg = [0] * len(children)
    adj: list[list[int]] = [[] for _ in children]
    for i, c in enumerate(children):
        for p in (c.get("parents") or []):
            adj[p].append(i)
            in_deg[i] += 1
    queue = [i for i in range(len(children)) if in_deg[i] == 0]
    seen = 0
    while queue:
        seen += 1
        for nb in adj[queue.pop()]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)
    if seen != len(children):
        raise ValueError("cyclic dependency detected in decomposed children list")


def decompose_triage_task(
    conn: sqlite3.Connection, task_id: str, *, root_assignee: Optional[str], children: list[dict],
    author: Optional[str] = None, auto_promote: bool = True,
) -> Optional[list[str]]:
    """Fan a triage task out into children and move the root to ``todo``; the root
    waits on every child and wakes (``ready``) when all are done.

    ``children``: dicts of ``title`` (required), ``body``, ``assignee``,
    ``parents`` (indices into this list), optional workspace overrides.
    Returns child ids in input order, or None when the root is missing / not
    in triage, or has already decomposed. Atomic: malformed entries abort fan-out.
    """
    from hermes_cli.kanban_db import (
        _canonical_assignee, _link, _append_event, _insert_comment,
        write_txn, recompute_ready, _fold_into_outcome_key,
    )

    if not children:
        return None
    if root_assignee is not None:
        root_assignee = _canonical_assignee(root_assignee)
    _validate_children_graph(children)

    # ONE txn so the fan-out is atomic; helpers that open their own write_txn
    # (create_task, link_tasks, add_comment) must not be called in here.
    now = int(time.time())
    with write_txn(conn):
        root_row = conn.execute(
            "SELECT id, status, tenant, workspace_kind, workspace_path, project_id "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if root_row is None or root_row["status"] != "triage":
            return None
        # Dependency links alone do not imply lineage. The completion event is
        # committed with the graph, and survives re-triage or unlinking.
        if conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'decomposed' LIMIT 1",
            (task_id,),
        ).fetchone():
            return None
        # Resolve identities FIRST (outcome_key folding can map a child onto
        # an existing task, the root, or a sibling), then validate the
        # RESOLVED graph against existing links before any write.
        plan = _resolve_decomposed_children(conn, task_id, root_row, children)
        child_ids = [p["id"] for p in plan]
        edges: list[tuple[str, str]] = []
        for idx, child in enumerate(children):
            for p_idx in child.get("parents") or []:
                edges.append((child_ids[p_idx], child_ids[idx]))
        # Root waits for the whole graph: link it under EVERY child.
        edges.extend((cid, task_id) for cid in dict.fromkeys(child_ids))
        _validate_resolved_edges(conn, edges)
        for child, entry in zip(children, plan):
            if entry["fold_into"]:
                _fold_into_outcome_key(
                    conn, entry["id"], title=child["title"], body=child.get("body"),
                    author=author or "decomposer", outcome_key=entry["outcome_key"], now=now,
                )
            elif entry["insert"]:
                _insert_decomposed_child(conn, task_id, root_row, child, author, now,
                                         new_id=entry["id"], outcome_key=entry["outcome_key"])
        linked: set[tuple[str, str]] = set()
        for parent_id, child_id in edges:
            if (parent_id, child_id) in linked:
                continue
            linked.add((parent_id, child_id))
            _link(conn, parent_id, child_id)
            if child_id != task_id:
                _append_event(conn, child_id, "linked", {"parent": parent_id, "child": child_id})
        # Flip the root triage -> todo, assignee -> orchestrator.
        sets = ["status = 'todo'"]
        params: list[Any] = []
        if root_assignee is not None:
            sets.append("assignee = ?")
            params.append(root_assignee)
        params.append(task_id)
        conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", tuple(params))
        if author and author.strip():
            _insert_comment(
                conn, task_id, author.strip(),
                "Decomposed into " + ", ".join(child_ids)
                + ". Root will wake when all children complete.",
                now,
            )
        _append_event(
            conn, task_id, "decomposed", {"child_ids": child_ids, "root_assignee": root_assignee},
        )
    # Outside the txn (own IMMEDIATE txn). ``auto_promote=False`` leaves the
    # children in ``todo`` for manual-review-first workflows.
    if auto_promote:
        recompute_ready(conn)
    return child_ids


def _resolve_decomposed_children(
    conn: sqlite3.Connection, root_id: str, root_row: sqlite3.Row, children: list[dict],
) -> list[dict]:
    """Per child: ``{id, outcome_key, fold_into, insert}`` without writing.

    Same outcome_key rule as create_task (the TARGET board's policy, resolved
    from ``conn``; one OPEN task per (project, key)). A keyed child is scoped
    to the root's project. In-batch duplicates share the first sibling's id.
    """
    from hermes_cli.kanban_db import (
        _new_task_id, check_outcome_key, _open_task_for_outcome_key, board_for_conn,
    )

    board = board_for_conn(conn)
    plan: list[dict] = []
    batch: dict[str, str] = {}
    for child in children:
        key = check_outcome_key(child.get("outcome_key"), board)
        if key and key in batch:
            plan.append({"id": batch[key], "outcome_key": key, "fold_into": True, "insert": False})
            continue
        existing = _open_task_for_outcome_key(conn, root_row["project_id"], key) if key else None
        if existing:
            entry = {"id": existing, "outcome_key": key, "fold_into": True, "insert": False}
        else:
            entry = {"id": _new_task_id(), "outcome_key": key, "fold_into": False, "insert": True}
        if key:
            batch[key] = entry["id"]
        plan.append(entry)
    return plan


def _validate_resolved_edges(conn: sqlite3.Connection, edges: list[tuple[str, str]]) -> None:
    """Reject self-dependencies and cycles in the resolved graph, including
    links already in the DB (a folded child may already depend on the root)."""
    for parent, child in edges:
        if parent == child:
            raise ValueError(
                f"decomposition folds a dependency onto the same task {parent} "
                "(outcome_key collision); cyclic dependency rejected"
            )
    adj: dict[str, set[str]] = {}
    for parent, child in edges:
        adj.setdefault(parent, set()).add(child)
    # Reachability over existing + proposed links from each proposed child
    # back to its parent means the new edge closes a cycle.
    def _reaches(src: str, dst: str) -> bool:
        seen, stack = set(), [src]
        while stack:
            n = stack.pop()
            if n == dst:
                return True
            if n in seen:
                continue
            seen.add(n)
            stack.extend(adj.get(n, ()))
            stack.extend(r[0] for r in conn.execute(
                "SELECT child_id FROM task_links WHERE parent_id = ?", (n,)))
        return False
    for parent, child in edges:
        if _reaches(child, parent):
            raise ValueError(
                f"cyclic dependency detected: {child} already leads to {parent} "
                "after outcome_key folding"
            )


def _insert_decomposed_child(
    conn: sqlite3.Connection, root_id: str, root_row: sqlite3.Row, child: dict,
    author: Optional[str], now: int, *, new_id: str, outcome_key: Optional[str],
) -> str:
    """Insert one decomposed child as ``todo`` (linked under the root later so
    the dispatcher only ever sees a coherent graph); returns its id.

    Workspace: per-child override wins, else inherit the root's kind. Path
    inherits only when kinds match (a 'dir' child must not point at the
    root's worktree) and NEVER for worktrees — siblings dispatch concurrently
    and one shared checkout would put them all on the first sibling's branch
    with no lock; leaving it unset makes dispatch materialize a fresh
    ``<repo>/.worktrees/<child-id>`` per child from the board anchor.
    """
    from hermes_cli.kanban_db import _canonical_assignee, _append_event

    child_project = root_row["project_id"] if outcome_key else None
    root_ws_kind = root_row["workspace_kind"] or "scratch"
    child_ws_kind = child.get("workspace_kind") or root_ws_kind
    if child.get("workspace_path"):
        child_ws_path = child.get("workspace_path")
    elif child_ws_kind == "worktree":
        child_ws_path = None
    elif child_ws_kind == root_ws_kind:
        child_ws_path = root_row["workspace_path"]
    else:
        child_ws_path = None
    body = child.get("body")
    conn.execute(
        "INSERT INTO tasks "
        "(id, title, body, assignee, status, workspace_kind, "
        " workspace_path, tenant, created_at, created_by, outcome_key, project_id) "
        "VALUES (?, ?, ?, ?, 'todo', ?, ?, ?, ?, ?, ?, ?)",
        (
            new_id, child["title"].strip(), body if isinstance(body, str) else None,
            _canonical_assignee(child.get("assignee")), child_ws_kind, child_ws_path,
            root_row["tenant"], now, (author or "decomposer"), outcome_key, child_project,
        ),
    )
    _append_event(
        conn, new_id, "created", {"by": author or "decomposer", "from_decompose_of": root_id},
    )
    inherit_creator_origin(conn, new_id, root_id, created_at=now)
    return new_id
