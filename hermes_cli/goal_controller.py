"""The single logical writer of session goals.

Commands reduce the current row inside SessionDB's BEGIN IMMEDIATE. Slow effects
(gates and the judge) reduce an isolated snapshot outside that transaction; their
whole result, including notices and continuation prompts, commits only if its
revision still owns the row. A lock around an old manager snapshot cannot provide
that guarantee: clear, replacement, compression and a completion audit all race it.

The manager is the compatibility/policy facade. Its decorated methods are reducers,
not writers; _save stages a value. Cancellation leaves a versioned tombstone, so an
old verdict can never recreate a goal, even after a clear followed by a new goal.
"""
from __future__ import annotations

import copy
from functools import wraps


class GoalConflict(RuntimeError):
    """A slow effect no longer describes the current goal."""


def read(conn, session_id):
    from hermes_cli.goals import GoalState
    row = conn.execute("SELECT value FROM state_meta WHERE key = ?", (f"goal:{session_id}",)).fetchone()
    return GoalState.from_json(row[0]) if row and row[0] else None


def write(conn, session_id, state, previous):
    """Only goal-row write primitive, also used inside session lifecycle transactions."""
    if state is None:
        return
    state.normalize()
    state.revision = (previous.revision if previous else 0) + 1
    conn.execute("INSERT INTO state_meta(key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (f"goal:{session_id}", state.to_json()))


def cancel(conn, session_id, reason="cancelled"):
    state = read(conn, session_id)
    if state is None or state.status in {"done", "cancelled"}:
        return
    previous = copy.deepcopy(state)
    state.clear_wait()
    state.status = "cancelled"
    state.paused_reason = reason
    write(conn, session_id, state, previous)


def _db():
    from hermes_cli.goals import _get_session_db
    db = _get_session_db()
    if db is None:
        raise RuntimeError("goal storage unavailable")
    return db


def save(session_id, state):
    """Versioned compatibility entry point; callers cannot overwrite a newer row."""
    def apply(conn):
        current = read(conn, session_id)
        if (current.revision if current else 0) != state.revision:
            raise GoalConflict("goal changed before save")
        write(conn, session_id, state, current)
    _db()._execute_write(apply)


def migrate(old_id, new_id):
    def apply(conn):
        state = read(conn, old_id)
        if state is None or state.status == "cancelled" or read(conn, new_id) is not None:
            return False
        write(conn, new_id, copy.deepcopy(state), None)
        cancel(conn, old_id, "compression continuation")
        return True
    return _db()._execute_write(apply)


def _worker(manager, state):
    worker = copy.copy(manager)
    worker._state = copy.deepcopy(state)
    worker._reducing = True
    worker._pending_state = None
    worker._archives = []
    return worker


def _commit(conn, manager, worker, previous):
    for state in worker._archives:
        from hermes_cli.goals import _archive_key
        conn.execute("INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, ?)",
                     (_archive_key(manager.session_id), state.to_json()))
    staged = worker._pending_state
    if staged is not None:
        write(conn, manager.session_id, staged, previous)
    manager._state = worker._state


def command(fn):
    """Serialize read → reduce → write, including nested compatibility commands."""
    @wraps(fn)
    def dispatch(self, *args, **kwargs):
        if getattr(self, "_reducing", False):
            return fn(self, *args, **kwargs)
        def apply(conn):
            previous = read(conn, self.session_id)
            worker = _worker(self, previous)
            result = fn(worker, *args, **kwargs)
            _commit(conn, self, worker, previous)
            return result
        return _db()._execute_write(apply)
    return dispatch


def effect(fn):
    """Optimistic transaction: never hold the SQLite writer during a model/shell call."""
    @wraps(fn)
    def dispatch(self, *args, **kwargs):
        if getattr(self, "_reducing", False):
            return fn(self, *args, **kwargs)
        from hermes_cli.goals import load_goal, _decision
        previous = load_goal(self.session_id)
        worker = _worker(self, previous)
        result = fn(worker, *args, **kwargs)
        if isinstance(result, dict) and result.get("should_continue") and worker._state:
            worker._state.wake_pending = True
            worker._pending_state = worker._state
        def apply(conn):
            current = read(conn, self.session_id)
            if (current.revision if current else 0) != (previous.revision if previous else 0):
                self._state = current
                return _decision(current.status if current else None, False, None,
                                 "stale", "goal changed during evaluation", "")
            _commit(conn, self, worker, previous)
            if isinstance(result, dict):
                result["revision"] = self._state.revision if self._state else None
                if self._state:
                    result["status"] = self._state.status
            return result
        return _db()._execute_write(apply)
    return dispatch
