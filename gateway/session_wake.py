"""One admission owner per (profile, session), shared by every goal/heartbeat path.

The poller is just a clock; ownership lives in SQLite, not in a timer's task or a
routing key (handoff leaves old keys behind). A goal owns the lane while active OR
waiting; an independent heartbeat must not poll through its barrier. The token
covers FIFO admission through completion, and is revalidated at agent entry.
"""
import json
import time
import uuid


def claim(session_id, kind, *, revision=None):
    from hermes_cli.goal_controller import _db, read
    now = time.time()
    def apply(conn):
        goal = read(conn, session_id)
        if kind == "goal":
            if goal is None or goal.status != "active" or goal.revision != revision:
                return None
        elif goal and goal.status in {"active", "waiting"}:
            return None
        key = f"session_wake:{session_id}"
        row = conn.execute("SELECT value FROM state_meta WHERE key=?", (key,)).fetchone()
        if row:
            prior = json.loads(row[0])
            # Crash recovery only. Normal completion releases its exact token.
            if now < prior["expires"]:
                return None
        token = uuid.uuid4().hex
        value = {"token": token, "kind": kind, "revision": revision, "expires": now + 3600}
        conn.execute("INSERT OR REPLACE INTO state_meta(key,value) VALUES (?,?)", (key, json.dumps(value)))
        return token
    return _db()._execute_write(apply)


def release(session_id, token):
    from hermes_cli.goal_controller import _db
    def apply(conn):
        key = f"session_wake:{session_id}"
        row = conn.execute("SELECT value FROM state_meta WHERE key=?", (key,)).fetchone()
        if row and json.loads(row[0]).get("token") == token:
            conn.execute("DELETE FROM state_meta WHERE key=?", (key,))
    _db()._execute_write(apply)


def stamp(event, session_id, token):
    event._session_wake_id = session_id
    event._session_wake_token = token


def current(event):
    sid = getattr(event, "_session_wake_id", None)
    if not sid:
        return True
    from hermes_cli.goal_controller import _db, read
    def check(conn):
        row = conn.execute("SELECT value FROM state_meta WHERE key=?", (f"session_wake:{sid}",)).fetchone()
        if not row:
            return False
        owner = json.loads(row[0])
        if owner["token"] != event._session_wake_token:
            return False
        goal = read(conn, sid)
        if owner["kind"] == "goal":
            return goal is not None and goal.status == "active" and goal.revision == owner["revision"]
        return goal is None or goal.status not in {"active", "waiting"}
    return _db()._execute_write(check)


def settle(event):
    sid = getattr(event, "_session_wake_id", None)
    if sid:
        release(sid, event._session_wake_token)


def track(adapter, key, event):
    """Release even when auth/stop/preparation rejects before post-turn hooks run."""
    task = getattr(adapter, "_session_tasks", {}).get(key)
    if task is not None:
        task.add_done_callback(lambda _: settle(event))
    elif key not in getattr(adapter, "_active_sessions", {}):
        settle(event)
