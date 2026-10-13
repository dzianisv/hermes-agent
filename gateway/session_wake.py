"""One admission owner per (profile, session), shared by every goal/heartbeat path.

The poller is just a clock; ownership lives in SQLite, not in a timer's task or a
routing key (handoff leaves old keys behind). A goal owns the lane while active OR
waiting; an independent heartbeat must not poll through its barrier. The token
covers FIFO admission through completion, and is revalidated at agent entry.
"""
import contextvars
import json
import os
import uuid


def _owner_alive(owner):
    from gateway.status import get_process_start_time, start_time_fingerprints_match
    import psutil
    pid = owner.get("pid")
    # Legacy leases have no identity; fail closed rather than steal a live turn.
    if not isinstance(pid, int) or pid <= 0:
        return True
    try:
        process = psutil.Process(pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return False
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True
    started = get_process_start_time(pid)
    recorded = owner.get("start_time")
    return (started is None or recorded is None
            or start_time_fingerprints_match(recorded, started))


def claim(session_id, kind, *, revision=None):
    from hermes_cli.goal_controller import _db, read
    from gateway.status import get_process_start_time
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
            # Elapsed wall time never proves the owner died (long turns / suspended judge).
            if _owner_alive(prior):
                return None
        token = uuid.uuid4().hex
        value = {"token": token, "kind": kind, "revision": revision,
                 "pid": os.getpid(), "start_time": get_process_start_time(os.getpid())}
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
    event._session_wake_context = contextvars.copy_context()


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
    if not sid or getattr(event, "_session_wake_settled", False):
        return
    def finish():
        release(sid, event._session_wake_token)
        manager = getattr(event, "_heartbeat_manager", None)
        if manager is not None:
            from gateway.run_heartbeat_acceptance import settle_heartbeat_attempt
            settle_heartbeat_attempt(event, manager)
    scope = getattr(event, "_session_wake_context", None)
    if scope is not None:
        scope.copy().run(finish)
    else:
        finish()
    event._session_wake_settled = True


def bind_task(event, task):
    """Bind cleanup before scheduling can be cancelled, including FIFO drain tasks."""
    if getattr(event, "_session_wake_token", None):
        event._session_wake_task = task
        task.add_done_callback(lambda _: settle(event))


def track(adapter, key, event):
    """Rejected attempts settle now; accepted events settle on their own task/drop.

    The task currently at key may still belong to a human. Never attach this
    event's cleanup to it: queue acceptance isn't execution ownership.
    """
    if getattr(event, "_session_wake_task", None) is None and not getattr(event, "_gateway_accepted", False):
        settle(event)
