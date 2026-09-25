"""Supervisor for a kanban task executed by a native coding-agent CLI.

A raw coding-agent CLI never calls ``kanban_complete`` — it just exits. This
module is the thin process that bridges that gap: the dispatcher spawns *this*
(its PID is the worker PID the board records), it runs the CLI in the task's
workspace, and when the CLI exits it drives the task to a terminal state:

* rc == 0  → ``complete_task``
* rc != 0  → ``block_task`` with the tail of the worker log as the error

So for a native-executor run the CLI's **exit status is the protocol**. The
dispatcher knows this (``tasks.executor`` is set) and therefore does not count
a clean exit as a protocol violation the way it does for a Hermes worker.

Invoked as ``python -m hermes_cli.kanban_native_worker``; every parameter
arrives through the environment (see ``_default_spawn``), so no task text ever
lands on this process's command line.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

# How much of the worker log is attached to a failing task's block reason.
LOG_TAIL_BYTES = 4000
LOG_TAIL_CHARS = 2000


def read_log_tail(log_path: str, *, max_bytes: int = LOG_TAIL_BYTES) -> str:
    """Return the tail of the worker log, best-effort."""
    if not log_path:
        return ""
    try:
        size = os.path.getsize(log_path)
        with open(log_path, "rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            data = fh.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace").strip()
    if len(text) > LOG_TAIL_CHARS:
        text = text[-LOG_TAIL_CHARS:]
    return text


def finalize(task_id: str, returncode: int, log_path: str) -> bool:
    """Drive the task to a terminal state from the CLI's exit status."""
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        run_id_raw = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
        try:
            expected_run_id = int(run_id_raw) if run_id_raw else None
        except ValueError:
            expected_run_id = None
        if returncode == 0:
            return bool(
                kb.complete_task(
                    conn,
                    task_id,
                    result=(
                        f"native executor run finished (rc=0). "
                        f"Worker log: {log_path}"
                    ),
                    summary="native executor run finished (rc=0)",
                    metadata={
                        "executor": os.environ.get("HERMES_KANBAN_EXECUTOR"),
                        "exit_code": 0,
                        "log": log_path,
                    },
                    expected_run_id=expected_run_id,
                )
            )
        tail = read_log_tail(log_path)
        reason = (
            f"native executor exited with code {returncode}"
            + (f"\n--- worker log tail ---\n{tail}" if tail else "")
        )
        return bool(
            kb.block_task(
                conn,
                task_id,
                reason=reason,
                kind="transient",
                expected_run_id=expected_run_id,
            )
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    raw_argv = os.environ.get("HERMES_KANBAN_NATIVE_ARGV", "")
    log_path = os.environ.get("HERMES_KANBAN_NATIVE_LOG", "")
    workspace = os.environ.get("HERMES_KANBAN_WORKSPACE", "")
    if not task_id or not raw_argv:
        print(
            "kanban native worker: HERMES_KANBAN_TASK and "
            "HERMES_KANBAN_NATIVE_ARGV are required",
            file=sys.stderr,
        )
        return 2
    try:
        cmd = json.loads(raw_argv)
        if not isinstance(cmd, list) or not cmd:
            raise ValueError("argv must be a non-empty JSON list")
    except Exception as exc:
        print(f"kanban native worker: bad argv payload ({exc})", file=sys.stderr)
        return 2

    cwd = workspace if workspace and os.path.isdir(workspace) else None
    try:
        proc = subprocess.run(  # noqa: S603 -- argv built by the dispatcher
            cmd,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        returncode = int(proc.returncode)
    except FileNotFoundError:
        print(f"kanban native worker: {cmd[0]} not found", file=sys.stderr)
        returncode = 127
    except OSError as exc:
        print(f"kanban native worker: {exc}", file=sys.stderr)
        returncode = 126
    sys.stdout.flush()
    sys.stderr.flush()

    try:
        finalize(task_id, returncode, log_path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"kanban native worker: finalize failed ({exc})", file=sys.stderr)
        return 1
    # Always exit 0: the task's outcome is already recorded in the DB. A
    # non-zero exit here would be read as a crashed supervisor.
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
