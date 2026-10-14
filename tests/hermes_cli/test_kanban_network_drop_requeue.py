"""#32: a worker whose network drops must not leave its card running nor count as finished.

3f2d79fa9e deleted the network-wait hooks on the premise that a connection error is
classified ``timeout`` and the worker exits rc 75, which the dispatcher requeues without
spending the failure budget. This walks that chain end to end: real classifier, real
exit-code mapping, real dispatcher reap on a real kanban DB.
"""
from __future__ import annotations

import time
from pathlib import Path

import httpx
import openai
import pytest

from agent.error_classifier import classify_api_error
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.cli_single_query import _single_query_exit_code

_REQ = httpx.Request("POST", "https://api.example/v1/chat")
NETWORK_ERRORS = [
    ConnectionError("network down"),
    OSError(51, "Network is unreachable"),
    httpx.ConnectError("Name resolution failed", request=_REQ),
    httpx.ReadError("connection reset", request=_REQ),
    openai.APIConnectionError(request=_REQ),
]


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kbd, "_worker_alive", lambda *_a: False)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return h


@pytest.mark.parametrize("err", NETWORK_ERRORS, ids=lambda e: type(e).__name__)
def test_network_drop_requeues_card_without_failure_or_finish(home, monkeypatch, err):
    reason = classify_api_error(err).reason.value
    result = {"completed": False, "failed": True, "failure_reason": reason}
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="swe")
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        rc = _single_query_exit_code(result)
        assert rc == kb.KANBAN_RATE_LIMIT_EXIT_CODE, reason
        for i in range(4):  # more drops than any failure_limit would allow
            assert kb.claim_task(conn, tid) is not None
            pid = 86000 + i
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET worker_pid = ?, started_at = ? WHERE id = ?",
                             (pid, int(time.time()) - 3600, tid))
            kbd._record_worker_exit(pid, rc << 8)
            kbd.detect_crashed_workers(conn)
            task = kb.get_task(conn, tid)
            assert task.status == "ready"  # requeued: not stuck running, not done
            assert task.consecutive_failures == 0
            assert kb.list_runs(conn, tid)[-1].outcome == "rate_limited"
