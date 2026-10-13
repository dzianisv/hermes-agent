"""Harness #32: kanban workers must not quit when the network drops.

Real incidents: runs 6693 (t_ec089e4b) / 6695 (t_eca1f25c) burned both stop-loop nudges against an
offline pi-acp provider and exited rc=0 with no board call; run 6801 (t_4a1dcb17) hit "API call
failed after 3 retries ... You may be offline" and exited rc=75. The fix: wait for connectivity
with bounded backoff and resume the same turn; if the budget is spent, record an explicit
``network_unavailable`` block before exiting.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.error_classifier import classify_api_error
from agent.turn_retry_state import TurnRetryState


OFFLINE_MSG = ("Hermes can't reach the model provider. You may be offline. "
               "Check your internet connection and try again.")


class APIConnectionError(Exception):
    """Same class name as openai.APIConnectionError (classifier keys on the name)."""

    def __init__(self, message="Connection error."):
        super().__init__(message)
        self.message = message
        self.status_code = None
        self.response = SimpleNamespace(headers={})
        self.body = None


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_net32")
    monkeypatch.delenv("HERMES_KANBAN_STOP_NUDGE", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_NETWORK_WAIT_SECONDS", "600")
    return monkeypatch


# ---------------------------------------------------------------- retry path (run 6801)
class _Agent:
    log_prefix = ""
    verbose = False
    provider = "copilot"
    platform = "cli"
    _credential_pool = None
    _current_streamed_assistant_text = ""
    _interrupt_requested = False

    def __init__(self):
        self.statuses = []
        self._auto_recovery_cycles = 5

    def _has_pending_fallback(self):
        return False

    def _try_activate_fallback(self, **kw):
        return False

    def _try_recover_primary_transport(self, *a, **k):
        return False

    def _has_content_after_think_block(self, text):
        return bool(text.strip())

    def _emit_diagnostic_status(self, text):
        self.statuses.append(str(text))

    def _summarize_api_error(self, e):
        return str(e)

    def _client_log_context(self):
        return ""

    def clear_interrupt(self, **kw):
        return False

    def __getattr__(self, name):
        return lambda *a, **k: None


def _settle(agent, err, retry):
    from agent.turn_api_error import settle_unrecovered_error
    classified = classify_api_error(err, provider=agent.provider)
    with patch("agent.conversation_loop._is_copilot_provider", lambda a: False), \
            patch("agent.turn_recovery.interruptible_backoff_sleep", lambda *a, **k: None), \
            patch("agent.turn_api_error.max_retries_exhausted_result",
                  lambda *a, **k: {"failed": True, "failure_reason": "timeout", "error": OFFLINE_MSG}):
        return settle_unrecovered_error(
            agent, api_error=err, classified=classified, _retry=retry, status_code=None,
            error_msg=str(err).lower(), is_context_length_error=False, is_rate_limited=False,
            _is_zai_coding_overload=False, _provider=agent.provider, _base="https://api.githubcopilot.com",
            _model="m", messages=[], api_messages=[], api_kwargs=None, active_system_prompt=None,
            conversation_history=None, approx_tokens=0, retry_count=3, max_retries=3,
            compression_attempts=0, api_call_count=1,
        )


def test_offline_worker_waits_for_network_and_resumes_same_call_after_ladder_spent(worker_env):
    """Run 6801 shape: ladder 5/5 spent, connection error. Old: return (turn ends, rc=75).
    New: wait for connectivity, then continue the same API call with a fresh retry budget."""
    import agent.network_wait as nw
    probes = iter([False, False, True])
    worker_env.setattr(nw, "network_available", lambda *a, **k: next(probes))
    worker_env.setattr(nw.time, "sleep", lambda s: None)
    agent = _Agent()
    retry = TurnRetryState(primary_recovery_attempted=True, auto_recovery_cycles_used=5)
    verdict = _settle(agent, APIConnectionError(), retry)
    assert verdict.action == "continue"
    assert verdict.retry_count == 0
    assert any("Network unavailable" in s for s in agent.statuses)


def test_offline_worker_budget_spent_tags_network_unavailable(worker_env):
    import agent.network_wait as nw
    worker_env.setenv("HERMES_KANBAN_NETWORK_WAIT_SECONDS", "0")
    worker_env.setattr(nw, "network_available", lambda *a, **k: False)
    agent = _Agent()
    retry = TurnRetryState(primary_recovery_attempted=True, auto_recovery_cycles_used=5)
    verdict = _settle(agent, APIConnectionError(), retry)
    assert verdict.action == "return"
    assert verdict.result["failure_reason"] == "network_unavailable"


def test_wait_for_network_is_bounded_exponential():
    from agent.network_wait import wait_for_network
    t = {"now": 0.0}
    sleeps = []

    def _sleep(s):
        sleeps.append(s)
        t["now"] += s
    ok = wait_for_network(budget_s=200, probe=lambda: False, sleep=_sleep, clock=lambda: t["now"])
    assert ok is False
    assert sleeps[:5] == [5, 10, 20, 40, 60]
    assert sum(sleeps) <= 200


def test_is_network_error_matches_real_offline_texts():
    from agent.network_wait import is_network_error
    assert is_network_error(APIConnectionError())
    assert is_network_error(Exception(OFFLINE_MSG))
    assert is_network_error(OSError("[Errno 51] Network is unreachable"))
    assert is_network_error(Exception("nodename nor servname provided, or not known"))
    assert not is_network_error(Exception("HTTP 401 Incorrect API key"))


# ---------------------------------------------------------------- stop-gate (runs 6693/6695)
def test_stop_gate_does_not_burn_nudges_while_offline(worker_env):
    """Runs 6693/6695: nudge #2 at 20:46:22 went to an offline pi-acp, which answered with an
    empty "Retrying…" text; the budget was spent and the worker exited rc=0. New: wait for the
    network, refund the offline-burned nudges, then nudge once connectivity is back."""
    import agent.network_wait as nw
    from agent.turn_stop_gates import _kanban_stop_nudge
    probes = iter([False, False, True])
    worker_env.setattr(nw, "network_available", lambda *a, **k: next(probes))
    worker_env.setattr(nw.time, "sleep", lambda s: None)
    agent = SimpleNamespace(_kanban_stop_nudges=2, _interrupt_requested=False,
                            _emit_diagnostic_status=lambda s: None)
    nudge = _kanban_stop_nudge(agent, [{"role": "assistant", "content": "Retrying (attempt 3/3)..."}])
    assert nudge is not None and "kanban_complete" in nudge
    assert agent._kanban_stop_nudges == 0


# ---------------------------------------------------------------- exit guard (all three runs)
def _board(tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect(tmp_path / "kb.db")
    tid = kb.create_task(conn, title="net32", assignee="software-engineer")
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    run_id = conn.execute("SELECT current_run_id FROM tasks WHERE id=?", (tid,)).fetchone()[0]
    return kb, conn, tid, run_id


@pytest.mark.parametrize("result,rc", [
    ({"final_response": "Retrying (attempt 3/3, waiting 8s)...Retry finished, resuming."}, 0),  # 6693/6695
    ({"failed": True, "failure_reason": "network_unavailable", "error": OFFLINE_MSG}, 75),        # 6801
])
def test_worker_never_exits_running_without_board_call_offline(worker_env, tmp_path, result, rc):
    from hermes_cli.kanban_worker_network import ensure_terminal_board_call
    kb, conn, tid, run_id = _board(tmp_path)
    worker_env.setenv("HERMES_KANBAN_TASK", tid)
    worker_env.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    code = ensure_terminal_board_call(result, rc, conn=conn, network_ok=lambda: False)
    task = kb.get_task(conn, tid)
    assert task.status == "blocked"
    assert code == 0
    ev = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='blocked'", (tid,)).fetchone()
    assert ev and "network_unavailable" in ev[0]


def test_exit_guard_noop_when_handoff_already_landed(worker_env, tmp_path):
    from hermes_cli.kanban_worker_network import ensure_terminal_board_call
    kb, conn, tid, run_id = _board(tmp_path)
    worker_env.setenv("HERMES_KANBAN_TASK", tid)
    worker_env.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    assert kb.complete_task(conn, tid, summary="done")
    assert ensure_terminal_board_call({"final_response": "ok"}, 0, conn=conn, network_ok=lambda: True) == 0
    assert kb.get_task(conn, tid).status == "done"


def test_cli_exit_path_invokes_guard(worker_env):
    """Both single-query exit paths route through the guard; network_unavailable is transient."""
    import cli
    from hermes_cli import cli_single_query as csq
    from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE
    assert cli._single_query_exit_code({"failed": True, "failure_reason": "network_unavailable"}) \
        == KANBAN_RATE_LIMIT_EXIT_CODE
    src = open(csq.__file__).read()
    assert src.count("_guard_kanban_exit(") >= 3
