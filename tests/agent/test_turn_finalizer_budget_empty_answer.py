"""Regression test for the at-ceiling blank-answer fallback (#37666734c66a).

A turn that hits the API-call ceiling can hand ``finalize_turn`` a falsy-but-not-None
answer: ``""`` from an empty content step, or whitespace. The budget fallback used to
fire only for ``final_response is None``, so the one bounded toolless summary request
was never made; the blank answer then fell into the ``pending_tool_result`` close,
which set ``failed=True`` and returned "No reply: ...", and
``cron.scheduler._final_response_from_result`` raised ``RuntimeError`` — the cycle's
findings were lost.

The fallback must fire for a blank answer at the ceiling, and must NOT fire for a turn
that already produced an answer (real text or streamed text) or for a turn that stopped
before the ceiling (the #55316/#54756 ``pending_tool_result`` close).
"""

from types import SimpleNamespace

import pytest

from agent.turn_failure_copy import is_max_iteration_handoff
from agent.turn_finalizer import finalize_turn

SUMMARY = "CI: 2 workflows red. SENTRY: 0 new issues. CWS: 4.3.1 published. Nothing else changed."
MAX_ITERATIONS = 250


class _StubBudget:
    def __init__(self, remaining):
        self.remaining = remaining
        self.used = MAX_ITERATIONS
        self.max_total = MAX_ITERATIONS


class _StubAgent:
    """Minimal agent surface ``finalize_turn`` reads, plus a counting summary request."""

    def __init__(self, *, max_iterations=MAX_ITERATIONS, budget_remaining=86, summary=SUMMARY):
        self.max_iterations = max_iterations
        self.iteration_budget = _StubBudget(budget_remaining)
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self.valid_tool_names = []
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self._current_streamed_assistant_text = ""
        self.persisted_messages = None
        self.summary_calls = 0
        self._summary = summary
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    # --- the bounded toolless summary request this test counts ----------
    def _handle_max_iterations(self, messages, api_call_count):
        self.summary_calls += 1
        return self._summary

    def _emit_diagnostic_status(self, *_args, **_kwargs):
        pass

    # --- harmless no-ops ------------------------------------------------
    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = [dict(m) for m in messages]

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    """No plugin hooks, no live kanban board mutation from a unit test."""
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(
        "agent.turn_finalizer.is_dispatcher_owned_worker_context", lambda: False
    )


def _tool_tail():
    """The incident shape: the loop stopped on a raw tool result, no assistant text."""
    return [
        {"role": "user", "content": "run the devops sweep"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "sweep output"},
    ]


def _finalize(
    agent,
    *,
    final_response,
    api_call_count=MAX_ITERATIONS,
    exit_reason="unknown",
    interrupted=False,
    failed=False,
    messages=None,
):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=interrupted,
        failed=failed,
        messages=_tool_tail() if messages is None else messages,
        conversation_history=[],
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="run the devops sweep",
        original_user_message="run the devops sweep",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
    )


@pytest.mark.parametrize("exit_reason", ["unknown", "budget_exhausted"])
@pytest.mark.parametrize("final_response", [None, "", "   "])
def test_blank_answer_at_ceiling_requests_exactly_one_summary(
    exit_reason, final_response
):
    agent = _StubAgent()
    messages = _tool_tail()

    result = _finalize(
        agent,
        final_response=final_response,
        exit_reason=exit_reason,
        messages=messages,
    )

    # Exactly ONE bounded toolless summary request — never a second one.
    assert agent.summary_calls == 1
    assert result["final_response"] == SUMMARY
    assert result["failed"] is False
    assert result["turn_exit_reason"] == f"max_iterations_reached({MAX_ITERATIONS}/{MAX_ITERATIONS})"
    # ``completed`` stays False at the ceiling by design: cron delivers this summary through
    # ``is_max_iteration_handoff`` (which requires ``completed is False``) instead of raising
    # ``RuntimeError``, so the findings reach the channel. Setting it True here would make that
    # predicate reject the turn and lose the resumable boundary.
    assert result["completed"] is False
    assert is_max_iteration_handoff(result) is True
    # The durable tail closes on an assistant row, not a raw tool result.
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == SUMMARY


def test_real_answer_at_ceiling_is_not_replaced_by_a_summary():
    agent = _StubAgent()
    answer = "All six probes clean; nothing to report."

    result = _finalize(agent, final_response=answer, exit_reason="unknown")

    assert agent.summary_calls == 0
    assert result["final_response"] == answer
    assert result["failed"] is False


def test_streamed_answer_at_ceiling_is_recovered_not_summarised():
    agent = _StubAgent()
    agent._current_streamed_assistant_text = "Streamed findings the user already saw."
    messages = _tool_tail()

    result = _finalize(
        agent, final_response="", exit_reason="unknown", messages=messages
    )

    # #95514 owns this recovery; the ceiling must not spend its last call on a summary
    # and discard text the user already received.
    assert agent.summary_calls == 0
    assert result["final_response"] == "Streamed findings the user already saw."
    assert result["failed"] is False


def test_blank_answer_below_ceiling_keeps_pending_tool_result_close():
    agent = _StubAgent()

    result = _finalize(
        agent, final_response="", api_call_count=12, exit_reason="unknown"
    )

    # A turn that stopped before the ceiling keeps the #55316/#54756 close.
    assert agent.summary_calls == 0
    assert result["turn_exit_reason"] == "pending_tool_result"
    assert result["failed"] is True
    assert result["final_response"].strip()


def test_interrupted_blank_answer_at_ceiling_is_not_summarised():
    agent = _StubAgent()

    result = _finalize(
        agent, final_response="", exit_reason="interrupted_by_user", interrupted=True
    )

    assert agent.summary_calls == 0
    assert result["interrupted"] is True


@pytest.mark.parametrize(
    "exit_reason",
    [
        "all_retries_exhausted_no_response",
        "provider_failure",
        "error_near_max_iterations",
    ],
)
def test_ineligible_exit_reason_never_requests_a_summary(exit_reason):
    agent = _StubAgent()

    _finalize(agent, final_response="", exit_reason=exit_reason)

    assert agent.summary_calls == 0
