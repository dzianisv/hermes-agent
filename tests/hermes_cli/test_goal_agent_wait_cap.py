"""Agent-declared goals treat a ``wait`` as a short sleep, not a stop.

Muse semantics: waiting on CI / a review / a background process / a subagent is itself a work
item (poll it, review the diff, prepare the next requirement). So for ``source="agent"`` goals any
wait barrier is capped at ``goals.max_wait_seconds`` (default 300) and the continuation that
resumes after a barrier lifted opens with the "wait has elapsed" kick. User-typed goals keep
their judge-chosen wait and their verbatim continuation template.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import goals


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    goals._DB_CACHE.clear()
    goals._get_session_db()
    yield home
    reset_hermes_home_override(token)
    goals._DB_CACHE.clear()


def _agent_goal(sid: str) -> goals.GoalManager:
    mgr = goals.GoalManager(sid)
    assert mgr.set_agent_goal("ship the PR") is not None
    return mgr


def _judge_wait(seconds: int, reason: str):
    return patch("hermes_cli.goals.judge_goal", return_value=("wait", reason, False, {"seconds": seconds}, False))


class TestAgentGoalWaitCap:
    def test_cap_reads_goals_max_wait_seconds(self, hermes_home):
        mgr = _agent_goal("agent-wait-cfg")
        with patch("hermes_cli.config.load_config", return_value={"goals": {"max_wait_seconds": 42}}):
            before = time.time()
            mgr.wait_for_seconds(900, reason="review pending")
        assert mgr.state.waiting_until - before <= 43

    def test_shorter_waits_are_not_lengthened(self, hermes_home):
        mgr = _agent_goal("agent-wait-short")
        before = time.time()
        mgr.wait_for_seconds(30, reason="backoff")
        assert mgr.state.waiting_until - before <= 31

    def test_pid_barrier_on_agent_goal_lifts_after_cap_even_if_pid_lives(self, hermes_home):
        import os

        mgr = _agent_goal("agent-wait-pid")
        mgr.wait_on(os.getpid(), reason="subagent running")
        assert mgr.is_waiting() is True
        mgr.state.waiting_since = time.time() - goals.DEFAULT_MAX_WAIT_SECONDS - 5
        mgr._save()
        assert mgr.is_waiting() is False
        assert mgr.state.waiting_on_pid is None

    def test_user_goal_wait_is_unchanged(self, hermes_home):
        mgr = goals.GoalManager("user-wait")
        mgr.set("ship the PR")
        before = time.time()
        with _judge_wait(900, "waiting on CI"):
            decision = mgr.evaluate_after_turn("pushed; CI running")
        assert mgr.state.waiting_until - before > 890
        assert decision["message"] == ""  # wait verdicts never speak, user goals included


class TestWaitElapsedContinuation:
    def test_agent_goal_resumes_with_the_wait_elapsed_kick_once(self, hermes_home):
        mgr = _agent_goal("agent-wait-kick")
        mgr.wait_for_seconds(60, reason="waiting on CI")
        mgr.state.waiting_until = time.time() - 1
        mgr._save()
        assert mgr.is_waiting() is False                      # barrier lifted by the check itself

        prompt = goals.GoalManager("agent-wait-kick").next_continuation_prompt()   # fresh load: persisted
        assert prompt.startswith("The wait has elapsed; check what you were waiting on and keep going.\n")
        assert "Continue working toward the active goal" in prompt
        # One-shot: the next continuation is the plain kick again.
        assert not goals.GoalManager("agent-wait-kick").next_continuation_prompt().startswith("The wait has elapsed")

    def test_explicit_unwait_does_not_add_the_kick(self, hermes_home):
        mgr = _agent_goal("agent-unwait")
        mgr.wait_for_seconds(60, reason="x")
        mgr.stop_waiting()
        assert not mgr.next_continuation_prompt().startswith("The wait has elapsed")

    def test_user_goal_continuation_template_is_verbatim(self, hermes_home):
        mgr = goals.GoalManager("user-wait-kick")
        mgr.set("ship the PR")
        mgr.wait_for_seconds(60, reason="x")
        mgr.state.waiting_until = time.time() - 1
        mgr._save()
        assert mgr.is_waiting() is False
        prompt = mgr.next_continuation_prompt()
        # The gateway identifies queued continuations by this first line; it must not move.
        assert prompt.startswith("[Continuing toward your standing goal]\nGoal:")
