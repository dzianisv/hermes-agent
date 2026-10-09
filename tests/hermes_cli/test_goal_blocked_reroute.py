"""Agent-declared goals must not park on transient blockers (approval timeout etc.).

Real GoalManager + real SessionDB state in a temp HERMES_HOME. Only the judge's LLM call
(_call_goal_judge_llm) is replaced by a fixed local function, so judge_goal's prompt + parsing run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def goals_mod(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals
    goals._DB_CACHE.clear()
    yield goals
    goals._DB_CACHE.clear()


def _judge(goals, monkeypatch, reply: dict, seen: list):
    def fake(call_llm, system_prompt, user_prompt, timeout):
        seen.append(system_prompt)
        return json.dumps(reply)
    monkeypatch.setattr(goals, "_call_goal_judge_llm", fake)


APPROVAL_TIMEOUT = {"verdict": "blocked", "blocker": "transient",
                    "reason": "AGENTS.md patch hit 'approval prompt timed out'; agent asked user to type retry"}


def _agent_goal(goals, sid="s1"):
    mgr = goals.GoalManager(session_id=sid)
    assert mgr.set_agent_goal("ship the EM rule into the repo") is not None
    return mgr


def test_approval_timeout_reroutes_instead_of_parking(goals_mod, monkeypatch):
    seen: list = []
    _judge(goals_mod, monkeypatch, APPROVAL_TIMEOUT, seen)
    mgr = _agent_goal(goals_mod)
    d = mgr.evaluate_after_turn("approval prompt timed out. Type retry to try again.")
    assert d["should_continue"] is True
    assert d["status"] == "active"
    assert "non-protected file" in d["continuation_prompt"]
    assert "approval prompt timed out" in d["continuation_prompt"]
    # persisted across a fresh manager (real state_meta)
    again = goals_mod.GoalManager(session_id="s1")
    assert again.state.status == "active" and again.state.consecutive_blocked == 1
    assert "human_only" in seen[0]  # judge prompt asks for the category


def test_unknown_category_is_transient_within_budget_then_parks_with_question(goals_mod, monkeypatch):
    _judge(goals_mod, monkeypatch, {"verdict": "blocked", "reason": "tool error"}, [])
    mgr = _agent_goal(goals_mod)
    for i in range(goals_mod.DEFAULT_MAX_CONSECUTIVE_TRANSIENT_BLOCKS):
        d = mgr.evaluate_after_turn(f"stuck {i}")
        assert d["should_continue"] is True, d
    d = mgr.evaluate_after_turn("stuck again")
    assert d["status"] == "paused" and d["should_continue"] is False
    assert "human decision needed" in d["message"] and "how should it proceed?" in d["message"]


def test_progress_resets_blocked_budget(goals_mod, monkeypatch):
    mgr = _agent_goal(goals_mod)
    _judge(goals_mod, monkeypatch, APPROVAL_TIMEOUT, [])
    mgr.evaluate_after_turn("blocked 1")
    mgr.evaluate_after_turn("blocked 2")
    _judge(goals_mod, monkeypatch, {"verdict": "continue", "reason": "progress"}, [])
    mgr.evaluate_after_turn("made progress")
    assert mgr.state.consecutive_blocked == 0


def test_human_only_parks_immediately_with_exact_question(goals_mod, monkeypatch):
    q = "Approve spending $240/month on a second GPU node?"
    _judge(goals_mod, monkeypatch, {"verdict": "blocked", "blocker": "human_only",
                                     "human_decision": q, "reason": "needs budget approval"}, [])
    mgr = _agent_goal(goals_mod)
    d = mgr.evaluate_after_turn("I need approval to buy a node")
    assert d["status"] == "paused" and d["should_continue"] is False
    assert q in d["message"]


def test_user_goal_blocked_still_pauses(goals_mod, monkeypatch):
    _judge(goals_mod, monkeypatch, APPROVAL_TIMEOUT, [])
    mgr = goals_mod.GoalManager(session_id="u1")
    mgr.set("user goal")
    d = mgr.evaluate_after_turn("blocked")
    assert d["status"] == "paused"
