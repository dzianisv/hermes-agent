"""A BLOCKED verdict the judge classifies as ``transient`` (agent-recoverable) must keep a user
goal running with a corrective nudge, bounded by the goal's turn budget; human-only blocks
(money / credentials / irreversible) still pause. Agent goals are unaffected (never judged).

Behaviour contracts through the real GoalManager + judge_goal parsing in a temp HERMES_HOME; only
the judge LLM call is replaced. The transient case replays the verdict recorded in agent.log
(2026-10-14 09:19:25, session 20261014_070313_92d520) that paused Den's goal.
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


def _judge(goals, monkeypatch, reply: dict):
    monkeypatch.setattr(goals, "_call_goal_judge_llm", lambda *a, **k: json.dumps(reply))


# Verbatim from agent.log line ~34999 (reason is the logged prefix; log truncated the rest).
LOGGED_TRANSIENT = {
    "verdict": "blocked", "blocker": "transient", "human_decision": "", "human_category": "",
    "reason": "The agent stopped to ask for clarification instead of proceeding to resolve the "
              "open issue and deliver it to main on its own.",
}
GOAL = "Resolve the open Notion issue rows and deliver each fix to main"
REPLY = "I found the issue. Should I go ahead and open the PR and merge it to main?"


def test_logged_transient_verdict_continues_user_goal(goals_mod, monkeypatch):
    _judge(goals_mod, monkeypatch, LOGGED_TRANSIENT)
    mgr = goals_mod.GoalManager(session_id="t1")
    mgr.set(GOAL, max_turns=5)
    d = mgr.evaluate_after_turn(REPLY)
    assert d["status"] == "active" and d["should_continue"] is True
    assert "paused" not in (d.get("message") or "").lower()
    assert "stopped to ask for clarification" in d["continuation_prompt"]
    assert goals_mod.GoalManager(session_id="t1").state.status == "active"


def test_transient_blocks_are_bounded_by_turn_budget(goals_mod, monkeypatch):
    _judge(goals_mod, monkeypatch, LOGGED_TRANSIENT)
    mgr = goals_mod.GoalManager(session_id="t2")
    mgr.set(GOAL, max_turns=3)
    decisions = [mgr.evaluate_after_turn(f"{REPLY} ({i})") for i in range(6)]
    stopped = [i for i, d in enumerate(decisions) if not d["should_continue"]]
    assert stopped and stopped[0] <= 3, decisions
    assert goals_mod.GoalManager(session_id="t2").state.status != "active"


@pytest.mark.parametrize("category", ["money", "credentials", "irreversible"])
def test_human_only_block_still_pauses_user_goal(goals_mod, monkeypatch, category):
    _judge(goals_mod, monkeypatch, {
        "verdict": "blocked", "blocker": "human_only", "human_category": category,
        "human_decision": "Approve the $200 charge / grant the token / delete prod data?",
        "reason": "Needs the user's authorization."})
    mgr = goals_mod.GoalManager(session_id=f"h-{category}")
    mgr.set(GOAL, max_turns=5)
    d = mgr.evaluate_after_turn("I need your approval to proceed.")
    assert d["status"] != "active" and d["should_continue"] is False
    assert goals_mod.GoalManager(session_id=f"h-{category}").state.status != "active"


def test_agent_goal_unaffected_by_transient_verdict(goals_mod, monkeypatch):
    _judge(goals_mod, monkeypatch, LOGGED_TRANSIENT)
    mgr = goals_mod.GoalManager(session_id="a1")
    assert mgr.set_agent_goal(GOAL, contract=goals_mod.GoalContract(outcome="fix on main")) is not None
    d = mgr.evaluate_after_turn(REPLY)
    assert d["status"] == "active" and d["should_continue"] is True
