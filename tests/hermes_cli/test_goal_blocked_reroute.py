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


def test_repeated_blocked_verdicts_never_park_an_agent_goal(goals_mod, monkeypatch):
    # Muse semantics: no park-on-blocked. Only complete / user pause|clear / max_turns end it.
    _judge(goals_mod, monkeypatch, {"verdict": "blocked", "reason": "tool error"}, [])
    mgr = _agent_goal(goals_mod)
    for i in range(6):
        d = mgr.evaluate_after_turn(f"stuck {i}")
        assert d["should_continue"] is True and d["status"] == "active", d
        assert "Continue working toward the active goal" in d["continuation_prompt"]
    assert goals_mod.GoalManager(session_id="s1").state.status == "active"


def test_progress_resets_blocked_budget(goals_mod, monkeypatch):
    mgr = _agent_goal(goals_mod)
    _judge(goals_mod, monkeypatch, APPROVAL_TIMEOUT, [])
    mgr.evaluate_after_turn("blocked 1")
    mgr.evaluate_after_turn("blocked 2")
    _judge(goals_mod, monkeypatch, {"verdict": "continue", "reason": "progress"}, [])
    mgr.evaluate_after_turn("made progress")
    assert mgr.state.consecutive_blocked == 0


def test_user_goal_blocked_still_pauses(goals_mod, monkeypatch):
    _judge(goals_mod, monkeypatch, APPROVAL_TIMEOUT, [])
    mgr = goals_mod.GoalManager(session_id="u1")
    mgr.set("user goal")
    d = mgr.evaluate_after_turn("blocked")
    assert d["status"] == "waiting"
