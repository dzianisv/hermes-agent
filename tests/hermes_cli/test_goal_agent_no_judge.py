"""Agent-set goals (source=agent) are never LLM-judged per turn; user /goal still is."""
from __future__ import annotations

import pytest

from hermes_cli import goals

EVIDENCE = "pytest tests/x.py -> 12 passed; PR https://github.com/o/r/pull/1 merged at abc123"


@pytest.fixture
def mgr_factory(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = []

    def fake_judge(*a, **k):
        calls.append(a)
        return "continue", "judged", False, None, False

    monkeypatch.setattr(goals, "judge_goal", fake_judge)

    def make(source, sid="s1", max_turns=3):
        m = goals.GoalManager(session_id=sid)
        if source == "agent":
            m.set_agent_goal("ship the thing", max_turns=max_turns)
        else:
            m.set("ship the thing", max_turns=max_turns)
        return m

    make.calls = calls
    return make


def test_agent_goal_never_judged_per_turn_and_has_no_turn_budget(mgr_factory):
    """Live 2026-10-09: '23/20 turns used' on an agent goal with work left. No budget on agent goals."""
    m = mgr_factory("agent", max_turns=2)
    out = [m.evaluate_after_turn(f"t{i}") for i in range(5)]
    assert mgr_factory.calls == []
    assert all(d["status"] == "active" and d["should_continue"] for d in out)
    assert m.state.turns_used == 5 and not m.is_waiting()


def test_user_goal_still_judged(mgr_factory):
    m = mgr_factory("user", sid="s2")
    m.evaluate_after_turn("some work")
    m.complete_with_evidence(EVIDENCE)
    assert len(mgr_factory.calls) == 2
