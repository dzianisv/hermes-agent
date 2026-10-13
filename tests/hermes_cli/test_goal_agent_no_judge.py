"""Agent-set goals (source=agent) are never LLM-judged; user /goal still is."""
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


def test_agent_goal_completes_without_judge(mgr_factory):
    m = mgr_factory("agent")
    d = m.complete_with_evidence(EVIDENCE)
    assert d["status"] == "done" and mgr_factory.calls == []
    assert m.state.evidence == EVIDENCE


def test_agent_goal_short_evidence_refused(mgr_factory):
    m = mgr_factory("agent")
    assert m.complete_with_evidence("done").get("status") == "active"
    assert mgr_factory.calls == []


def test_agent_goal_never_judged_parked_or_paused_and_stops_at_budget(mgr_factory):
    m = mgr_factory("agent", max_turns=3)
    out = [m.evaluate_after_turn("same reply every turn") for _ in range(5)]
    assert mgr_factory.calls == []
    assert all(d["status"] == "active" for d in out)
    assert [d["should_continue"] for d in out] == [True, True, False, False, False]
    assert m.state.status == "active" and not m.is_waiting()


def test_user_goal_still_judged(mgr_factory):
    m = mgr_factory("user", sid="s2")
    m.evaluate_after_turn("some work")
    m.complete_with_evidence(EVIDENCE)
    assert len(mgr_factory.calls) == 2
