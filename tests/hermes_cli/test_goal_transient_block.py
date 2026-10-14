"""Recorded-field row 34 replay, not a live model-output claim."""
import json
from pathlib import Path

import pytest
from hermes_cli import goals
from hermes_state import SessionDB

INCIDENT = json.loads((Path(__file__).parents[1] / "fixtures/goals/transient_incident.json").read_text())


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    yield db
    db.close()


def reply(monkeypatch, payload):
    monkeypatch.setattr(goals, "_call_goal_judge_llm", lambda *a, **kw: json.dumps(payload))


@pytest.mark.parametrize("source", ["user", "agent"])
def test_incident_transient_continues_under_shared_guards(db, monkeypatch, source):
    reply(monkeypatch, INCIDENT["TRANSIENT"])
    manager = goals.GoalManager("s")
    setter = manager.set if source == "user" else manager.set_agent_goal
    setter(INCIDENT["INCIDENT_GOAL"])
    decision = manager.evaluate_after_turn(INCIDENT["INCIDENT_RESPONSE"])
    assert decision["should_continue"]
    assert decision["status"] == "active"
    assert "unachievable" not in decision["message"]
    assert goals.TRANSIENT_BLOCK_SUFFIX in decision["continuation_prompt"]
    saved = goals.load_goal("s")
    assert saved.last_verdict == "blocked"
    assert saved.last_reason == INCIDENT["TRANSIENT"]["reason"]
    assert saved.turns_used == 1


@pytest.mark.parametrize("blocker,category,decision", [
    ("human", "money", "approve purchase"), ("human", "auth", "log in"),
    ("human", "irreversible", "delete data"), ("unknown", "", ""),
    (None, "", ""), ("transient", "money", "approve purchase"),
    ("transient", "auth", "log in"), ("transient", "irreversible", "delete data"),
    ("transient", "unknown", ""), ("transient", "", "need human consent"),
])
def test_safety_or_unknown_block_still_pauses(db, monkeypatch, blocker, category, decision):
    payload = dict(INCIDENT["TRANSIENT"], human_category=category, human_decision=decision)
    if blocker is None:
        payload.pop("blocker")
    else:
        payload["blocker"] = blocker
    reply(monkeypatch, payload)
    manager = goals.GoalManager("s")
    manager.set("work")
    result = manager.evaluate_after_turn("blocked")
    assert result["status"] == "paused"
    assert "unachievable" in result["message"]
    assert result["continuation_prompt"] is None

@pytest.mark.parametrize("verdict", ["continue", "blocked"])
def test_sixth_identical_response_stalls_but_changed_response_resets(db, monkeypatch, verdict):
    reply(monkeypatch, dict(INCIDENT["TRANSIENT"], verdict=verdict))
    manager = goals.GoalManager("s")
    manager.set("work", max_turns=50)
    for turn in range(6):
        # Reload each turn: progress accounting must be durable, not just in memory.
        manager = goals.GoalManager("s")
        result = manager.evaluate_after_turn("same response")
        assert result["should_continue"] is (turn < 5)
    assert result["verdict"] == "stalled"
    assert "stalled" in manager.state.paused_reason
    manager.resume()
    assert manager.evaluate_after_turn("same response")["should_continue"]
    for _ in range(3):
        manager.evaluate_after_turn("same response")
    manager.evaluate_after_turn("different work completed")
    assert manager.state.consecutive_stalled == 0
    assert manager.evaluate_after_turn("different work completed")["should_continue"]


@pytest.mark.parametrize("source", ["user", "agent"])
@pytest.mark.parametrize("verdict", ["continue", "blocked"])
def test_varying_responses_still_hit_default_cap(db, monkeypatch, source, verdict):
    reply(monkeypatch, dict(INCIDENT["TRANSIENT"], verdict=verdict))
    manager = goals.GoalManager("s")
    setter = manager.set if source == "user" else manager.set_agent_goal
    state = setter("work")
    for turn in range(state.max_turns):
        result = manager.evaluate_after_turn(f"Completed distinct item {turn}")
        assert result["should_continue"] is (turn < state.max_turns - 1)
    assert result["status"] == "paused"
    assert "budget exhausted" in manager.state.paused_reason
    assert result["continuation_prompt"] is None


def test_judge_request_explains_blocker_metadata(db, monkeypatch):
    def call(_client, system, user, timeout):
        assert "human_category" in system
        assert "transient" in system
        assert "money" in system and "irreversible" in system
        return json.dumps(INCIDENT["TRANSIENT"])
    monkeypatch.setattr(goals, "_call_goal_judge_llm", call)
    result = goals.judge_goal("work", "needs clarification")
    assert result[0] == "blocked"
    assert result[3]["blocker"] == "transient"

@pytest.mark.parametrize("threshold,stop_after", [(2, 3), (0, 6), (-1, 6), ("false", 6), (True, 6), (None, 6)])
def test_stall_config_read_through_real_yaml(db, monkeypatch, tmp_path, threshold, stop_after):
    from hermes_cli.config import atomic_config_write
    atomic_config_write(tmp_path / "config.yaml", {"goals": {"max_stalled_turns": threshold}})
    reply(monkeypatch, INCIDENT["TRANSIENT"])
    manager = goals.GoalManager("s")
    manager.set("work", max_turns=50)
    for turn in range(stop_after):
        result = manager.evaluate_after_turn("unchanged")
        assert result["should_continue"] is (turn < stop_after - 1)
    assert result["verdict"] == "stalled"


@pytest.mark.parametrize("mutation", ["clear", "pause", "replace"])
def test_stall_pause_cannot_overwrite_concurrent_boundary(db, monkeypatch, mutation):
    manager = goals.GoalManager("s")
    manager.set_agent_goal("G1")
    reply(monkeypatch, INCIDENT["TRANSIENT"])
    for _ in range(5):
        assert manager.evaluate_after_turn("same")["should_continue"]
    def call(*args, **kwargs):
        other = goals.GoalManager("s")
        if mutation == "replace":
            other.set_agent_goal("G2", replace=True)
        else:
            getattr(other, mutation)()
        return json.dumps(INCIDENT["TRANSIENT"])
    monkeypatch.setattr(goals, "_call_goal_judge_llm", call)
    result = manager.evaluate_after_turn("same")
    assert result["verdict"] == "stale"
    assert result["message"] == ""
    assert not result["should_continue"]
    current = goals.load_goal("s")
    assert current.goal == ("G2" if mutation == "replace" else "G1")
    assert current.status == {"clear": "cleared", "pause": "paused", "replace": "active"}[mutation]

def test_blocked_metadata_keeps_human_decision_text(db, monkeypatch):
    payload = {"verdict": "blocked", "reason": "Needs consent", "blocker": "human",
               "human_category": "money", "human_decision": "Approve USD 100 for Project X"}
    reply(monkeypatch, payload)
    result = goals.judge_goal("work", "need approval")
    assert result[3]["human_decision"] == payload["human_decision"]
