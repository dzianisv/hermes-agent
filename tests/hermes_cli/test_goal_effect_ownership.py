"""Row 31: a slow evaluation cannot resurrect a cleared or paused goal."""
import pytest
from hermes_cli import goals
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    yield db
    db.close()


@pytest.mark.parametrize("mutation", ["clear", "pause"])
@pytest.mark.parametrize("result", [
    ("continue", "more", False, None, False),
    ("done", "finished", False, None, False),
    ("blocked", "impossible", False, None, False),
    ("wait", "wait", False, {"seconds": 60}, False),
    ("wait", "wait", False, {"session_id": "worker"}, False),
    ("wait", "wait", False, {"pid": 99}, False),
    ("continue", "bad parse", True, None, False),
    ("continue", "bad transport", False, None, True),
])
def test_concurrent_boundary_suppresses_all_effects(db, monkeypatch, mutation, result):
    manager = goals.GoalManager("session")
    state = manager.set("G1", max_turns=1)
    state.consecutive_parse_failures = 50
    state.consecutive_transport_failures = 50
    manager._save()
    monkeypatch.setattr(goals, "_pid_alive", lambda pid: True)
    def judge(*args, **kwargs):
        getattr(goals.GoalManager("session"), mutation)()
        return result
    monkeypatch.setattr(goals, "judge_goal", judge)
    decision = manager.evaluate_after_turn("work")
    expected = "cleared" if mutation == "clear" else "paused"
    assert goals.load_goal("session").status == expected
    assert manager.state.status == expected
    assert decision["verdict"] == "stale"
    assert decision["message"] == ""
    assert decision["should_continue"] is False
    assert decision["continuation_prompt"] is None
    assert manager.next_continuation_prompt() is None

def test_agent_create_owns_empty_slot_and_defaults_user_source(db):
    user = goals.GoalManager("user").set("User objective")
    assert getattr(user, "source", None) == "user"
    manager = goals.GoalManager("session", default_max_turns=4)
    assert hasattr(manager, "set_agent_goal"), "main lacks agent-owned creation"
    created = manager.set_agent_goal("Agent objective", contract=goals.GoalContract(outcome="artifact"))
    assert created.source == "agent"
    assert goals.load_goal("session").source == "agent"
    assert created.max_turns == 4

@pytest.mark.parametrize("status", ["active", "paused", "done", "cleared"])
@pytest.mark.parametrize("source", ["user", "agent"])
@pytest.mark.parametrize("replace", [False, True])
def test_replacement_slot_ownership_and_archive(db, status, source, replace):
    manager = goals.GoalManager("session")
    old = manager.set("G1") if source == "user" else manager.set_agent_goal("G1")
    old.status = status
    goals.save_goal("session", old)
    occupied = status in {"active", "paused"}
    if occupied and (source == "user" or not replace):
        with pytest.raises(ValueError, match="user|replace"):
            manager.set_agent_goal("G2", replace=replace)
        assert goals.load_goal("session").goal == "G1"
        assert db.list_meta_prefix("goal_archive:") == []
    else:
        new = manager.set_agent_goal("G2", replace=replace)
        assert new.source == "agent"
        assert goals.load_goal("session").goal == "G2"
        archives = db.list_meta_prefix("goal_archive:session:")
        assert len(archives) == int(occupied)
        if occupied:
            assert goals.GoalState.from_json(archives[0][1]).goal == "G1"

@pytest.mark.parametrize("window", ["judge", "gate"])
@pytest.mark.parametrize("mutation", ["clear", "pause", "replace", "none"])
def test_snapshot_precedes_gates_and_normal_decisions_survive(db, monkeypatch, window, mutation):
    manager = goals.GoalManager("session")
    manager.set_agent_goal("G1")
    def mutate():
        other = goals.GoalManager("session")
        if mutation == "replace":
            other.set_agent_goal("G2", replace=True)
        elif mutation != "none":
            getattr(other, mutation)()
    def judge(*args, **kwargs):
        if window == "judge":
            mutate()
        return "continue", "more", False, None, False
    monkeypatch.setattr(goals, "judge_goal", judge)
    if window == "gate":
        manager.state.gates = [goals.GoalGate(command="ignored", max_retries=2)]
        manager._save()
        def gate(g):
            mutate()
            return False, 1, "not ready"
        monkeypatch.setattr(goals, "run_gate", gate)
    result = manager.evaluate_after_turn("work")
    current = goals.load_goal("session")
    if mutation == "none":
        assert result["should_continue"]
        assert result["message"]
        assert result["continuation_prompt"]
        assert current.turns_used == 1
    else:
        assert result["message"] == ""
        assert not result["should_continue"]
        assert result["verdict"] == "stale"
        assert manager.state.to_json() == current.to_json()
        if mutation == "replace":
            assert current.goal == "G2"
            assert current.turns_used == 0
            assert db.list_meta_prefix("goal_archive:session:")


def test_documented_identical_clock_replacement_can_lose_changed_contract(db, monkeypatch):
    # B11 is a LIMITATION, not proof of detection: even a different contract is
    # overwritten if all four approved identity fields collide.
    monkeypatch.setattr(goals.time, "time", lambda: 1234.0)
    manager = goals.GoalManager("session")
    manager.set_agent_goal("same", contract=goals.GoalContract(outcome="old contract"))
    def judge(*args, **kwargs):
        goals.GoalManager("session").set_agent_goal("same", replace=True,
            contract=goals.GoalContract(outcome="different contract"))
        return "continue", "more", False, None, False
    monkeypatch.setattr(goals, "judge_goal", judge)
    result = manager.evaluate_after_turn("work")
    assert result["should_continue"]
    assert goals.load_goal("session").contract.outcome == "old contract"


def test_same_second_archive_overwrites_and_transaction_rolls_back(db, monkeypatch):
    monkeypatch.setattr(goals.time, "time", lambda: 1234.0)
    manager = goals.GoalManager("session")
    manager.set_agent_goal("G1")
    manager.set_agent_goal("G2", replace=True)
    manager.set_agent_goal("G3", replace=True)
    rows = db.list_meta_prefix("goal_archive:session:")
    assert len(rows) == 1
    assert goals.GoalState.from_json(rows[0][1]).goal == "G2"
    original = db.set_meta
    def fail_current(key, value, **kwargs):
        if key == "goal:session":
            raise RuntimeError("injected current-row write failure")
        return original(key, value, **kwargs)
    monkeypatch.setattr(db, "set_meta", fail_current)
    with pytest.raises(RuntimeError):
        manager.set_agent_goal("G4", replace=True)
    assert goals.load_goal("session").goal == "G3"
    assert db.list_meta_prefix("goal_archive:session:") == rows

def test_captured_cross_version_json_preserves_main_fields(db):
    import json
    from pathlib import Path
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/goals/installed_serialization.json").read_text())
    for record in fixture["installed_records"]:
        state = goals.GoalState.from_json(json.dumps(record))
        roundtrip = goals.GoalState.from_json(state.to_json())
        assert roundtrip.source == "agent"
        assert roundtrip.goal == record["goal"]
        assert roundtrip.max_turns == record["max_turns"]
        assert roundtrip.turns_used == record["turns_used"]
        assert roundtrip.contract.outcome == record["contract"]["outcome"]
        assert roundtrip.status == {"waiting": "paused", "cancelled": "cleared"}.get(record["status"], record["status"])
    main_record = goals.GoalState.from_json(json.dumps(fixture["main_record"]))
    installed = fixture["installed_readback"]
    assert main_record.source == installed["source"] == "agent"
    assert main_record.max_turns == installed["max_turns"]
    assert main_record.contract.outcome == installed["contract"]["outcome"]
