"""Real SQLite races: delayed effects may not overwrite a newer lifecycle decision."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from hermes_cli import goals
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    yield db
    db.close()


@pytest.mark.parametrize("change", ["clear", "replace", "pause", "done", "migrate", "end", "criteria"])
@pytest.mark.parametrize("audit", [False, True])
def test_delayed_effect_loses_to_current_owner(db, monkeypatch, change, audit):
    db.create_session("parent", source="telegram")
    manager = goals.GoalManager("parent")
    manager.set_agent_goal("Ship the verified artifact")
    entered, finish = threading.Event(), threading.Event()

    def judge(*args, **kwargs):
        entered.set()
        assert finish.wait(10)
        return "done" if audit else "continue", "old evidence", False, None, False

    monkeypatch.setattr(goals, "judge_goal", judge)
    with ThreadPoolExecutor() as executor:
        future = executor.submit(manager.complete_with_evidence if audit else manager.evaluate_after_turn,
                                 "The artifact exists at the verified URL")
        assert entered.wait(10)
        owner = goals.GoalManager("parent")
        try:
            if change == "clear":
                owner.clear()
            elif change == "replace":
                owner.clear()
                owner.set("Different user objective")
            elif change == "pause":
                owner.pause()
            elif change == "done":
                owner.mark_done("audited by another boundary")
            elif change == "migrate":
                assert goals.migrate_goal_to_session("parent", "child")
            elif change == "end":
                db.end_session("parent", end_reason="user_exit")
            else:
                owner.add_subgoal("New requirement must also be verified")
            expected = db.get_meta("goal:parent")
        finally:
            finish.set()
        result = future.result(timeout=10)
    assert result["verdict"] == "stale"
    assert not result["should_continue"] and not result["message"]
    assert db.get_meta("goal:parent") == expected
    assert goals.load_goal("parent").goal
    if change == "migrate":
        assert goals.load_goal("parent").status == "cancelled"
        assert goals.load_goal("child").status == "active"


def test_commands_reduce_current_row_and_audit_includes_all_requirements(db, monkeypatch):
    a, b = goals.GoalManager("one"), goals.GoalManager("one")
    a.set_agent_goal("Ship the verified artifact")
    b.add_subgoal("Verify the customer-facing URL")
    a.link_tasks(["t_abcdef"])
    assert goals.load_goal("one").subgoals == ["Verify the customer-facing URL"]
    assert goals.load_goal("one").linked_task_ids == ["t_abcdef"]
    seen = []
    monkeypatch.setattr(goals, "_call_goal_judge_llm",
                        lambda _call, system, user, timeout: seen.append(user) or json.dumps({"verdict": "done"}))
    assert b.complete_with_evidence("Verified the customer-facing URL")["verdict"] == "done"
    assert "Verify the customer-facing URL" in seen[0]
    assert goals.load_goal("one").status == "done"
    assert a.resume() is None  # terminal goals cannot be resurrected by stale controls
    a.set("Verify failing quality gate")
    a.add_gate("exit 1")
    assert a.complete_with_evidence("Claimed success without passing the gate")["verdict"] == "gate_failed"
    assert goals.load_goal("one").status == "active"
    assert len(seen) == 1
