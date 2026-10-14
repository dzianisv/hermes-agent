"""Opt-in goal surface: registry dispatch, real scoped storage and YAML."""
import importlib
import importlib.util
import json
from types import SimpleNamespace

import pytest
from gateway.session_context import scoped_current_session_id
from hermes_cli import goals
from hermes_cli.config import atomic_config_write
from hermes_state import SessionDB
from tools.registry import registry
import tools
from importlib.machinery import PathFinder
from toolsets import resolve_toolset


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    yield tmp_path
    db.close()


def load_tool():
    assert PathFinder.find_spec("tools.goal_tool", tools.__path__) is not None, "goal tool is absent on main"
    return importlib.import_module("tools.goal_tool")


@pytest.mark.parametrize("flag", [True, False, "false", None])
def test_strict_opt_in_schema_and_dispatch(home, flag, monkeypatch):
    atomic_config_write(home / "config.yaml", {"goals": {"agent_tool": flag, "max_turns": 3}})
    load_tool()
    monkeypatch.setattr("tools.registry._check_fn_cache", {})
    monkeypatch.setattr("tools.registry._check_fn_last_good", {})
    schemas = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    assert any(s["function"]["name"] == "goal" for s in schemas) is (flag is True)
    with scoped_current_session_id("tool-session"):
        result = json.loads(registry.dispatch("goal", {"action": "create", "objective": "Produce tested artifact", "acceptance_criteria": "Tests pass"}))
    if flag is True:
        assert result["created"] is True
        assert goals.load_goal("tool-session").source == "agent"
        assert goals.load_goal("tool-session").max_turns == 3
    else:
        assert "error" in result
        assert goals.load_goal("tool-session") is None

@pytest.mark.parametrize("sid", ["cron_job", ""])
def test_cron_and_missing_session_refused(home, sid):
    atomic_config_write(home / "config.yaml", {"goals": {"agent_tool": True}})
    load_tool()
    with scoped_current_session_id(sid):
        result = json.loads(registry.dispatch("goal", {"action": "create", "objective": "Do the work"}))
    assert "error" in result
    assert goals.load_goal(sid) is None


@pytest.mark.parametrize("status", ["active", "paused"])
def test_tool_user_protection_precedes_manager_write(home, monkeypatch, status):
    atomic_config_write(home / "config.yaml", {"goals": {"agent_tool": True}})
    load_tool()
    manager = goals.GoalManager("s")
    state = manager.set("user goal")
    state.status = status
    goals.save_goal("s", state)
    def forbidden(*args, **kwargs):
        pytest.fail("user goal reached manager write despite handler protection")
    monkeypatch.setattr(goals.GoalManager, "set_agent_goal", forbidden)
    with scoped_current_session_id("s"):
        result = json.loads(registry.dispatch("goal", {"action": "create", "objective": "Other work", "replace": True}))
    assert "user" in result["error"]

def test_guidance_only_with_goal_tool():
    from agent.system_prompt import _tool_guidance_block
    without = _tool_guidance_block(SimpleNamespace(valid_tool_names=set())) or ""
    with_goal = _tool_guidance_block(SimpleNamespace(valid_tool_names={"goal"})) or ""
    assert "replace=true" not in without
    assert "replace=true" in with_goal
    assert "/goal clear" in with_goal

def test_tool_to_manager_race_cannot_overwrite_user(home, monkeypatch):
    atomic_config_write(home / "config.yaml", {"goals": {"agent_tool": True}})
    load_tool()
    create = goals.GoalManager.set_agent_goal
    def race(manager, *args, **kwargs):
        goals.GoalManager("s").set("concurrent user goal")
        return create(manager, *args, **kwargs)
    monkeypatch.setattr(goals.GoalManager, "set_agent_goal", race)
    with scoped_current_session_id("s"):
        result = json.loads(registry.dispatch("goal", {"action": "create", "objective": "Agent work", "replace": True}))
    assert "user" in result["error"]
    assert goals.load_goal("s").goal == "concurrent user goal"


def test_registry_get_and_own_goal_redirect(home):
    atomic_config_write(home / "config.yaml", {"goals": {"agent_tool": True}})
    load_tool()
    def call(**args):
        return json.loads(registry.dispatch("goal", args))
    with scoped_current_session_id("s"):
        assert call(action="get")["goal"] is None
        assert call(action="create", objective="First objective")["created"]
        assert "replace" in call(action="create", objective="Redirected objective")["error"]
        assert call(action="create", objective="Redirected objective", replace=True)["created"]
        assert call(action="get")["goal"]["goal"] == "Redirected objective"
        assert "error" in call(action="complete")
        assert "error" in call(action="wait")
