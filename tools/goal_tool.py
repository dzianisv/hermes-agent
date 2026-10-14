"""Opt-in agent goal declarations; completion remains owned by the turn judge."""
import json

from gateway.session_context import get_session_env
from hermes_cli.config_effective import load_user_config_effective
from hermes_cli.goals import GoalContract, GoalManager, DEFAULT_MAX_TURNS
from tools.registry import registry


def check_goal_requirements() -> bool:
    return (load_user_config_effective().get("goals") or {}).get("agent_tool") is True


def goal_tool(args, **kwargs) -> str:
    if not check_goal_requirements():
        return json.dumps({"error": "goal tool is disabled; goals.agent_tool must be true"})
    session_id = get_session_env("HERMES_SESSION_ID", "")
    if not session_id or session_id.startswith("cron_") or get_session_env("HERMES_CRON_SESSION", "") == "1":
        return json.dumps({"error": "goal tool requires an interactive, non-cron session"})
    config = load_user_config_effective().get("goals") or {}
    manager = GoalManager(session_id, default_max_turns=config.get("max_turns", DEFAULT_MAX_TURNS))
    action = args.get("action")
    if action == "get":
        return json.dumps({"goal": json.loads(manager.state.to_json()) if manager.has_goal() else None})
    if action != "create":
        return json.dumps({"error": "supported actions: get, create"})
    if manager.has_goal() and manager.state.source != "agent":
        return json.dumps({"error": "a user-set /goal cannot be replaced by the agent"})
    try:
        state = manager.set_agent_goal(
            args.get("objective", ""),
            contract=GoalContract(outcome=args.get("acceptance_criteria", ""),
                                  verification=args.get("verification", "")),
            max_turns=args.get("max_turns"), replace=args.get("replace", False),
        )
    except (ValueError, RuntimeError) as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps({"created": True, "goal": json.loads(state.to_json())})


GOAL_SCHEMA = {
    "name": "goal",
    "description": (
        "Declare a durable session objective for a multi-step user request. Call get first. "
        "Completion is judged after each turn under a binding turn budget. When the user redirects "
        "your own agent-set goal, call create with replace=true; do not ask for /goal clear. "
        "A user-set /goal is protected and cannot be replaced by this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["get", "create"]},
            "objective": {"type": "string", "description": "Verifiable user-requested outcome."},
            "acceptance_criteria": {"type": "string", "description": "Observable end state the judge must verify."},
            "verification": {"type": "string", "description": "Commands, artifacts or checks proving the outcome."},
            "replace": {"type": "boolean", "default": False},
            "max_turns": {"type": "integer", "minimum": 1, "description": "Binding turn cap; defaults to goals.max_turns."},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}

registry.register(name="goal", toolset="goal", schema=GOAL_SCHEMA,
                  handler=goal_tool, check_fn=check_goal_requirements)
