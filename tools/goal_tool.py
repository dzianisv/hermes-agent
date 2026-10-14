"""goal — the agent's own handle on its session goal (Muse-style).

The model declares, inspects, and closes the per-session goal itself. Backed by the same ``GoalManager`` / ``state_meta goal:<sid>``
row as ``/goal``, so the post-turn judge, gates, heartbeat and ``/goal status`` all see it.
Service-gated on ``goals.agent_tool`` (off by default): a profile that wants goal-owning agents
opts in; everyone else keeps the schema untouched.

Rules the handler enforces (never the prompt alone):
- ``create`` never overwrites a user-set goal — the user's ``/goal`` always wins. It replaces the
  agent's OWN goal only with ``replace=true`` (the user redirected the work), archiving the old row.
- ``complete`` runs the completion audit (``judge_goal`` against the agent's stated evidence) and
  refuses when the verdict is not DONE, so an agent cannot wave its own goal through.
- The goal is tagged ``source="agent"``: ``self-set`` in the status line; never judged per turn,
  only heartbeat-continued (no turn budget).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_MIN_OBJECTIVE_CHARS = 12
_MAX_OBJECTIVE_CHARS = 600


def check_goal_tool_requirements() -> bool:
    """Opt-in per profile via ``goals.agent_tool: true``."""
    try:
        from hermes_cli.goals import agent_goal_tool_enabled
        return bool(agent_goal_tool_enabled())
    except Exception:
        return False


GOAL_SCHEMA = {
    "name": "goal",
    "description": (
        "Your session goal: one durable objective this conversation is working toward, judged "
        "after each of your replies and driven forward on the heartbeat until it is done. "
        "Call `get` first. Call `create` when you commit to a multi-step outcome the user wants "
        "and no goal is set — state the outcome, not the next step, plus the acceptance criteria "
        "the audit will check. Call `complete` only when "
        "you can cite evidence the outcome exists; the completion audit refuses otherwise. "
        "Never create a goal for a plain question, a one-shot answer, or work the user cancelled. "
        "If the user redirects you and the current goal is one YOU set (source=agent), call `create` "
        "with replace=true — do not ask the user to run /goal clear. A user-set /goal always wins "
        "and cannot be replaced here. When you cannot act until something finishes, call `wait` "
        "(on=subagents|process|seconds|user) instead of replying with silence."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["get", "create", "link", "complete", "wait"]},
            "on": {
                "type": "string", "enum": ["subagents", "process", "seconds", "user"],
                "description": (
                    "For wait: what the goal is blocked on. subagents = your live delegations (lifts when one "
                    "returns); process = a background process (process_session); seconds = a fixed time "
                    "(seconds); user = a step only the user can do (question) — lifted only by their reply."
                ),
            },
            "process_session": {"type": "string", "description": "For wait on=process: the process session id."},
            "seconds": {"type": "integer", "description": "For wait on=seconds: how long."},
            "question": {"type": "string", "description": "For wait on=user: the one concrete question for the user."},
            "reason": {"type": "string", "description": "For wait: one line on why."},
            "replace": {
                "type": "boolean",
                "description": "For create: replace the agent's own current goal (source=agent) because the user "
                               "redirected the work. Refused for user-set goals.",
            },
            "objective": {
                "type": "string",
                "description": "For create: the outcome in one sentence, verifiable, 12–600 chars.",
            },
            "acceptance_criteria": {
                "type": "string",
                "description": (
                    "For create: the observable end state the completion audit checks — what must be true, "
                    "where it can be seen. Write it yourself from the user's ask; do not leave it to inference."
                ),
            },
            "verification": {
                "type": "string",
                "description": "For create: how the audit confirms the criteria (commands, URLs, checks).",
            },
            "evidence": {
                "type": "string",
                "description": "For complete: what proves the acceptance criteria are met (commands run, URLs, ids).",
            },
            "task_ids": {
                "type": "array", "items": {"type": "string"},
                "description": (
                    "For create/link: kanban card ids (t_...) this goal is about. Board DONE wakes for "
                    "cards not linked to the active goal are held until the next admitted wake."
                ),
            },
            "max_turns": {
                "type": "integer",
                "description": "For create: optional continuation budget; default from goals.max_turns.",
            },
        },
        "required": ["action"],
    },
}


def _ok(**payload: Any) -> str:
    return json.dumps(payload)


def _manager(session_id: Optional[str]):
    from hermes_cli.goals import GoalManager
    sid = (session_id or "").strip()
    if not sid:
        return None
    return GoalManager(session_id=sid)


def _state_view(mgr) -> Dict[str, Any]:
    s = mgr.state
    if s is None or not mgr.has_goal():
        return {"goal": None}
    view: Dict[str, Any] = {
        "goal": s.goal, "status": s.status, "source": s.source,
        "turns_used": s.turns_used, "max_turns": s.max_turns,
        "last_verdict": s.last_verdict, "last_reason": s.last_reason,
    }
    if s.has_contract():
        c = s.contract
        view["contract"] = {k: getattr(c, k) for k in ("outcome", "verification", "stop_when") if getattr(c, k, None)}
    if s.linked_task_ids:
        view["linked_task_ids"] = list(s.linked_task_ids)
    if s.pending_human_decisions:
        view["pending_human_decisions"] = list(s.pending_human_decisions)
    if s.waiting_reason:
        view["waiting"] = s.waiting_reason
    return view


def _build_contract(objective: str, acceptance_criteria: str, verification: str):
    """The agent's own acceptance criteria are the contract (Muse parity). The judge drafts one
    only when the agent gave none; it never overrides what the agent stated."""
    from hermes_cli.goals import GoalContract, draft_contract
    acceptance_criteria = (acceptance_criteria or "").strip()
    verification = (verification or "").strip()
    if acceptance_criteria:
        return GoalContract(outcome=acceptance_criteria, verification=verification)
    try:
        return draft_contract(objective)
    except Exception as exc:  # contract is a quality aid, never a blocker
        logger.info("goal tool: draft_contract failed (%s)", exc)
        return None


def goal_tool(action: str, *, objective: str = "", evidence: str = "", max_turns: Optional[int] = None,
              acceptance_criteria: str = "", verification: str = "", replace: bool = False,
              task_ids: Optional[list] = None, session_id: Optional[str] = None, on: str = "",
              process_session: str = "", seconds: Optional[int] = None, question: str = "", reason: str = "") -> str:
    mgr = _manager(session_id)
    if mgr is None:
        return tool_error("goal tool needs a session; none is bound to this call")

    if action == "get":
        return _ok(**_state_view(mgr))

    if action == "create":
        if mgr.session_id.startswith("cron_"):
            # A cron run is one turn with no gateway loop behind it: its goal can never be judged
            # or woken, so it only becomes a stale "active goal idle" row (probe I2).
            return tool_error("cron runs are one-shot and have no goal loop; do the work in this turn "
                              "or schedule a follow-up job instead of setting a goal")
        objective = (objective or "").strip()
        if not (_MIN_OBJECTIVE_CHARS <= len(objective) <= _MAX_OBJECTIVE_CHARS):
            return tool_error(f"objective must be {_MIN_OBJECTIVE_CHARS}–{_MAX_OBJECTIVE_CHARS} chars")
        if replace and mgr.has_goal() and mgr.state is not None and mgr.state.source != "agent":
            return tool_error("the current goal was set by the user with /goal; it cannot be replaced by the "
                              "agent — the user clears it with /goal clear", **_state_view(mgr))
        if not mgr.accepts_agent_goal() and not (replace and mgr.state is not None and mgr.state.source == "agent"):
            s = mgr.state
            if s is not None and s.status == "waiting" and s.source == "agent":
                return tool_error("this session's goal is parked and still holds the slot; the user answers "
                                  "its pending question and runs /goal resume, or it becomes replaceable after "
                                  "goals.parked_release_hours", **_state_view(mgr))
            return tool_error("a goal is already set for this session; finish it with complete, "
                              "pass replace=true if it is your own goal and the user redirected you, "
                              "or the user clears it with /goal clear", **_state_view(mgr))
        contract = _build_contract(objective, acceptance_criteria, verification)
        state = mgr.set_agent_goal(objective, contract=contract, max_turns=max_turns, replace=bool(replace))
        if state is None:
            return tool_error("goal not set", **_state_view(mgr))
        if task_ids:
            mgr.link_tasks(task_ids)
        return _ok(created=True, **_state_view(mgr))

    if action == "link":
        if not task_ids:
            return tool_error("link needs task_ids")
        if mgr.link_tasks(task_ids) is None:
            return tool_error("no goal to link cards to")
        return _ok(linked=True, **_state_view(mgr))

    if action == "complete":
        if not mgr.has_goal():
            return tool_error("no goal to complete")
        evidence = (evidence or "").strip()
        if len(evidence) < _MIN_OBJECTIVE_CHARS:
            return tool_error("complete needs evidence that the outcome exists")
        decision = mgr.complete_with_evidence(evidence)
        verdict, reason = decision["verdict"], decision["reason"]
        if verdict == "unavailable":
            return tool_error(f"completion audit unavailable ({reason}); keep working or retry")
        if verdict != "done":
            return tool_error(f"completion audit refused: {verdict} — {reason}", verdict=verdict, reason=reason)
        return _ok(completed=True, reason=reason)

    if action == "wait":
        return _wait(mgr, on=(on or "").strip().lower(), process_session=process_session, seconds=seconds,
                     question=question, reason=reason)

    return tool_error(f"unknown action {action!r}")


def _wait(mgr, *, on: str, process_session: str, seconds: Optional[int], question: str, reason: str) -> str:
    """Agent-declared barrier: the goal sleeps on a named thing, not on a judge guess or silence."""
    from hermes_cli.goals import count_active_delegations
    if not mgr.is_active():
        return tool_error("no active goal to wait on")
    try:
        if on == "subagents":
            live = count_active_delegations(mgr.session_id)
            if live <= 0:
                return tool_error("no live delegated subagents to wait on; keep working")
            mgr.wait_on_delegations(live, reason=reason)
        elif on == "process":
            if not (process_session or "").strip():
                return tool_error("wait on=process needs process_session")
            mgr.wait_on_session(process_session, reason=reason)
        elif on == "seconds":
            mgr.wait_for_seconds(int(seconds or 0), reason=reason)
        elif on == "user":
            mgr.wait_on_user(question or reason, reason=reason)
        else:
            return tool_error("wait needs on=subagents|process|seconds|user")
    except (TypeError, ValueError, RuntimeError) as exc:
        return tool_error(f"wait refused: {exc}")
    return _ok(**{**_state_view(mgr), "waiting": True, "on": on})


registry.register(
    name="goal", toolset="goal", schema=GOAL_SCHEMA, check_fn=check_goal_tool_requirements,
    handler=lambda args, **kw: goal_tool(
        str(args.get("action") or ""), objective=args.get("objective") or "", evidence=args.get("evidence") or "",
        acceptance_criteria=args.get("acceptance_criteria") or "", verification=args.get("verification") or "",
        max_turns=args.get("max_turns"), replace=bool(args.get("replace")),
        task_ids=args.get("task_ids") if isinstance(args.get("task_ids"), list) else None, session_id=kw.get("session_id"),
        on=str(args.get("on") or ""), process_session=str(args.get("process_session") or ""),
        seconds=args.get("seconds"), question=str(args.get("question") or ""), reason=str(args.get("reason") or "")),
    emoji="⊙",
)
