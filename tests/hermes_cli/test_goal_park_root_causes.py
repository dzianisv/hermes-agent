"""Root causes of bad goal parking (one test group per pattern).

Driven through the real GoalManager + real SessionDB in a temp HERMES_HOME. Only the judge's LLM
call (``_call_goal_judge_llm``) is replaced by a local function, so judge_goal's prompt building
and parsing run for real.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def goals(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals as g
    g._DB_CACHE.clear()
    monkeypatch.setattr(g, "draft_contract", lambda objective, **kw: None)
    yield g
    g._DB_CACHE.clear()


def _judge(goals, monkeypatch, replies):
    """Judge returns each reply in turn (last one repeats). A reply that is an Exception is raised."""
    seq = list(replies)

    def fake(call_llm, system_prompt, user_prompt, timeout):
        r = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(r, Exception):
            raise r
        return json.dumps(r)
    monkeypatch.setattr(goals, "_call_goal_judge_llm", fake)


CONTINUE = {"verdict": "continue", "reason": "more to do"}


# ── (1) silence: never charged to the budget; live subagents = wait on them ────────────────────

def test_silent_turn_does_not_consume_turn_budget(goals, monkeypatch):
    _judge(goals, monkeypatch, [CONTINUE])
    mgr = goals.GoalManager(session_id="p1-budget")
    mgr.set_agent_goal("watch the deploy")
    for _ in range(3):
        mgr.state.waiting_until = 0.0
        mgr.state.barrier_kind = ""
        mgr.state.status = "active"
        mgr.evaluate_after_turn("[SILENT]")
    assert goals.GoalManager("p1-budget").state.turns_used == 0


def test_silence_with_live_subagents_waits_on_them(goals, monkeypatch):
    _judge(goals, monkeypatch, [CONTINUE])
    mgr = goals.GoalManager(session_id="p1-deleg")
    mgr.set_agent_goal("fan out the fixes")
    d = mgr.evaluate_after_turn("[SILENT]", active_delegations=2)
    s = goals.GoalManager("p1-deleg").state
    assert d["should_continue"] is False and d["message"] == ""
    assert s.waiting_on_delegations == 2 and s.status != "paused"


def test_self_wake_prompt_asks_for_declared_wait_not_silence(goals):
    mgr = goals.GoalManager(session_id="p1-prompt")
    mgr.set_agent_goal("ship it")
    prompt = mgr.next_continuation_prompt() or ""
    assert "[SILENT]" not in prompt
    assert 'action=\\"wait\\"' in json.dumps(prompt) or 'action="wait"' in prompt


def test_goal_tool_wait_on_user_parks_until_user_reply(goals, monkeypatch):
    import tools.goal_tool  # noqa: F401
    from tools.registry import registry
    monkeypatch.setattr(goals, "agent_goal_tool_enabled", lambda: True)
    _judge(goals, monkeypatch, [CONTINUE])
    mgr = goals.GoalManager(session_id="p1-tool")
    mgr.set_agent_goal("publish the extension")
    out = json.loads(registry.dispatch("goal", {"action": "wait", "on": "user",
                                                "question": "Approve the store publish with Touch ID?"},
                                       session_id="p1-tool"))
    assert out.get("waiting") is True, out
    mgr = goals.GoalManager(session_id="p1-tool")
    assert mgr.is_waiting() and mgr.state.barrier_kind == "user"


# ── (2) same human step reworded: wait on the user, lifted only by their reply ─────────────────

# Real rewordings from session af900b (agent.log "human-only step repeated").
AF900B = [
    "Will you approve publishing VibeBrowser 1.1.43 to the Chrome Web Store using Touch ID?",
    "Can you provide Touch ID approval to publish the extension to the Chrome Web Store?",
    "Please confirm with Touch ID so I can submit 1.1.43 to the Chrome Web Store.",
]


# ── (3) long-running goals stop on lack of progress, not on a raw turn count ──────────────────

def test_long_running_user_goal_with_progress_not_capped_at_20(goals, monkeypatch):
    _judge(goals, monkeypatch, [CONTINUE])
    mgr = goals.GoalManager(session_id="p3-long")
    mgr.set("monitor and fix CI until green")
    for i in range(30):
        import hashlib
        h = hashlib.sha256(str(i).encode()).hexdigest()
        d = mgr.evaluate_after_turn(f"Turn {i}: {h} — {h[::-1]}")
        assert d["should_continue"] is True, (i, d)


def test_user_goal_pauses_when_turns_make_no_progress(goals, monkeypatch):
    _judge(goals, monkeypatch, [CONTINUE])
    mgr = goals.GoalManager(session_id="p3-stall")
    mgr.set("monitor and fix CI until green")
    outcomes = [mgr.evaluate_after_turn("Still checking CI; nothing new yet.") for _ in range(10)]
    assert any(o["verdict"] == "stalled" and o["message"] for o in outcomes)
    s = goals.GoalManager("p3-stall").state
    assert s.status == "waiting" and s.barrier_kind == "safety"


def test_explicit_user_turn_budget_still_enforced(goals, monkeypatch):
    _judge(goals, monkeypatch, [CONTINUE])
    mgr = goals.GoalManager(session_id="p3-explicit")
    mgr.set("bounded goal", max_turns=2)
    mgr.evaluate_after_turn("step one")
    d = mgr.evaluate_after_turn("step two different")
    assert d["should_continue"] is False and "turns used" in d["message"]


# ── (4) reversible choices go back to the agent; money/creds/irreversible still pause ──────────

def test_reversible_choice_returns_to_agent_on_user_goal(goals, monkeypatch):
    _judge(goals, monkeypatch, [{"verdict": "blocked", "blocker": "human_only", "human_category": "reversible_choice",
                                 "human_decision": "Which model should the SE agent use?", "reason": "model choice"}])
    mgr = goals.GoalManager(session_id="p4-rev")
    mgr.set("set up the SE agent")
    d = mgr.evaluate_after_turn("which model do you want?")
    assert d["status"] == "active" and d["should_continue"] is True
    assert "Which model should the SE agent use?" in d["continuation_prompt"]
    assert goals.GoalManager("p4-rev").state.status == "active"


def test_reversible_choice_keyword_fallback_without_category(goals):
    assert goals.classify_human_block(None, "Which model should the SE agent use?") == "reversible_choice"
    assert goals.classify_human_block(None, "Approve the $40/month charge?") == "money"
    assert goals.classify_human_block(None, "Please provide the GitHub token") == "credentials"


@pytest.mark.parametrize("category,question", [
    ("money", "Approve spending $200 on the domain?"),
    ("credentials", "Grant repo admin permission?"),
    ("irreversible", "Delete the production database?"),
])
def test_legit_human_blocks_still_pause_user_goal(goals, monkeypatch, category, question):
    _judge(goals, monkeypatch, [{"verdict": "blocked", "blocker": "human_only", "human_category": category,
                                 "human_decision": question, "reason": category}])
    mgr = goals.GoalManager(session_id=f"p4-{category}")
    mgr.set("do the thing")
    d = mgr.evaluate_after_turn("need the human")
    assert d["should_continue"] is False and d["status"] == "waiting"
    s = goals.GoalManager(f"p4-{category}").state
    assert s.status == "waiting" and s.barrier_kind == "safety"


# ── (5) judge outage never pauses; back off, alert once, resume ───────────────────────────────

def test_judge_outage_never_pauses_alerts_once_and_resumes(goals, monkeypatch):
    err = ConnectionError("Connection error.")
    _judge(goals, monkeypatch, [err] * 8 + [CONTINUE])
    mgr = goals.GoalManager(session_id="p5")
    mgr.set("long user goal")
    alerts = []
    for _ in range(8):
        mgr.state.waiting_until = 0.0
        mgr.state.status = "active"
        mgr._save()
        d = mgr.evaluate_after_turn("did work")
        assert goals.GoalManager("p5").state.barrier_kind != "safety", d
        if d["message"]:
            alerts.append(d["message"])
    assert len(alerts) == 1
    s = goals.GoalManager("p5").state
    assert s.turns_used == 0  # judge failures are not charged to the goal
    mgr.state.waiting_until = 0.0
    mgr.state.status = "active"
    mgr._save()
    d = mgr.evaluate_after_turn("did more work")
    assert d["should_continue"] is True and d["verdict"] == "continue"
    assert goals.GoalManager("p5").state.consecutive_transport_failures == 0
