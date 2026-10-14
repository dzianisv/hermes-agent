"""goal complete: one AI evidence check. 'No' keeps the goal open with the reason; checker down =
'unverified', never done. Replays the logged real refusals (agent.log 2026-10-08..13)."""
from __future__ import annotations

import json

import pytest

import hermes_cli.goals as goals_mod
import tools.goal_tool  # noqa: F401
from tools.registry import registry

# Refusal verdicts as logged by `completion audit refused:` in ~/.hermes/logs/agent.log*.
LOGGED_REFUSALS = [
    ("continue", "The response summarizes results but does not show concrete verification evidence (command output for git log fork/installed, pytest results, and the Notion page)."),
    ("continue", "The required Telegram-ready report covering all ten symbols with the specified metrics and verdicts was not included in the response."),
    ("continue", "The agent has not yet provided the final line-referenced review report containing concrete bugs and minimal fixes."),
    ("continue", "The response makes claims without providing the required verification evidence (command outputs for `gh pr view` and `ntn pages get`)."),
    ("continue", "The response asserts the verification criteria are met but does not include the concrete command outputs or excerpts required by the decision rules."),
    ("continue", "The goal required creating the Slack app named 'Jared Dunn', but the app installed was 'Doc Brown'."),
    ("blocked", "Paperclip inspection failed because the managed shim encountered \"MODULE_NOT_FOUND\" and instances/default/config.json is missing."),
    ("continue", "The response summarizes diagnostic findings but lacks concrete command outputs or excerpted evidence required to satisfy the verification criterion."),
    ("continue", "The agent provided a summary of actions taken but has not yet delivered the actual Telegram-ready stock review report required by the completion contract."),
    ("continue", "The response only claims the report was written to disk without providing the Telegram-ready report text, file excerpt, or verification command output."),
]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(goals_mod, "_get_session_db", lambda: db)
    monkeypatch.setattr(goals_mod, "draft_contract", lambda objective, **kw: None)
    monkeypatch.setattr(goals_mod, "agent_goal_tool_enabled", lambda: True)
    yield db
    db.close()


def _call(**args):
    return json.loads(registry.dispatch("goal", args, session_id="s1"))


def _llm(monkeypatch, reply, seen=None):
    def fake(call_llm, system_prompt, user_prompt, timeout):
        if seen is not None:
            seen.append(user_prompt)
        if isinstance(reply, Exception):
            raise reply
        return json.dumps(reply)
    monkeypatch.setattr(goals_mod, "_call_goal_judge_llm", fake)


@pytest.mark.parametrize("verdict,reason", LOGGED_REFUSALS)
def test_logged_refusals_replay_keep_goal_open_with_reason(monkeypatch, verdict, reason):
    _call(action="create", objective="Ship the reviewed fix and report proof to Den")
    _llm(monkeypatch, {"verdict": verdict, "reason": reason})
    r = _call(action="complete", evidence="Done, everything verified and the report was sent.")
    assert "error" in r and r["reason"] == reason
    assert goals_mod.GoalManager(session_id="s1").is_active()


def test_concrete_evidence_passes_and_is_judged_once(monkeypatch):
    _call(action="create", objective="Ship the reviewed fix and report proof to Den")
    seen: list = []
    _llm(monkeypatch, {"verdict": "done", "reason": "pytest 12 passed, PR merged"}, seen)
    r = _call(action="complete", evidence="pytest tests/x.py: 12 passed; https://github.com/o/r/pull/1 merged")
    assert r["completed"] is True and len(seen) == 1
    assert "pull/1" in seen[0] and "concrete evidence" in seen[0]
    assert not goals_mod.GoalManager(session_id="s1").has_goal()


@pytest.mark.parametrize("failure", [TimeoutError("judge timed out"), "not json"])
def test_checker_down_is_unverified_and_goal_stays_open(monkeypatch, failure):
    _call(action="create", objective="Ship the reviewed fix and report proof to Den")
    if isinstance(failure, Exception):
        _llm(monkeypatch, failure)
    else:
        monkeypatch.setattr(goals_mod, "_call_goal_judge_llm", lambda *a: failure)
    r = _call(action="complete", evidence="pytest tests/x.py: 12 passed; run id 20261013_1")
    assert "error" in r and r["verdict"] == "unverified"
    mgr = goals_mod.GoalManager(session_id="s1")
    assert mgr.is_active() and mgr.state.status != "done"
