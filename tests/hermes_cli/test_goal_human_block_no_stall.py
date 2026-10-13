"""A human-only block on ONE step must not stall the whole agent goal, and a parked goal must not
pin the session's goal slot forever.

Behaviour contracts, driven through the real GoalManager + real SessionDB state in a temp
HERMES_HOME. Only the judge's LLM call is replaced by a fixed local function, so judge_goal's
prompt + parsing run for real. No source text is read.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


@pytest.fixture
def goals_mod(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals
    goals._DB_CACHE.clear()
    yield goals
    goals._DB_CACHE.clear()


@pytest.fixture
def hermes_home(goals_mod, tmp_path):
    return tmp_path / ".hermes"


def _judge(goals, monkeypatch, reply: dict):
    seen: list = []

    def fake(call_llm, system_prompt, user_prompt, timeout):
        seen.append(system_prompt)
        return json.dumps(reply)
    monkeypatch.setattr(goals, "_call_goal_judge_llm", fake)
    return seen


TOUCH_ID = "Complete the Chrome Web Store privacy form with your Touch ID."
RELEASE_GOAL = "Release VibeBrowser 1.1.43 to the Chrome Web Store, then launch on Product Hunt"
RELEASE_CRITERIA = ("PR #2270 merged; reconnect-fix card closed; release-gate run green; master CI green; "
                    "extension published; Product Hunt post live")


def _release_goal(goals, sid="rel"):
    mgr = goals.GoalManager(session_id=sid)
    contract = goals.GoalContract(outcome=RELEASE_CRITERIA, verification="gh pr view / CI dashboard")
    assert mgr.set_agent_goal(RELEASE_GOAL, contract=contract) is not None
    return mgr



def _human_block(question=TOUCH_ID):
    return {"verdict": "blocked", "blocker": "human_only", "human_decision": question,
            "reason": "Publishing to the Chrome Web Store needs the user's Touch ID."}


MUSE_KICK = "Continue working toward the active goal, verify requirement by requirement."


# ── 1) blocked / human_only never parks an agent goal ─────────────────────────────────────────


def test_human_only_block_keeps_the_goal_active_and_records_the_question(goals_mod, monkeypatch):
    seen = _judge(goals_mod, monkeypatch, _human_block())
    mgr = _release_goal(goals_mod)
    d = mgr.evaluate_after_turn("The privacy form needs Touch ID; I can't complete it. Stopping here.")
    assert d["status"] == "active" and d["should_continue"] is True
    kick = d["continuation_prompt"]
    assert MUSE_KICK in kick
    assert f"Step needing the human: {TOUCH_ID}" in kick and "waiting-on-user" in kick
    assert "PR #2270 merged" in kick                      # the other requirements ride along
    assert TOUCH_ID in d["message"]
    # persisted for /goal status and the status bar, on a fresh manager
    again = goals_mod.GoalManager(session_id="rel")
    assert again.state.status == "active"
    assert again.state.pending_human_decisions == [TOUCH_ID]
    assert "waiting on user" in again.status_line() and "Touch ID" in again.status_line()
    assert "human_only" in seen[0]


def test_same_question_is_recorded_once_and_never_pauses(goals_mod, monkeypatch):
    """The same human-only question again = nothing else left to do. The goal stays ACTIVE (never
    paused) but waits with a doubling backoff instead of re-poking at once — re-poking re-asked the
    Touch ID question ~6x/min for an hour (82 notices in the user's chat, 2026-10-10)."""
    _judge(goals_mod, monkeypatch, _human_block())
    mgr = _release_goal(goals_mod)
    d = mgr.evaluate_after_turn("still needs Touch ID 0")
    assert d["status"] == "active" and d["should_continue"] is True and MUSE_KICK in d["continuation_prompt"]
    for i in range(1, 8):
        mgr.stop_waiting()
        d = mgr.evaluate_after_turn(f"still needs Touch ID {i}")
        assert d["status"] == "waiting" and d["should_continue"] is False and d["message"] == "", d
        assert goals_mod.load_goal("rel").status == "waiting"
    s = goals_mod.load_goal("rel")
    assert s.pending_human_decisions == [TOUCH_ID]
    assert s.consecutive_human_pending == 7 and s.barrier_kind == "user" and s.waiting_until == 0


def test_distinct_human_questions_accumulate(goals_mod, monkeypatch):
    mgr = _release_goal(goals_mod)
    _judge(goals_mod, monkeypatch, _human_block())
    mgr.evaluate_after_turn("needs Touch ID")
    _judge(goals_mod, monkeypatch, _human_block(question="Approve the $49 Product Hunt promo spend?"))
    d = mgr.evaluate_after_turn("needs spend approval")
    assert d["status"] == "active"
    pend = goals_mod.load_goal("rel").pending_human_decisions
    assert pend == [TOUCH_ID, "Approve the $49 Product Hunt promo spend?"]
    assert "Waiting on the user" in d["continuation_prompt"] and TOUCH_ID in d["continuation_prompt"]


def test_human_only_without_a_question_falls_back_to_the_reason(goals_mod, monkeypatch):
    _judge(goals_mod, monkeypatch, {"verdict": "blocked", "blocker": "human_only", "reason": "needs owner sign-off"})
    mgr = _release_goal(goals_mod)
    d = mgr.evaluate_after_turn("waiting on sign-off")
    assert d["status"] == "active"
    assert goals_mod.load_goal("rel").pending_human_decisions == ["needs owner sign-off — approve or decide how to proceed?"]


def test_pending_question_stays_listed_after_progress(goals_mod, monkeypatch):
    mgr = _release_goal(goals_mod)
    _judge(goals_mod, monkeypatch, _human_block())
    mgr.evaluate_after_turn("needs Touch ID")
    _judge(goals_mod, monkeypatch, {"verdict": "continue", "reason": "PR #2270 merged, CI running"})
    d = mgr.evaluate_after_turn("merged PR #2270")
    assert d["status"] == "active" and MUSE_KICK in d["continuation_prompt"]
    assert goals_mod.load_goal("rel").pending_human_decisions == [TOUCH_ID]


def test_transient_block_continues_with_muse_kick(goals_mod, monkeypatch):
    _judge(goals_mod, monkeypatch, {"verdict": "blocked", "blocker": "transient", "reason": "approval prompt timed out"})
    mgr = _release_goal(goals_mod)
    d = mgr.evaluate_after_turn("approval prompt timed out")
    assert d["status"] == "active" and d["should_continue"] is True
    assert MUSE_KICK in d["continuation_prompt"] and "approval prompt timed out" in d["continuation_prompt"]
    assert goals_mod.load_goal("rel").pending_human_decisions == []


def test_idle_turn_with_continue_verdict_fires_the_muse_kick(goals_mod, monkeypatch):
    _judge(goals_mod, monkeypatch, {"verdict": "continue", "reason": "4 requirements still open"})
    mgr = _release_goal(goals_mod)
    d = mgr.evaluate_after_turn("Done for now, let me know if you want me to proceed.")
    assert d["status"] == "active" and d["should_continue"] is True
    assert MUSE_KICK in d["continuation_prompt"]


# ── only complete / user pause end an active agent goal (no turn budget) ───────────────────────


def test_user_pause_still_pauses_an_agent_goal(goals_mod, monkeypatch):
    mgr = _release_goal(goals_mod)
    assert mgr.pause() is not None
    assert goals_mod.load_goal("rel").status == "waiting"
    _judge(goals_mod, monkeypatch, _human_block())
    assert mgr.evaluate_after_turn("anything")["should_continue"] is False


def test_agent_goal_has_no_turn_budget(goals_mod, monkeypatch):
    """Live 2026-10-09: '⏸ Goal paused — 23/20 turns used' on an agent goal that still had work.
    Muse has no budget on agent goals; only acceptance criteria or a human-only blocker end them."""
    mgr = goals_mod.GoalManager(session_id="mt")
    assert mgr.set_agent_goal(RELEASE_GOAL, max_turns=2) is not None
    _judge(goals_mod, monkeypatch, {"verdict": "continue", "reason": "PR still open"})
    for i in range(5):
        d = mgr.evaluate_after_turn(f"t{i}")
        assert d["status"] == "active" and d["should_continue"] is True
    assert goals_mod.load_goal("mt").turns_used == 5


def test_user_goal_keeps_turn_budget(goals_mod, monkeypatch):
    mgr = goals_mod.GoalManager(session_id="ub")
    mgr.set("user goal", max_turns=2)
    _judge(goals_mod, monkeypatch, {"verdict": "continue", "reason": "more"})
    assert mgr.evaluate_after_turn("t1")["status"] == "active"
    d = mgr.evaluate_after_turn("t2")
    assert d["status"] == "waiting" and "budget" in goals_mod.load_goal("ub").paused_reason


def test_complete_still_closes_the_goal(goals_mod, monkeypatch):
    _judge(goals_mod, monkeypatch, {"verdict": "done", "reason": "all requirements verified"})
    mgr = _release_goal(goals_mod)
    assert mgr.evaluate_after_turn("everything shipped")["status"] == "done"


def test_user_goal_human_block_still_pauses_for_the_user(goals_mod, monkeypatch):
    _judge(goals_mod, monkeypatch, _human_block())
    mgr = goals_mod.GoalManager(session_id="u1")
    mgr.set("user goal: publish the extension")
    d = mgr.evaluate_after_turn("needs Touch ID")
    assert d["status"] == "waiting" and d["should_continue"] is False


# ── B) a parked (legacy / budget) agent goal must not pin the slot forever ─────────────────────


def _park(goals, monkeypatch, sid="p1"):
    """A legacy loop-parked row (written before Muse semantics) with a pending question."""
    mgr = _release_goal(goals, sid=sid)
    s = goals.load_goal(sid)
    s.status, s.paused_reason, s.paused_at = "paused", "agent goal parked (human_only): needs Touch ID", time.time()
    s.pending_human_decisions = [TOUCH_ID]
    goals.save_goal(sid, s)
    return mgr


def _age_park(goals, sid, hours):
    s = goals.load_goal(sid)
    s.paused_at = time.time() - hours * 3600
    goals.save_goal(sid, s)


def test_fresh_park_still_holds_the_slot(goals_mod, monkeypatch):
    from tools.goal_tool import goal_tool
    _park(goals_mod, monkeypatch)
    r = json.loads(goal_tool("create", objective="Launch VibeBrowser 1.1.43 on Product Hunt", session_id="p1"))
    assert "error" in r and "parked" in r["error"]
    assert goals_mod.load_goal("p1").goal == RELEASE_GOAL


def test_park_older_than_the_window_is_replaceable_and_archived(goals_mod, monkeypatch):
    from tools.goal_tool import goal_tool
    _park(goals_mod, monkeypatch)
    _age_park(goals_mod, "p1", hours=25)
    r = json.loads(goal_tool("create", objective="Launch VibeBrowser 1.1.43 on Product Hunt", session_id="p1"))
    assert r.get("created") is True
    live = goals_mod.load_goal("p1")
    assert live.goal == "Launch VibeBrowser 1.1.43 on Product Hunt" and live.status == "active"
    # nothing lost: the old state sits under a goal_archive key, pending question included
    rows = goals_mod._get_session_db().list_meta_prefix("goal_archive:p1:")
    assert rows, "archived goal row missing"
    archived = goals_mod.GoalState.from_json(rows[0][1])
    assert archived.goal == RELEASE_GOAL and archived.pending_human_decisions == [TOUCH_ID]


def test_release_window_is_configurable(goals_mod, monkeypatch, hermes_home):
    from tools.goal_tool import goal_tool
    (hermes_home / "config.yaml").write_text("goals:\n  parked_release_hours: 2\n", encoding="utf-8")
    _park(goals_mod, monkeypatch, sid="p2")
    _age_park(goals_mod, "p2", hours=1)
    assert "error" in json.loads(goal_tool("create", objective="Something else entirely now", session_id="p2"))
    _age_park(goals_mod, "p2", hours=3)
    assert json.loads(goal_tool("create", objective="Something else entirely now", session_id="p2")).get("created") is True


def test_user_paused_agent_goal_is_never_released(goals_mod, monkeypatch):
    mgr = _release_goal(goals_mod, sid="p3")
    mgr.pause()                       # explicit /goal pause
    _age_park(goals_mod, "p3", hours=1000)
    assert goals_mod.GoalManager(session_id="p3").accepts_agent_goal() is False


def test_user_set_goal_is_never_released(goals_mod, monkeypatch):
    mgr = goals_mod.GoalManager(session_id="p4")
    mgr.set("user goal")
    mgr.pause("turn budget exhausted (20/20)")
    _age_park(goals_mod, "p4", hours=1000)
    assert goals_mod.GoalManager(session_id="p4").accepts_agent_goal() is False


def test_legacy_parked_row_without_paused_at_uses_last_turn_time(goals_mod, monkeypatch):
    """Rows written before paused_at existed (the 12-day stall) must still age out."""
    _park(goals_mod, monkeypatch, sid="p5")
    s = goals_mod.load_goal("p5")
    s.paused_at = 0.0
    s.last_turn_at = time.time() - 12 * 24 * 3600
    goals_mod.save_goal("p5", s)
    assert goals_mod.GoalManager(session_id="p5").accepts_agent_goal() is True