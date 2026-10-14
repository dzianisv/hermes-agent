"""A goal continuation answered with only a silence marker must not re-poke the agent at once.

Live failure: the model kept answering ``NO_REPLY`` to "Continue working toward the active
goal"; the judge said ``continue`` every time (nothing to judge), and the gateway fired ~300
continuations in 30 minutes. Silent turns now park with growing backoff, skip the judge, and
pause the goal after repeats.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import time

import pytest

from hermes_cli import goals


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    goals._DB_CACHE.clear()
    goals._get_session_db()
    yield home
    reset_hermes_home_override(token)
    goals._DB_CACHE.clear()


def _goal(sid: str) -> goals.GoalManager:
    # Silence backoff applies to user /goal; agent goals get one reminder then pause
    # (test_goal_agent_no_judge.py).
    mgr = goals.GoalManager(sid)
    assert mgr.set("ship the PR") is not None
    return mgr


@pytest.mark.parametrize("text", ["NO_REPLY", "[SILENT]", "*NO_REPLY*", "",
                                  "⚠️ The model returned only a silence marker for a message that needed a reply. Try again or rephrase."])
def test_silence_marker_is_a_silent_turn(text):
    assert goals.is_silent_turn_response(text)


def test_prose_mentioning_marker_is_not_silent():
    assert not goals.is_silent_turn_response("I replied NO_REPLY earlier, now CI is green.")


def test_silent_turn_parks_with_backoff_and_skips_judge(hermes_home):
    mgr = _goal("sid-silent-1")
    with patch("hermes_cli.goals.judge_goal") as judge:
        d1 = mgr.evaluate_after_turn("NO_REPLY")
        assert not d1["should_continue"] and d1["verdict"] == "silent"
        assert judge.call_count == 0
        s = goals.GoalManager("sid-silent-1").state
        assert s.waiting_until > 0 and s.consecutive_silent == 1

        # Still inside the backoff: the loop stays quiet without burning a turn.
        d_wait = mgr.evaluate_after_turn("NO_REPLY")
        assert d_wait["verdict"] == "waiting"
        assert judge.call_count == 0

        # Barrier lifted -> second silent turn doubles the backoff.
        mgr.stop_waiting()
        mgr.evaluate_after_turn("NO_REPLY")
        s2 = goals.GoalManager("sid-silent-1").state
        assert s2.consecutive_silent == 2
        assert round(s2.waiting_until - s2.waiting_since) >= 2 * goals.SILENT_TURN_BACKOFF_BASE_S - 2


def test_real_reply_resets_silent_counter(hermes_home):
    mgr = _goal("sid-silent-2")
    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "more to do", False, None, False)):
        mgr.evaluate_after_turn("NO_REPLY")
        mgr.stop_waiting()
        d = mgr.evaluate_after_turn("Pushed the fix; CI running.")
        assert d["should_continue"]
        assert goals.GoalManager("sid-silent-2").state.consecutive_silent == 0


def test_repeated_silence_never_pauses_goal_and_caps_backoff(hermes_home):
    """Silence is not a blocker. A paused goal just becomes a park a supervisor un-parks,
    which restarts the continuation spam (live 2026-10-09: parked twice in two hours)."""
    mgr = _goal("sid-silent-3")
    with patch("hermes_cli.goals.judge_goal") as judge:
        for _ in range(goals.DEFAULT_MAX_CONSECUTIVE_SILENT_TURNS + 4):
            mgr.stop_waiting()
            d = mgr.evaluate_after_turn("NO_REPLY")
        assert d["status"] == "waiting"
        assert judge.call_count == 0
        st = goals.GoalManager("sid-silent-3").state
        assert st.status == "waiting"
        remaining = st.waiting_until - time.time()
        cap = goals.SILENT_TURN_BACKOFF_MAX_S
        assert cap - 5 < remaining <= cap


def test_silent_park_emits_no_user_message(hermes_home):
    """A silent turn on a continuation is internal: no '⏳ Goal parked …' notice (it spammed the
    user's chat every 30 min on 2026-10-10)."""
    mgr = _goal("silent-quiet")
    for _ in range(3):
        mgr.state.waiting_until = 0.0
        mgr._save()
        d = mgr.evaluate_after_turn("[SILENT]", user_initiated=True)
        assert d["verdict"] == "silent"
        assert d["message"] == ""


def test_turn_inside_silence_backoff_emits_no_user_message(hermes_home):
    """A reply landing while the silence park is still running must stay quiet too (second leak
    path: _waiting_decision re-announced the park as '⏳ Goal parked — waiting on 17s remaining')."""
    mgr = _goal("silent-quiet-2")
    d1 = mgr.evaluate_after_turn("[SILENT]", user_initiated=True)
    assert d1["verdict"] == "silent" and d1["message"] == ""
    d2 = mgr.evaluate_after_turn("some real text", user_initiated=True)   # still inside the 60s park
    assert d2["verdict"] == "waiting"
    assert d2["message"] == ""


def test_turn_inside_any_park_emits_no_user_message(hermes_home):
    """A judge/agent wait is announced once by the parker; later turns inside it stay quiet."""
    mgr = _goal("wait-quiet")
    mgr.wait_for_seconds(300, reason="Waiting for the delegated live-test subagent")
    d = mgr.evaluate_after_turn("subagent finished, here is the output", user_initiated=False)
    assert d["verdict"] == "waiting"
    assert d["message"] == ""

