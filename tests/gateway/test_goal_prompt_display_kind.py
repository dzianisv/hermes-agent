"""Goal continuation prompts are self-injected machinery: the agent may answer them with a bare
silence marker and nothing is delivered.

Live failure (2026-10-09): the agent answered "Continue working toward the active goal" with
``[SILENT]``; the gateway treated the synthetic prompt as a HUMAN turn and sent the user
"⚠️ The model returned only a silence marker for a message that needed a reply" every time.
"""
from gateway.config import Platform
from gateway.response_filters import (
    INTERNAL_NOTIFICATION_DISPLAY_KIND, display_kind_for_event, is_machinery_display_kind,
)
from gateway.run_goals import GatewayGoalsMixin
from gateway.session import SessionSource


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="1", user_id="1", chat_type="dm",
                         thread_id="7", message_id="42")


def test_goal_prompt_event_is_machinery_but_not_internal():
    event = GatewayGoalsMixin._synthetic_prompt_event(_source(), "Continue working toward the active goal")
    assert not event.internal  # authorization and the emergency stop still apply
    assert display_kind_for_event(event) == INTERNAL_NOTIFICATION_DISPLAY_KIND
    assert is_machinery_display_kind(display_kind_for_event(event))
    assert event.source.message_id is None


def test_internal_goal_prompt_event_stays_machinery():
    event = GatewayGoalsMixin._synthetic_prompt_event(_source(), "wake", internal=True)
    assert event.internal
    assert display_kind_for_event(event) == INTERNAL_NOTIFICATION_DISPLAY_KIND
