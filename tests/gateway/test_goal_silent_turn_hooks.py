"""A goal continuation answered with a bare silence marker is still a completed turn.

Live failure: the gateway lifted a wait barrier and injected the continuation, the agent
replied ``NO_REPLY`` (nothing to say yet), delivery suppressed the marker and returned
``None`` with no stashed text, so ``_run_post_turn_hooks`` skipped the goal judge. The
goal stayed ``active`` with no barrier and no continuation for 25+ minutes until a human
kicked it.
"""

import pytest

from gateway.platforms.event import MessageEvent, MessageType, SessionSource
from gateway.response_filters import SILENT_REPLY_TOKEN
from gateway.run import GatewayRunner
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _make_runner


def _event(adapter):
    source = SessionSource(platform=adapter.platform, chat_id="1", chat_type="dm", thread_id="7")
    return MessageEvent(text="continue", message_type=MessageType.TEXT, source=source, message_id="1"), source


@pytest.mark.asyncio
async def test_suppressed_silence_marker_still_reaches_post_turn_hooks():
    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    event, source = _event(adapter)

    class _Entry:
        session_id = "s-goal"

    agent_result = {"final_response": SILENT_REPLY_TOKEN, "messages": [], "api_calls": 1}
    delivered = await runner._hmwa_deliver_turn_response(
        event, source, _Entry(), "agent:main:telegram:dm:1:7", None,
        agent_result, [], SILENT_REPLY_TOKEN, None, True,
    )
    assert delivered == ""  # nothing goes to the chat
    # ...but the hooks must see a non-empty completed turn so the goal judge runs.
    assert GatewayRunner._final_text_for_post_turn_hooks(None, event).strip() == SILENT_REPLY_TOKEN


def test_hooks_treat_silence_marker_as_a_turn():
    event, _ = _event(ProgressCaptureAdapter())
    event._streamed_final_response = SILENT_REPLY_TOKEN
    assert GatewayRunner._final_text_for_post_turn_hooks(None, event).strip()
