"""`hermes pause` must never turn a stream of blocked inbound events into a message flood.

Regression: with ESTOP engaged, every blocked event got the user-facing "Hermes is paused" reply;
a 5s event source produced ~10,000 Telegram messages and a ~6h Telegram flood ban (twice)."""

import asyncio

import pytest

from agent import estop
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _WireAdapter(BasePlatformAdapter):
    """Real adapter lifecycle with an in-memory wire transport."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content))
        return SendResult(success=True, message_id="wire-1")


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    adapter = _WireAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._is_user_authorized = lambda source: True
    adapter.set_message_handler(runner._handle_message)

    async def deliver(chat_id, times=1, thread_id=None):
        source = SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, user_id="u1", chat_type="dm",
                               thread_id=thread_id)
        for _ in range(times):
            await adapter.handle_message(MessageEvent(text="are you there?", source=source))
            await asyncio.gather(*adapter._background_tasks)

    yield adapter, deliver, tmp_path
    estop.disengage()


@pytest.mark.asyncio
async def test_paused_notice_once_per_chat_per_pause(gateway):
    adapter, deliver, _ = gateway
    estop.engage(reason="maintenance")
    await deliver("42", times=6)
    await deliver("43", times=3)
    await deliver("42", times=2, thread_id="7")
    assert [chat for chat, _ in adapter.sent] == ["42", "43", "42"]
    assert "maintenance" in adapter.sent[0][1]

    estop.disengage()
    estop.engage(reason="second pause")
    await deliver("42", times=4)
    assert [chat for chat, _ in adapter.sent] == ["42", "43", "42", "42"]
    assert "second pause" in adapter.sent[-1][1]


@pytest.mark.asyncio
async def test_notice_interval_is_read_from_config_yaml(gateway):
    adapter, deliver, home = gateway
    (home / "config.yaml").write_text("gateway:\n  estop_notice_interval_seconds: 1\n", encoding="utf-8")
    estop.engage()
    await deliver("42", times=3)
    assert len(adapter.sent) == 1
    await asyncio.sleep(1.5)
    await deliver("42", times=3)
    assert len(adapter.sent) == 2
