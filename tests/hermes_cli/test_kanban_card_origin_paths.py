"""Every card-creation path records where the card's events must go (Notion harness #33).

Contract: chat session -> that chat/thread; worker inside a card run -> the creating card's
origin; cron job -> the job's deliver target; plain CLI -> no origin (never a guess).
"""
import json

import pytest


@pytest.fixture
def board(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("HERMES_KANBAN_TASK", "HERMES_SESSION_KEY", "HERMES_SESSION_PLATFORM",
                 "HERMES_SESSION_CHAT_ID", "HERMES_CRON_SESSION"):
        monkeypatch.delenv(name, raising=False)
    kb.init_db()
    from gateway.session_context import reset_session_vars
    reset_session_vars()
    yield
    reset_session_vars()


def _create_via_tool(**args):
    from tools.kanban_tools import _handle_create
    out = _handle_create({"title": "card", "assignee": "w", **args})
    return json.loads(out)


def _routes(task_id):
    from hermes_cli import kanban_db_connect as kbc, kanban_db_notify as kbn
    with kbc.connect_closing() as conn:
        return [(s["platform"], s["chat_id"], s["thread_id"], s["delivery_mode"])
                for s in kbn.list_notify_subs(conn, task_id)]


def test_chat_session_card_routes_to_its_thread(board):
    from gateway.session_context import set_session_vars
    set_session_vars(platform="telegram", chat_id="111", chat_type="dm", thread_id="X1",
                     user_id="111", session_key="agent:main:telegram:dm:111:X1")
    tid = _create_via_tool()["task_id"]
    assert _routes(tid) == [("telegram", "111", "X1", "notify+wake")]


def test_cron_created_card_routes_to_job_deliver_target(board):
    from gateway.session_context import _VAR_MAP, set_session_vars
    set_session_vars(cron_session="1")  # cron clears HERMES_SESSION_* routing
    _VAR_MAP["HERMES_CRON_AUTO_DELIVER_PLATFORM"].set("telegram")
    _VAR_MAP["HERMES_CRON_AUTO_DELIVER_CHAT_ID"].set("111")
    _VAR_MAP["HERMES_CRON_AUTO_DELIVER_THREAD_ID"].set("CRON7")
    tid = _create_via_tool()["task_id"]
    assert _routes(tid) == [("telegram", "111", "CRON7", "notify+wake")]


def test_cron_with_local_delivery_records_no_origin(board):
    from gateway.session_context import set_session_vars
    set_session_vars(cron_session="1")
    assert _routes(_create_via_tool()["task_id"]) == []


def test_worker_child_inherits_creating_cards_origin(board, monkeypatch):
    from gateway.session_context import set_session_vars
    set_session_vars(platform="telegram", chat_id="111", chat_type="dm", thread_id="X1")
    parent = _create_via_tool()["task_id"]
    # Worker run of `parent`: no chat context, only the dispatcher's task binding.
    from gateway.session_context import reset_session_vars
    reset_session_vars()
    monkeypatch.setenv("HERMES_KANBAN_TASK", parent)
    monkeypatch.setattr("tools.kanban_tools._is_dispatcher_owned_worker", lambda: True)
    child = _create_via_tool(title="child")["task_id"]
    assert _routes(child) == [("telegram", "111", "X1", "notify+wake")]


def test_plain_cli_card_has_no_origin(board, capsys):
    import argparse
    from hermes_cli.kanban import kanban_command
    from hermes_cli.kanban_parser import build_parser
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers())
    assert kanban_command(parser.parse_args(["kanban", "create", "cli card", "--json"])) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["subscribed"] is False
    assert _routes(out["id"]) == []
