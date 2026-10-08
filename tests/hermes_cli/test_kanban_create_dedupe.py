"""Contract: creating a card that duplicates an OPEN card returns the existing id.

Driven through the real ``hermes kanban create`` parser and the ``kanban_create``
tool registry on a real temp DB. Match rules: normalized title (lowercase,
whitespace collapsed, trailing @sha/#run/date stripped, leading prefix kept),
same idempotency key, or alert titles with the same PR number + check name.
``kanban.create_dedupe: false`` or ``--allow-duplicate`` / ``allow_duplicate``
bypasses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _cli(*argv):
    root = argparse.ArgumentParser(prog="hermes")
    kc.build_parser(root.add_subparsers())
    return kc._cmd_create(root.parse_args(["kanban", "create", *argv]))


def _tasks():
    with kbc.connect_closing() as conn:
        return kb.list_tasks(conn, include_archived=True)


def _comments(tid):
    with kbc.connect_closing() as conn:
        return [c.body for c in kb.list_comments(conn, tid)]


def _cli_id(capsys, *argv):
    assert _cli(*argv, "--json") == 0
    return json.loads(capsys.readouterr().out)


def test_cli_same_normalized_title_prints_exists_and_creates_nothing(kanban_home, capsys):
    first = _cli_id(capsys, "CI RED on main: unit tests @abc1234", "--body", "run 1 log")
    assert _cli("ci red on   MAIN: Unit tests @def5678", "--body", "run 2 log: new failure") == 0
    out = capsys.readouterr().out
    assert out.startswith(f"exists: {first['id']}")
    assert len(_tasks()) == 1
    # New body text appended as exactly one comment; repeating it adds nothing.
    assert _cli("CI RED on main: unit tests #991", "--body", "run 2 log: new failure") == 0
    capsys.readouterr()
    comments = _comments(first["id"])
    assert len(comments) == 1 and "run 2 log: new failure" in comments[0]


def test_cli_json_reports_created_false_and_prefix_is_kept(kanban_home, capsys):
    a = _cli_id(capsys, "CI RED on main: lint 2026-10-01")
    dup = _cli_id(capsys, "CI RED on main: lint 2026-10-07")
    assert dup["created"] is False and dup["duplicate_of"] == a["id"]
    other = _cli_id(capsys, "CI RED on release: lint")  # different prefix = different card
    assert other["id"] != a["id"]
    assert len(_tasks()) == 2


def test_cli_allow_duplicate_bypasses(kanban_home, capsys):
    a = _cli_id(capsys, "Fix flaky test")
    b = _cli_id(capsys, "Fix flaky test", "--allow-duplicate")
    assert a["id"] != b["id"] and len(_tasks()) == 2


def test_done_and_archived_cards_do_not_block_creation(kanban_home, capsys):
    a = _cli_id(capsys, "Rotate key")
    with kbc.connect_closing() as conn:
        kb.complete_task(conn, a["id"], result="ok")
    b = _cli_id(capsys, "Rotate key")
    assert b["id"] != a["id"]
    with kbc.connect_closing() as conn:
        kb.archive_task(conn, b["id"])
    c = _cli_id(capsys, "Rotate key")
    assert c["id"] not in (a["id"], b["id"])


def test_idempotency_key_and_alert_signature_match(kanban_home, capsys):
    a = _cli_id(capsys, "first wording", "--idempotency-key", "alert-42")
    assert _cli_id(capsys, "totally different wording", "--idempotency-key", "alert-42")["id"] == a["id"]
    pr = _cli_id(capsys, "PR #4714: check `canary-pr-control-plane` failed")
    again = _cli_id(capsys, "canary RED on PR #4714 — `canary-pr-control-plane` (attempt 3)")
    assert again["id"] == pr["id"] and again["dedupe_reason"] == "alert"
    other = _cli_id(capsys, "PR #4714: check `lint` failed")
    assert other["id"] != pr["id"]


def test_config_off_disables_dedupe(kanban_home, capsys):
    (kanban_home / "config.yaml").write_text("kanban:\n  create_dedupe: false\n")
    a = _cli_id(capsys, "Same title")
    b = _cli_id(capsys, "Same title")
    assert a["id"] != b["id"]


def test_default_config_has_create_dedupe_on():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["kanban"]["create_dedupe"] is True


def test_tool_registry_returns_existing_id(kanban_home, monkeypatch):
    from tools.registry import registry
    import tools.kanban_tools  # noqa: F401  (registers)

    first = json.loads(registry.dispatch("kanban_create", {
        "title": "Deploy stuck", "assignee": "eng", "body": "a"}))
    assert first["ok"] and first["created"] is True
    dup = json.loads(registry.dispatch("kanban_create", {
        "title": "deploy  stuck @0123abcd", "assignee": "eng", "body": "b: extra detail"}))
    assert dup["created"] is False
    assert dup["id"] == dup["duplicate_of"] == first["task_id"]
    assert any("b: extra detail" in c for c in _comments(first["task_id"]))
    forced = json.loads(registry.dispatch("kanban_create", {
        "title": "Deploy stuck", "assignee": "eng", "allow_duplicate": True}))
    assert forced["created"] is True and forced["task_id"] != first["task_id"]


@pytest.mark.parametrize("raw,norm", [
    ("CI RED on main: build @a1b2c3d4", "ci red on main: build"),
    ("Nightly  failed 2026-10-07T12:00Z", "nightly failed"),
    ("Watchdog red #1234", "watchdog red"),
    ("#42", "#42"),
    ("pipeline: merge-gate PR #5232", "pipeline: merge-gate pr #5232"),
    ("Review PR #12 @abcdef12", "review pr #12"),
])
def test_normalize_card_title(raw, norm):
    assert kb.normalize_card_title(raw) == norm
