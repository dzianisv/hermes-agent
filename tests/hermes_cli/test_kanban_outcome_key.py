"""outcome_key: one OPEN task per (project, key), enforced in the DB.

Real sqlite temp DB, no mocks.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_graph import decompose_triage_task


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _count(conn, key):
    return conn.execute("SELECT COUNT(*) FROM tasks WHERE outcome_key = ?", (key,)).fetchone()[0]


def test_duplicate_create_folds_with_comment(kanban_home):
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="Fix G3 login", outcome_key="G3")
        assert a.deduped is False
        b = kb.create_task(conn, title="G3 login broken again", body="seen twice", outcome_key="G3")
        assert b == a and b.deduped is True
        assert _count(conn, "G3") == 1
        bodies = [c.body for c in kb.list_comments(conn, a)]
        assert any("G3 login broken again" in x and "seen twice" in x for x in bodies)
        assert kb.get_task(conn, a).outcome_key == "G3"
        # Different key / no key are unaffected.
        assert kb.create_task(conn, title="other", outcome_key="G4") != a
        assert kb.create_task(conn, title="plain") != kb.create_task(conn, title="plain")


def test_index_rejects_raw_duplicate_open_insert(kanban_home):
    with kbc.connect() as conn:
        kb.create_task(conn, title="x", outcome_key="G9")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO tasks (id, title, status, created_at, outcome_key) "
                "VALUES ('t_raw', 'dup', 'ready', 0, 'G9')"
            )


def test_closed_task_frees_key(kanban_home):
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="first", outcome_key="G3")
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (a,))
        conn.commit()
        b = kb.create_task(conn, title="second", outcome_key="G3")
        assert b != a and b.deduped is False
        assert kb.archive_task(conn, b) or True
        c = kb.create_task(conn, title="third", outcome_key="G3")
        assert c not in (a, b)


def test_concurrent_creates_produce_one_task(kanban_home):
    n = 8
    barrier = threading.Barrier(n)
    results, errors = [], []

    def worker(i):
        try:
            with kbc.connect() as conn:
                barrier.wait()
                results.append(kb.create_task(conn, title=f"race {i}", outcome_key="G7"))
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(set(results)) == 1
    assert sum(1 for r in results if not r.deduped) == 1
    with kbc.connect() as conn:
        assert _count(conn, "G7") == 1


def test_decompose_path_subject_to_rule(kanban_home):
    with kbc.connect() as conn:
        existing = kb.create_task(conn, title="already open", outcome_key="G5")
        root = kb.create_task(conn, title="root", triage=True)
        ids = decompose_triage_task(
            conn, root, root_assignee=None, author="tester",
            children=[{"title": "dup child", "outcome_key": "G5"},
                      {"title": "new child", "outcome_key": "G6"}],
        )
        assert ids[0] == existing
        assert _count(conn, "G5") == 1 and _count(conn, "G6") == 1
        assert any("dup child" in c.body for c in kb.list_comments(conn, existing))


def _write_config(home, text):
    (home / "config.yaml").write_text(text, encoding="utf-8")


def test_require_flag_rejects_missing_key(kanban_home):
    _write_config(kanban_home, "kanban:\n  require_outcome_key: true\n"
                               "  outcome_key_pattern: '^G\\d+$|^den:'\n")
    with kbc.connect() as conn:
        with pytest.raises(kb.OutcomeKeyError, match="required"):
            kb.create_task(conn, title="no key")
        with pytest.raises(kb.OutcomeKeyError, match="pattern"):
            kb.create_task(conn, title="bad key", outcome_key="X1")
        assert kb.create_task(conn, title="ok", outcome_key="G1")
        assert kb.create_task(conn, title="ok2", outcome_key="den:abc")


def test_default_does_not_require_key(kanban_home):
    with kbc.connect() as conn:
        assert kb.create_task(conn, title="no key, fine")


def test_cli_reports_deduped(kanban_home, capsys):
    import argparse
    from hermes_cli import kanban as kc
    kanban_command = kc.kanban_command

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args(["kanban", "create", "first", "--outcome-key", "G3"])
    assert kanban_command(args) == 0
    first = capsys.readouterr().out
    args = parser.parse_args(["kanban", "create", "second", "--outcome-key", "G3"])
    assert kanban_command(args) == 0
    out = capsys.readouterr().out
    tid = first.split()[1]
    assert f"deduped into {tid}" in out


def test_tool_reports_deduped(kanban_home):
    from tools import kanban_tools as kt
    a = json.loads(kt._handle_create({"title": "t", "assignee": "x", "outcome_key": "G2"}))
    b = json.loads(kt._handle_create({"title": "t2", "assignee": "x", "outcome_key": "G2"}))
    assert b.get("deduped") is True and b["task_id"] == a["task_id"]
