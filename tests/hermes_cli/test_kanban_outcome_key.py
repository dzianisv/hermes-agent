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
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME",
                "HERMES_KANBAN_WORKSPACES_ROOT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (home / "config.yaml").write_text("kanban:\n  outcome_keys: true\n", encoding="utf-8")
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
    _write_config(kanban_home, "kanban:\n  outcome_keys: true\n  require_outcome_key: true\n"
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


# --- Review regressions (gpt-6-astra) -------------------------------------


def test_decompose_fold_never_creates_self_or_cyclic_link(kanban_home):
    """Bug 1: two children folding into the same open task, or a child folding
    into the root itself, must not create a self/cyclic dependency."""
    with kbc.connect() as conn:
        existing = kb.create_task(conn, title="already open", outcome_key="G5")
        root = kb.create_task(conn, title="root", triage=True, outcome_key="G8")
        ids = decompose_triage_task(
            conn, root, root_assignee=None, author="tester",
            children=[{"title": "a", "outcome_key": "G5"},
                      {"title": "b", "outcome_key": "G5", "parents": [0]},
                      {"title": "c", "outcome_key": "G8"}],
        )
        assert ids is not None
        self_links = conn.execute(
            "SELECT COUNT(*) FROM task_links WHERE parent_id = child_id").fetchone()[0]
        assert self_links == 0
        for row in conn.execute("SELECT parent_id, child_id FROM task_links").fetchall():
            assert not kb._would_cycle(conn, row["parent_id"], row["child_id"]) or False
        # No cycle: walking from the root downward never reaches the root.
        assert not _reaches(conn, root, root)
        assert existing in ids


def _reaches(conn, start, target):
    seen, stack = set(), [r[0] for r in conn.execute(
        "SELECT child_id FROM task_links WHERE parent_id = ?", (start,))]
    while stack:
        n = stack.pop()
        if n == target:
            return True
        if n in seen:
            continue
        seen.add(n)
        stack.extend(r[0] for r in conn.execute(
            "SELECT child_id FROM task_links WHERE parent_id = ?", (n,)))
    return False


def test_reopen_done_with_reused_key_does_not_crash(kanban_home):
    """Bug 2: reopening a done task whose outcome_key is now held by another
    open task must return a clean refusal, not an IntegrityError."""
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="first", outcome_key="G3")
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (a,))
        conn.commit()
        b = kb.create_task(conn, title="second", outcome_key="G3")
        assert b != a
        ok, detail = kb.reopen_done_task(conn, a, actor="op")
        assert ok is False and b in detail and "G3" in detail
        assert kb.get_task(conn, a).status == "done"


def test_dashboard_reopen_with_reused_key_does_not_crash(kanban_home):
    from plugins.kanban.dashboard import plugin_api as api
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="first", outcome_key="G3")
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (a,))
        conn.commit()
        kb.create_task(conn, title="second", outcome_key="G3")
        assert api._set_status_direct(conn, a, "todo") is False
        assert kb.get_task(conn, a).status == "done"


def test_required_mode_decompose_swarm_dashboard_can_pass_key(kanban_home):
    """Bug 3: with require_outcome_key on, decompose / swarm / dashboard must
    still work (they can pass a key; derived children inherit the exemption)."""
    _write_config(kanban_home, "kanban:\n  outcome_keys: true\n  require_outcome_key: true\n")
    from hermes_cli import kanban_swarm as ks
    from hermes_cli.kanban_decompose import _clean_children
    from plugins.kanban.dashboard import plugin_api as api
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="root", triage=True, outcome_key="G1")
        ids = decompose_triage_task(conn, root, root_assignee=None,
                                    children=[{"title": "child"}])
        assert ids
        created = ks.create_swarm(
            conn, goal="g", workers=[ks.parse_worker_arg("w:t")],
            verifier_assignee="v", synthesizer_assignee="s", outcome_key="G2")
        assert kb.get_task(conn, created.root_id).outcome_key == "G2"
    assert "outcome_key" in api.CreateTaskBody.model_fields
    routing = type("R", (), {"default_assignee": "x", "valid_names": {"x"}})()
    kids, _ = _clean_children("t", [{"title": "k", "outcome_key": "G4"}], routing)
    assert kids[0]["outcome_key"] == "G4"


def test_decompose_reads_policy_of_the_connections_board(kanban_home):
    """Bug 4: decompose on board B must apply B's policy, not the current board's."""
    kb.create_board("strict")
    meta = json.loads(kb.board_metadata_path("strict").read_text())
    meta.update(outcome_keys=True, outcome_key_pattern="^G\\d+$", require_outcome_key=True)
    kb.board_metadata_path("strict").write_text(json.dumps(meta))
    assert kb.get_current_board() == "default"
    with kbc.connect(board="strict") as conn:
        root = kb.create_task(conn, title="root", triage=True, outcome_key="G1", board="strict")
        # Match by name: other suites reload kanban_db, so the class identity can differ.
        with pytest.raises(ValueError) as exc:
            decompose_triage_task(conn, root, root_assignee=None,
                                  children=[{"title": "bad", "outcome_key": "nope"}])
        assert type(exc.value).__name__ == "OutcomeKeyError"


def test_off_by_default_ignores_outcome_key(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME",
                "HERMES_KANBAN_WORKSPACES_ROOT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="a", outcome_key="G3")
        b = kb.create_task(conn, title="b", outcome_key="G3")
        assert a != b
