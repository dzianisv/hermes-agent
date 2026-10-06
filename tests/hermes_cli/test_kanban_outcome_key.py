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


# --- critic regressions (cycles, reopen conflicts, producers, per-board policy) ---


def _strict_board(slug="strict", pattern=None):
    kb.create_board(slug)
    p = kb.board_metadata_path(slug)
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["require_outcome_key"] = True
    if pattern:
        meta["outcome_key_pattern"] = pattern
    p.write_text(json.dumps(meta), encoding="utf-8")
    return slug


def _has_cycle(conn):
    edges = conn.execute("SELECT parent_id, child_id FROM task_links").fetchall()
    if any(p == c for p, c in edges):
        return True
    adj = {}
    for p, c in edges:
        adj.setdefault(p, []).append(c)
    state = {}

    def visit(n):
        state[n] = 1
        for m in adj.get(n, []):
            if state.get(m) == 1 or (m not in state and visit(m)):
                return True
        state[n] = 2
        return False

    return any(n not in state and visit(n) for n in list(adj))


def test_decompose_fold_never_creates_cycle(kanban_home):
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="root", triage=True, outcome_key="R1")
        with pytest.raises(ValueError, match="cycl|itself|same task"):
            decompose_triage_task(conn, root, root_assignee=None, author="t",
                                  children=[{"title": "child is root", "outcome_key": "R1"}])
        assert not _has_cycle(conn)
        assert kb.get_task(conn, root).status == "triage"
        # Two dependent siblings collapsing onto one task must not self-loop.
        root2 = kb.create_task(conn, title="root2", triage=True)
        with pytest.raises(ValueError, match="cycl|itself|same task"):
            decompose_triage_task(conn, root2, root_assignee=None, author="t", children=[
                {"title": "a", "outcome_key": "S1"},
                {"title": "b", "outcome_key": "S1", "parents": [0]},
            ])
        assert not _has_cycle(conn)
        assert _count(conn, "S1") == 0


def test_decompose_fold_rejects_cycle_through_existing_links(kanban_home):
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="root", triage=True)
        # Existing open task that already depends on the root.
        dep = kb.create_task(conn, title="downstream", parents=[root], outcome_key="D1")
        with pytest.raises(ValueError, match="cycl"):
            decompose_triage_task(conn, root, root_assignee=None, author="t",
                                  children=[{"title": "folds into downstream", "outcome_key": "D1"}])
        assert not _has_cycle(conn)
        assert dep


def test_reopen_done_with_reused_key_returns_owner(kanban_home):
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="A", outcome_key="G3")
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (a,))
        conn.commit()
        b = kb.create_task(conn, title="B", outcome_key="G3")
        ok, detail = kb.reopen_done_task(conn, a, actor="op")
        assert ok is False
        assert b in detail and "G3" in detail
        assert kb.get_task(conn, a).status == "done"


def test_reopen_unkeyed_ancestor_with_reused_descendant_key(kanban_home):
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="P")
        child = kb.create_task(conn, title="C", parents=[parent], outcome_key="G8")
        conn.execute("UPDATE tasks SET status='done' WHERE id IN (?, ?)", (parent, child))
        conn.commit()
        owner = kb.create_task(conn, title="C again", outcome_key="G8")
        ok, detail = kb.reopen_done_task(conn, parent, actor="op")
        assert ok is False and owner in detail and child in detail
        assert kb.get_task(conn, parent).status == "done"
        assert kb.get_task(conn, child).status == "done"


def test_dashboard_create_carries_outcome_key_on_strict_board(kanban_home):
    import importlib.util, sys
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    plugin = Path(__file__).resolve().parents[2] / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("kanban_plugin_outcome_test", plugin)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    app = FastAPI()
    app.include_router(mod.router, prefix="/k")
    client = TestClient(app)
    slug = _strict_board()
    r = client.post(f"/k/tasks?board={slug}", json={"title": "x", "outcome_key": "G1"})
    assert r.status_code == 200, r.text
    assert r.json()["task"]["outcome_key"] == "G1"
    r2 = client.post(f"/k/tasks?board={slug}", json={"title": "y", "outcome_key": "G1"})
    assert r2.status_code == 200 and r2.json()["task"]["id"] == r.json()["task"]["id"]
    assert r2.json().get("deduped") is True
    assert client.post(f"/k/tasks?board={slug}", json={"title": "nokey"}).status_code == 400


def test_swarm_works_on_strict_board(kanban_home, monkeypatch):
    from hermes_cli import kanban_swarm as ks
    _write_config(kanban_home, "kanban:\n  require_outcome_key: true\n")
    with kbc.connect() as conn:
        created = ks.create_swarm(
            conn, goal="ship", outcome_key="G10",
            workers=[ks.SwarmWorkerSpec(profile="a", title="A", body="A", outcome_key="G10-a"),
                     ks.SwarmWorkerSpec(profile="b", title="B", body="B")],
            verifier_assignee="v", synthesizer_assignee="s",
        )
        keys = {kb.get_task(conn, t).outcome_key for t in
                [created.root_id, *created.worker_ids, created.verifier_id, created.synthesizer_id]}
        assert None not in keys and len(keys) == 5 and "G10" in keys and "G10-a" in keys


def test_decomposer_propagates_outcome_key(kanban_home):
    from hermes_cli import kanban_decompose as decomp
    routing = decomp._Routing(orchestrator="o", default_assignee="o", auto_promote=False,
                              roster=[], valid_names={"o"})
    children, reason = decomp._clean_children("t", [
        {"title": "a", "outcome_key": "G1"}, {"title": "b", "outcome_key": "  "}], routing)
    assert not reason
    assert children[0]["outcome_key"] == "G1"
    assert children[1].get("outcome_key") in (None, "")
    _write_config(kanban_home, "kanban:\n  require_outcome_key: true\n")
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="root", triage=True, outcome_key="G0")
    out = decomp._apply_fanout(root, {"tasks": [{"title": "a", "outcome_key": "G1"}]}, routing, "me")
    assert out.ok, out.reason
    with kbc.connect() as conn:
        assert kb.get_task(conn, out.child_ids[0]).outcome_key == "G1"


def test_decompose_uses_target_board_policy(kanban_home):
    slug = _strict_board()
    assert kb.get_current_board() == kb.DEFAULT_BOARD
    conn = kbc.connect(board=slug)
    try:
        root = kb.create_task(conn, title="root", triage=True, outcome_key="G0", board=slug)
        with pytest.raises(kb.OutcomeKeyError):
            decompose_triage_task(conn, root, root_assignee=None, author="t",
                                  children=[{"title": "unkeyed"}])
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
        # create_task without an explicit board also follows the connection.
        with pytest.raises(kb.OutcomeKeyError):
            kb.create_task(conn, title="unkeyed direct")
    finally:
        conn.close()


def test_migration_with_duplicate_open_keys_names_tasks(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db = tmp_path / "legacy.db"
    raw = sqlite3.connect(db)
    raw.row_factory = sqlite3.Row
    raw.executescript(kb.SCHEMA_SQL)
    kbc._migrate_add_optional_columns(raw)
    raw.execute("DROP INDEX uq_tasks_open_outcome_key")
    for tid in ("t_dup1", "t_dup2"):
        raw.execute("INSERT INTO tasks (id, title, status, created_at, outcome_key) "
                    "VALUES (?, 'x', 'ready', 0, 'G3')", (tid,))
    raw.commit()
    with pytest.raises(RuntimeError) as ei:
        kbc._migrate_add_optional_columns(raw)
    msg = str(ei.value)
    assert "t_dup1" in msg and "t_dup2" in msg and "G3" in msg
    assert raw.execute("SELECT COUNT(*) FROM tasks WHERE outcome_key='G3'").fetchone()[0] == 2
    raw.close()
