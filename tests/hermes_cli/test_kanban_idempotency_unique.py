"""Contract: at most one OPEN task per idempotency_key, enforced by the partial
UNIQUE index ``uq_tasks_open_idempotency_key`` -- not by SELECT-then-INSERT.

Real temp sqlite file, real processes/threads racing ``create_task``; no mocks.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _open_rows(key):
    with kbc.connect_closing() as conn:
        return conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? AND status NOT IN ('done','archived')",
            (key,),
        ).fetchall()


def _proc_create(home, key, idx, barrier, out):
    os.environ["HERMES_HOME"] = home
    os.environ["HERMES_KANBAN_HOME"] = home
    from hermes_cli import kanban_db as kb2
    from hermes_cli import kanban_db_connect as kbc2
    barrier.wait()
    with kbc2.connect_closing() as conn:
        tid = kb2.create_task(conn, title=f"racer {idx}", idempotency_key=key, allow_duplicate=True)
    out.put(str(tid))


def test_index_exists_on_fresh_db(kanban_home):
    with kbc.connect_closing() as conn:
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'uq_tasks_open_idempotency_key'"
        ).fetchone()
    assert sql and "UNIQUE" in sql[0] and "done" in sql[0] and "archived" in sql[0]


def test_concurrent_processes_same_key_one_row(kanban_home):
    n = 8
    ctx = mp.get_context("spawn")
    barrier, out = ctx.Barrier(n), ctx.Queue()
    procs = [ctx.Process(target=_proc_create, args=(str(kanban_home), "ci-red:o/r:ci", i, barrier, out))
             for i in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    ids = {out.get(timeout=5) for _ in range(n)}
    rows = _open_rows("ci-red:o/r:ci")
    assert len(rows) == 1
    assert ids == {rows[0][0]}


def test_concurrent_threads_same_key_one_row(kanban_home):
    n, ids, errs = 10, [], []
    barrier = threading.Barrier(n)

    def run(i):
        try:
            barrier.wait()
            with kbc.connect_closing() as conn:
                ids.append(str(kb.create_task(conn, title=f"t{i}", idempotency_key="k-thr",
                                              allow_duplicate=True)))
        except Exception as e:  # pragma: no cover - surfaced by assert
            errs.append(e)

    ts = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    [t.start() for t in ts]
    [t.join(60) for t in ts]
    assert not errs
    assert len(_open_rows("k-thr")) == 1
    assert set(ids) == {_open_rows("k-thr")[0][0]}


def test_raw_duplicate_insert_rejected_and_done_frees_key(kanban_home):
    with kbc.connect_closing() as conn:
        a = kb.create_task(conn, title="a", idempotency_key="k1")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO tasks (id, title, status, created_at, idempotency_key) "
                         "VALUES ('t_raw', 'x', 'ready', 0, 'k1')")
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (a,))
        b = kb.create_task(conn, title="b", idempotency_key="k1")
    assert b != a and b.created


def test_migration_adds_index_to_legacy_db(kanban_home):
    db = kb.kanban_db_path()
    raw = sqlite3.connect(db)
    raw.execute("DROP INDEX uq_tasks_open_idempotency_key")
    raw.commit()
    raw.close()
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    raw = sqlite3.connect(db)
    assert raw.execute("SELECT 1 FROM sqlite_master WHERE name='uq_tasks_open_idempotency_key'").fetchone()
    raw.close()
