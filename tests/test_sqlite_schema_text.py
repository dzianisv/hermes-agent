"""Every column of every shipped SQLite table must stay droppable.

Behaviour under test, not shape of the source. SQLite stores the *verbatim*
``CREATE TABLE`` text in ``sqlite_schema`` and re-parses that stored text
during ``ALTER TABLE ... DROP COLUMN``. Builds older than 3.46 (hosted CI runs
3.43.x) choke on a ``--`` line comment carrying a comma in the trailing column
region — ``error in table <t> after drop column: incomplete input`` — which
aborted the kanban ``scheduled_wake_at`` legacy migration in CI while passing
on a newer local SQLite.

The guard therefore builds the databases **the way production builds them**
(``kanban_db.connect`` / ``projects_db.connect`` / ``SessionDB``), then works
off the live catalog:

* every table and every column is discovered from the built database, so a new
  table or column is covered the moment it ships — nothing is hand-listed;
* each column is dropped on a fresh clone of that database;
* the text SQLite actually stored is asserted to be free of ``--`` comments,
  which fails on *any* SQLite version and is what catches a call site that
  stops normalising its schema text.

A drop that SQLite refuses for a structural reason (the column is indexed, is
part of a foreign key, is the primary key) is NOT this defect — those refusals
also contain the literal "after drop column" — so refusals are classified
against the live catalog and against SQLite's parser diagnostics rather than on
the presence of that substring.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Callable, Dict, List

import pytest

from sqlite_schema_text import strip_sql_line_comments

# SQLite parser diagnostics. When one of these comes back from DROP COLUMN it
# means SQLite could not re-parse its own stored schema text — the defect this
# guard exists for — as opposed to a structural refusal, which names a real
# dependency on the column instead.
_PARSE_DIAGNOSTICS = (
    "incomplete input",
    "syntax error",
    "unrecognized token",
    "malformed",
)


# ── Production database builders ────────────────────────────────────────────
# Each builder creates a real database through the code path the product uses
# at runtime, inside the per-test HERMES_HOME sandbox.


def _build_kanban_db(tmp_path: Path) -> Path:
    from hermes_cli import kanban_db

    path = tmp_path / "kanban.db"
    conn = kanban_db.connect(path)
    conn.close()
    return path


def _build_projects_db(tmp_path: Path) -> Path:
    from hermes_cli import projects_db

    path = tmp_path / "projects.db"
    conn = projects_db.connect(path)
    conn.close()
    return path


def _build_state_db(tmp_path: Path) -> Path:
    from hermes_state import SessionDB

    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    db.close()
    return path


BUILDERS: Dict[str, Callable[[Path], Path]] = {
    "kanban.db": _build_kanban_db,
    "projects.db": _build_projects_db,
    "state.db": _build_state_db,
}


@pytest.fixture(params=sorted(BUILDERS), ids=sorted(BUILDERS))
def built_db(request, tmp_path: Path) -> Path:
    """A real database, created through the production connect path."""
    return BUILDERS[request.param](tmp_path / request.param.replace(".", "_"))


# ── Helpers over the live catalog ───────────────────────────────────────────


def _clone(path: Path) -> sqlite3.Connection:
    """An in-memory copy that preserves the verbatim stored schema text."""
    src = sqlite3.connect(str(path))
    dst = sqlite3.connect(":memory:")
    try:
        src.backup(dst)
    finally:
        src.close()
    return dst


def _ordinary_tables(conn: sqlite3.Connection) -> List[str]:
    """Real tables only — the ones ALTER TABLE DROP COLUMN actually applies to.

    Virtual tables have no DROP COLUMN, and their shadow tables (``<vtab>_data``,
    ``<vtab>_config``, …) are private to the module that owns them: SQLite
    refuses to alter them because the virtual table constructor re-runs, not
    because of anything in our schema text.
    """
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    virtual = [
        name
        for name, sql in rows
        if sql and sql.lstrip().upper().startswith("CREATE VIRTUAL")
    ]
    out = []
    for name, sql in rows:
        if name in virtual:
            continue
        if any(name.startswith(f"{vtab}_") for vtab in virtual):
            continue
        out.append(name)
    return out


def _mentions_column(sql: str, column: str) -> bool:
    """Whether a stored SQL definition references ``column`` as an identifier."""
    import re

    return re.search(rf"\b{re.escape(column)}\b", sql or "") is not None


def _structural_dependencies(
    conn: sqlite3.Connection, table: str, column: str
) -> List[str]:
    """Reasons SQLite may legitimately refuse to drop ``table.column``.

    Everything here is read from SQLite's own catalog, so a new index or
    foreign key is accounted for automatically.
    """
    reasons: List[str] = []

    columns = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
    if len(columns) <= 1:
        # SQLite refuses to leave a table with no columns at all.
        reasons.append("sole column of the table")

    for row in conn.execute(f'PRAGMA table_info("{table}")'):
        if row[1] == column and row[5]:
            reasons.append("primary key")

    for idx_row in conn.execute(f'PRAGMA index_list("{table}")'):
        index = idx_row[1]
        indexed = {r[2] for r in conn.execute(f'PRAGMA index_info("{index}")')}
        if column in indexed:
            reasons.append(f"indexed by {index}")
            continue
        # Partial / expression indexes reference columns only in their SQL.
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (index,)
        ).fetchone()
        if row and row[0] and _mentions_column(row[0], column):
            reasons.append(f"referenced by index {index}")

    for fk in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
        if fk[3] == column:
            reasons.append(f"foreign key to {fk[2]}({fk[4]})")

    for other in _ordinary_tables(conn):
        for fk in conn.execute(f'PRAGMA foreign_key_list("{other}")'):
            if fk[2] == table and fk[4] == column:
                reasons.append(f"referenced by foreign key from {other}")

    for kind in ("view", "trigger"):
        for (name, sql) in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type=?", (kind,)
        ):
            if sql and _mentions_column(sql, table) and _mentions_column(sql, column):
                reasons.append(f"referenced by {kind} {name}")

    # A generated column or CHECK constraint lives only in the table's own SQL.
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if row and row[0]:
        upper = row[0].upper()
        if "CHECK" in upper or "GENERATED" in upper or "AS (" in upper:
            reasons.append("table has CHECK/generated-column expressions")

    return reasons


def _defect_reason(
    pristine: sqlite3.Connection, table: str, column: str, exc: Exception
) -> str | None:
    """Classify a DROP COLUMN failure. Returns None when it is legitimate.

    Two independent signals mark the re-parse corruption:

    1. SQLite answered with a *parser* diagnostic — it could not read back the
       schema text it had stored. A structural refusal never does this; it
       names the column ("no such column: source", "unknown column ... in
       foreign key definition").
    2. SQLite refused although the live catalog shows nothing depending on the
       column — an unexplained refusal is not something we accept either.
    """
    message = str(exc)
    tail = message
    if "after drop column:" in message:
        tail = message.split("after drop column:", 1)[1].strip()
    lowered = tail.lower()
    if any(diag in lowered for diag in _PARSE_DIAGNOSTICS):
        return f"stored schema text did not re-parse -> {message}"
    deps = _structural_dependencies(pristine, table, column)
    if not deps:
        return f"refused with no dependency in the live catalog -> {message}"
    return None


# ── The guard ───────────────────────────────────────────────────────────────


def test_every_column_of_a_production_database_can_be_dropped(built_db: Path) -> None:
    """Drop every column of every table of a database production just built."""
    pristine = _clone(built_db)
    try:
        plan = {
            table: [r[1] for r in pristine.execute(f'PRAGMA table_info("{table}")')]
            for table in _ordinary_tables(pristine)
        }
        assert plan, f"{built_db.name} came up with no ordinary tables"

        failures: List[str] = []
        for table, columns in plan.items():
            for column in columns:
                conn = _clone(built_db)
                try:
                    conn.execute(f'ALTER TABLE "{table}" DROP COLUMN "{column}"')
                except sqlite3.OperationalError as exc:
                    reason = _defect_reason(pristine, table, column, exc)
                    if reason:
                        failures.append(f"{table}.{column}: {reason}")
                finally:
                    conn.close()
    finally:
        pristine.close()

    assert not failures, (
        f"{built_db.name}: SQLite {sqlite3.sqlite_version} cannot re-parse the "
        "schema text this database stored. Offenders:\n  " + "\n  ".join(failures)
    )


def test_production_database_stores_no_line_comments(built_db: Path) -> None:
    """What SQLite stored — and will re-parse on DROP COLUMN — is comment-free.

    Version-independent companion to the drop test above: a call site that
    stops normalising its schema text fails here on every SQLite, including the
    ones new enough to tolerate the comment.
    """
    conn = _clone(built_db)
    try:
        offenders = [
            f"{kind} {name}"
            for kind, name, sql in conn.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL"
            )
            if strip_sql_line_comments(sql) != sql
        ]
    finally:
        conn.close()
    assert not offenders, (
        f"{built_db.name}: '--' comments reached sqlite_schema via {offenders}. "
        "Pre-3.46 SQLite re-parses this text on ALTER TABLE ... DROP COLUMN."
    )


def test_every_schema_constant_in_the_import_graph_is_materialised(
    tmp_path: Path,
) -> None:
    """Coverage is derived: no ``SCHEMA_SQL`` owner can sit outside this guard.

    After the builders have run, every loaded module that owns a ``SCHEMA_SQL``
    string must have its tables present in at least one of the databases under
    test. A newly added schema owner that nothing here materialises fails.
    """
    live_tables: set[str] = set()
    for name, builder in BUILDERS.items():
        conn = _clone(builder(tmp_path / name.replace(".", "_")))
        try:
            live_tables.update(_ordinary_tables(conn))
        finally:
            conn.close()

    owners = {
        name: module.SCHEMA_SQL
        for name, module in list(sys.modules.items())
        if isinstance(getattr(module, "SCHEMA_SQL", None), str)
    }
    assert owners, "no SCHEMA_SQL owner is even imported — builders did not run?"

    uncovered: List[str] = []
    for owner, schema_sql in owners.items():
        scratch = sqlite3.connect(":memory:")
        try:
            scratch.executescript(strip_sql_line_comments(schema_sql))
            declared = set(_ordinary_tables(scratch))
        finally:
            scratch.close()
        missing = declared - live_tables
        if missing:
            uncovered.append(f"{owner}: {sorted(missing)}")

    assert not uncovered, (
        "SCHEMA_SQL tables that no database under test materialises "
        f"(add a builder): {uncovered}"
    )


def test_a_comma_bearing_line_comment_is_what_breaks_drop_column() -> None:
    """Pin the SQLite behaviour this guard exists for, on the running build."""
    raw = (
        "CREATE TABLE t (\n"
        "    a INTEGER,\n"
        "    -- note, with a comma\n"
        "    b INTEGER\n"
        ");"
    )

    safe = sqlite3.connect(":memory:")
    try:
        safe.executescript(strip_sql_line_comments(raw))
        safe.execute("ALTER TABLE t DROP COLUMN b")
        assert [r[1] for r in safe.execute("PRAGMA table_info(t)")] == ["a"]
    finally:
        safe.close()

    if sqlite3.sqlite_version_info < (3, 46):
        unsafe = sqlite3.connect(":memory:")
        try:
            unsafe.executescript(raw)
            with pytest.raises(sqlite3.OperationalError, match="after drop column"):
                unsafe.execute("ALTER TABLE t DROP COLUMN b")
        finally:
            unsafe.close()


def test_strip_preserves_double_dash_inside_string_literals() -> None:
    sql = "CREATE TABLE t (a TEXT NOT NULL DEFAULT '-- not, a comment');"
    stripped = strip_sql_line_comments(sql)
    assert "-- not, a comment" in stripped
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(stripped)
        conn.execute("INSERT INTO t DEFAULT VALUES")
        assert conn.execute("SELECT a FROM t").fetchone()[0] == "-- not, a comment"
    finally:
        conn.close()
