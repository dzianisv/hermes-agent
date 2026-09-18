"""Schema-text normalisation shared by every ``executescript(SCHEMA_SQL)`` site.

Why this exists — a real, reproduced SQLite defect, not style:

SQLite stores the *verbatim* ``CREATE TABLE`` text in ``sqlite_schema`` and
re-parses that stored text during ``ALTER TABLE ... DROP COLUMN``. In SQLite
older than 3.46 (the hosted CI runners ship 3.43.x with CPython 3.11) that
re-parse mishandles a ``--`` line comment that sits in the trailing column
region of the definition when the comment body contains a comma: the drop
fails with ``error in table <t> after drop column: incomplete input`` and the
migration aborts.

Reproduced on sqlite 3.43.2::

    CREATE TABLE tasks (a INTEGER,
        -- note, with comma
        scheduled_wake_at INTEGER);
    ALTER TABLE tasks DROP COLUMN scheduled_wake_at;
    -- OperationalError: error in table tasks after drop column: incomplete input

The fix is to never persist ``--`` comments into ``sqlite_schema`` at all:
comments are documentation for humans reading the Python source, and they are
stripped on the way to ``executescript``. The constants keep their prose; the
stored schema text becomes comment-free, which makes DROP COLUMN safe on every
supported SQLite.

``tests/test_sqlite_schema_text.py`` derives its assertions from the live
SCHEMA_SQL constants (it enumerates them, it does not carry a hand-written
list), so a future schema that reintroduces a comment is caught.
"""

from __future__ import annotations

__all__ = ["strip_sql_line_comments"]


def strip_sql_line_comments(sql: str) -> str:
    """Remove ``--`` line comments from ``sql``, preserving string literals.

    A ``--`` inside a single- or double-quoted SQL literal is data, not a
    comment, and is left untouched. Lines that consisted only of a comment are
    dropped entirely; a trailing comment leaves its code prefix in place with
    trailing whitespace trimmed.
    """
    out_lines: list[str] = []
    in_single = False
    in_double = False
    for line in sql.split("\n"):
        cut = None
        i = 0
        n = len(line)
        while i < n:
            ch = line[i]
            if in_single:
                if ch == "'":
                    # '' is an escaped quote inside a literal.
                    if i + 1 < n and line[i + 1] == "'":
                        i += 1
                    else:
                        in_single = False
            elif in_double:
                if ch == '"':
                    if i + 1 < n and line[i + 1] == '"':
                        i += 1
                    else:
                        in_double = False
            elif ch == "'":
                in_single = True
            elif ch == '"':
                in_double = True
            elif ch == "-" and i + 1 < n and line[i + 1] == "-":
                cut = i
                break
            i += 1
        if cut is None:
            out_lines.append(line)
            continue
        head = line[:cut].rstrip()
        if head:
            out_lines.append(head)
        # A comment-only line is dropped so the stored schema stays compact.
    return "\n".join(out_lines)
