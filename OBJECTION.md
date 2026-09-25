# Objection: host cap overshoot is not a WAL reader fail-open

No enforcement change was made. The requested reproduction does not confirm the
stated mechanism, so this session stopped instead of inventing a fix.

## Hypothesis that was tested

`count_running_tasks_other_boards()` fail-opens to 0 when another board's DB
raises `database is locked` during a concurrent write (`BEGIN IMMEDIATE` /
WAL checkpoint). That undercount would let `dispatch_once` spawn past
`kanban.max_in_progress`.

The fail-open itself is real code:

- `count_running_tasks()` returns 0 on any exception
  (`hermes_cli/kanban_db_dispatch.py`).
- `count_running_tasks_other_boards()` `continue`s on any per-board exception
  and adds nothing for that board.

That path is not what a contended kanban DB actually does.

## Reproduction (isolated temp `HERMES_HOME`, pin env vars unset)

Two boards, board B seeded with 3 `status='running'` rows, journal mode
`wal`. A raw connection held `BEGIN IMMEDIATE` plus an uncommitted `UPDATE`
on B for the whole read.

- A second writer on B with `timeout=0.2` raised `sqlite3.OperationalError:
  database is locked` (the write lock was real).
- `count_running_tasks_other_boards(board="board-a")` returned **3** in
  **0.007s**, not 0. Busy timeout for that run was 400ms
  (`HERMES_KANBAN_BUSY_TIMEOUT_MS`); the read did not wait it out.
- An earlier run with `BEGIN EXCLUSIVE` and with the write held on a kanban
  `connect()` connection also returned 3 in ~5ms.

SQLite 3.53.4. WAL readers are not blocked by a writer, which is the mode
`kanban_db_connect.connect()` enables (`apply_wal_with_fallback`). A
transient write transaction therefore cannot be mistaken for "0 running
elsewhere".

## Why a silent transient undercount is also a poor fit even if a read did block

Kanban connections set `PRAGMA busy_timeout` from
`HERMES_KANBAN_BUSY_TIMEOUT_MS`, default **120_000** ms
(`hermes_cli/kanban_db_connect.py`, `DEFAULT_BUSY_TIMEOUT_MS`). A lock that
`busy_timeout` covers is waited out and then counted, not immediately
swallowed as 0. Returning 0 requires the exception to survive that wait (or
to be an error the busy handler does not retry). The dispatcher tick is
single-threaded and sequential across boards; a 120s stall would be the
visible symptom, not a quiet overshoot.

Missing/absent board files still skip without adding, which is the behavior
the ticket says must stay. That path was not the one under test, and it was
not changed.

## What this session did not do

- Did not change `max_in_progress` semantics or the cap check.
- Did not add a regression test that would freeze a non-reproducing mechanism.
- Did not investigate a replacement root cause. Re-diagnose from this
  objection; do not treat fail-open-on-`database is locked` as confirmed.
