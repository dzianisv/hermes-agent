"""Brief-gate park waits on the card's content, not on a blind re-dispatch.

Regression (kanban.db, 72h to 2026-10-13): 'BLOCKER:DEP EM — [design-phase] brief
incomplete: missing ACCEPTANCE:' produced 4 zero-minute 'scheduled' runs on
t_ec089e4b and t_86aac7df. Sequence: guard parks (scheduled run #1) -> EM
`specified`/`unblocked` the card -> next tick the guard re-parks on a brief that
still lacks the marker (run #2, 0 s) ... Nothing woke the card when the brief
actually changed; only manual unblocks did, each one costing a run.
"""
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as disp

DESIGN_PAGE = "3eeac25eb49f814da4d1d8fce929cbf4"
INCOMPLETE = f"DESIGN: https://www.notion.so/{DESIGN_PAGE}\nSCOPE: x\nPROOF: z"


@pytest.fixture
def conn(tmp_path, monkeypatch):
    for v in ("HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB", "HERMES_KANBAN_HOME"):
        monkeypatch.delenv(v, raising=False)
    root = tmp_path / "root"; root.mkdir()
    (root / "config.yaml").write_text(
        "kanban:\n  design_phase:\n    enabled: true\n"
        "    impl_assignees: [software-engineer]\n"
        "    architects: [architect-critic-fable]\n    architect: architect-critic-fable\n"
        "    required: ['DESIGN:', 'SCOPE:', 'ACCEPTANCE:', 'PROOF:']\n"
        f"    design_pages: ['{DESIGN_PAGE}']\n    verify_notion: false\n")
    monkeypatch.setenv("HERMES_ROOT_HOME", str(root))
    home = tmp_path / "home"; home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(disp, "_profile_exists_fn", lambda: None)
    from hermes_cli import kanban_wait_on
    monkeypatch.setattr(kanban_wait_on, "DEFAULT_POLL_SECONDS", 0)  # poll every tick
    kb.init_db()
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect() as c:
        yield c


def _scheduled_runs(conn, tid):
    return conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id=? AND outcome='scheduled'",
                        (tid,)).fetchone()[0]


def test_brief_gate_waits_on_card_body_and_wakes_on_edit(conn):
    tid = kb.create_task(conn, title="impl", body=INCOMPLETE, assignee="software-engineer",
                         created_by="product-lead-agentpod")
    spawned = []
    spawn = lambda task, ws, **k: spawned.append(task.id)

    disp.dispatch_once(conn, spawn_fn=spawn)
    t = conn.execute("SELECT status, wait_on FROM tasks WHERE id=?", (tid,)).fetchone()
    assert t["status"] == "scheduled" and t["wait_on"] == f"card_body:{tid}"
    assert _scheduled_runs(conn, tid) == 1

    # N passes, plus an EM unblock with the brief unchanged: no spawn, no new run.
    for _ in range(3):
        disp.dispatch_once(conn, spawn_fn=spawn)
    kb.unblock_task(conn, tid)
    disp.dispatch_once(conn, spawn_fn=spawn)
    disp.dispatch_once(conn, spawn_fn=spawn)
    assert spawned == []
    assert kb.get_task(conn, tid).status == "scheduled"
    assert _scheduled_runs(conn, tid) == 1

    # The worker's own comment is not an answer to the brief gate.
    kb.add_comment(conn, tid, author="software-engineer", body="still waiting")
    disp.dispatch_once(conn, spawn_fn=spawn)
    assert spawned == [] and kb.get_task(conn, tid).status == "scheduled"

    # Editing the body wakes it without a manual unblock; guard passes and it spawns.
    conn.execute("UPDATE tasks SET body=? WHERE id=?", (INCOMPLETE + "\nACCEPTANCE: 1. y", tid))
    conn.commit()
    disp.dispatch_once(conn, spawn_fn=spawn)
    assert spawned == [tid]
    assert _scheduled_runs(conn, tid) == 1


def test_em_comment_wakes_card(conn):
    tid = kb.create_task(conn, title="impl", body=INCOMPLETE, assignee="software-engineer",
                         created_by="product-lead-agentpod")
    spawned = []
    spawn = lambda task, ws, **k: spawned.append(task.id)
    disp.dispatch_once(conn, spawn_fn=spawn)
    kb.add_comment(conn, tid, author="product-lead-agentpod", body="ACCEPTANCE: 1. y")
    disp.dispatch_once(conn, spawn_fn=spawn)
    assert spawned == [tid]
