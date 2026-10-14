"""#29: the design-phase guard must not re-fire on the same card body.

Regression (2026-10-13): the brief-incomplete gate re-parked t_ec089e4b / t_86aac7df on an
unchanged brief every tick, one zero-minute 'scheduled' run per pass. 3f2d79fa9e deleted
that gate (a worker that finds the brief incomplete blocks needs_input itself), so the
only remaining park is design_phase_missing, which hands the card to the architect. This
proves neither path loops on an unchanged body.
"""
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as disp

DESIGN_PAGE = "3eeac25eb49f814da4d1d8fce929cbf4"
NO_ACCEPTANCE = f"DESIGN: https://www.notion.so/{DESIGN_PAGE}\nSCOPE: x\nPROOF: z"


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
    kb.init_db()
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect() as c:
        yield c


def _guard_fires(conn, tid):
    return conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='respawn_guarded'",
                        (tid,)).fetchone()[0]


def test_brief_missing_marker_does_not_park_on_any_pass(conn):
    tid = kb.create_task(conn, title="impl", body=NO_ACCEPTANCE, assignee="software-engineer",
                         created_by="product-lead-agentpod")
    spawned = []
    for _ in range(4):
        disp.dispatch_once(conn, spawn_fn=lambda task, ws, **k: spawned.append(task.id))
        kb.unblock_task(conn, tid)
    assert spawned[:1] == [tid]  # the worker decides; no dispatcher park
    assert _guard_fires(conn, tid) == 0


def test_design_phase_park_fires_once_on_unchanged_body(conn):
    tid = kb.create_task(conn, title="impl", body="SCOPE: x", assignee="software-engineer",
                         created_by="product-lead-agentpod")
    spawned = []
    for _ in range(4):
        disp.dispatch_once(conn, spawn_fn=lambda task, ws, **k: spawned.append(
            (task.id, task.assignee)))
    assert _guard_fires(conn, tid) == 1
    assert kb.get_task(conn, tid).assignee == "architect-critic-fable"
    assert all(a == "architect-critic-fable" for _, a in spawned)
