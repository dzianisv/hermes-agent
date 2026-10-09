"""Design-phase guard: a DESIGN: line only counts when it names a configured design page.

Regression: card t_5598219a (PR #5447) passed the guard with `DESIGN: ... https://app.notion.com/p/3e4ac25e...`
(the readiness-gaps page). Any 32-hex id was accepted, so it reached the engineer with no design for its
alert cases and was rejected three times in review.
"""
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as disp

DESIGN_PAGE = "3eeac25eb49f814da4d1d8fce929cbf4"
GAPS_PAGE = "3e4ac25eb49f81baa91cd2b23640c7e4"


@pytest.fixture
def conn(tmp_path, monkeypatch):
    for v in ("HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB", "HERMES_KANBAN_HOME"):
        monkeypatch.delenv(v, raising=False)
    root = tmp_path / "root"; root.mkdir()
    (root / "config.yaml").write_text(
        "kanban:\n  design_phase:\n    enabled: true\n    impl_assignees: [software-engineer]\n"
        "    architects: [architect-critic-fable]\n    architect: architect-critic-fable\n"
        "    required: ['DESIGN:', 'SCOPE:', 'ACCEPTANCE:', 'PROOF:']\n"
        f"    design_pages: ['{DESIGN_PAGE}']\n"
        "    verify_notion: false\n")  # Notion lookup covered by test_kanban_design_section_verify
    monkeypatch.setenv("HERMES_ROOT_HOME", str(root))
    home = tmp_path / "home"; home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect() as c:
        yield c


def _card(conn, design_line):
    body = f"{design_line}\nSCOPE: x\nACCEPTANCE: 1. y\nPROOF: z, no mocks"
    return kb.create_task(conn, title="impl", body=body, assignee="software-engineer", created_by="product-lead-agentpod")


def test_link_to_non_design_page_is_parked(conn):
    tid = _card(conn, f"DESIGN: DESIGN v7.1 §2 — https://app.notion.com/p/{GAPS_PAGE}")
    assert disp._design_phase_guard(conn, tid, "software-engineer") == "design_phase_missing"
    assert kb.get_task(conn, tid).assignee == "architect-critic-fable"


def test_link_to_design_page_passes(conn):
    tid = _card(conn, f"DESIGN: v7.2 §16 https://www.notion.so/{DESIGN_PAGE}")
    assert disp._design_phase_guard(conn, tid, "software-engineer") is None


def test_dashed_design_page_id_passes(conn):
    tid = _card(conn, "DESIGN: 3eeac25e-b49f-814d-a4d1-d8fce929cbf4 §16")
    assert disp._design_phase_guard(conn, tid, "software-engineer") is None


@pytest.mark.parametrize("stage", ["review", "merge", "deploy"])
def test_card_past_design_entry_is_not_rerouted(conn, stage):
    """Regression t_be2a94d1: PRs merged+deployed, card handed back to the engineer for
    tag/live proof, and the guard rerouted it to the architect because its brief lacked DESIGN:."""
    tid = _card(conn, "DESIGN: none")
    conn.execute("UPDATE tasks SET current_step_key=? WHERE id=?", (stage, tid)); conn.commit()
    assert disp._design_phase_guard(conn, tid, "software-engineer") is None
    assert kb.get_task(conn, tid).assignee == "software-engineer"


def test_design_done_on_same_card_passes(conn):
    """Regression t_f3e9efe4: architect finished the design ON the card and handed it
    to the engineer; the guard parked it again (3 bounces) because the design was a
    child page, not one of design_pages."""
    tid = _card(conn, "DESIGN: card comment #10638")
    conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
        "VALUES (?, 'architect-critic-fable', 'blocked', 'blocked', 1, 2)", (tid,))
    assert disp._design_phase_guard(conn, tid, "software-engineer") is None


def test_unfinished_architect_run_does_not_count(conn):
    tid = _card(conn, "DESIGN: card comment")
    conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, outcome, started_at) "
        "VALUES (?, 'architect-critic-fable', 'crashed', 'crashed', 1)", (tid,))
    assert disp._design_phase_guard(conn, tid, "software-engineer") == "design_phase_missing"
