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
        f"    design_pages: ['{DESIGN_PAGE}']\n")
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


DESIGN_LINE = f"DESIGN: v7.2 §16 https://www.notion.so/{DESIGN_PAGE}"


def _raw_card(conn, body):
    return kb.create_task(conn, title="impl", body=body, assignee="software-engineer", created_by="product-lead-agentpod")


def test_qualified_brief_keys_pass(conn):
    body = f"{DESIGN_LINE}\nSCOPE (fork): x\nACCEPTANCE (T3): y\nPROOF (after merge, no mocks): z"
    tid = _raw_card(conn, body)
    assert disp._design_phase_guard(conn, tid, "software-engineer") is None


def test_plain_brief_keys_still_pass(conn):
    tid = _raw_card(conn, f"{DESIGN_LINE}\nSCOPE: x\nACCEPTANCE: y\nPROOF: z")
    assert disp._design_phase_guard(conn, tid, "software-engineer") is None


def test_missing_key_is_parked(conn):
    tid = _raw_card(conn, f"{DESIGN_LINE}\nSCOPE: x\nACCEPTANCE: y\nno proof here")
    assert disp._design_phase_guard(conn, tid, "software-engineer") == "brief_incomplete"
    assert kb.get_task(conn, tid).status == "scheduled"
    reasons = [(e.payload or {}).get("reason") or "" for e in kb.list_events(conn, tid) if e.kind == "scheduled"]
    assert reasons and "PROOF:" in reasons[-1]
    assert "ACCEPTANCE:" not in reasons[-1].split("missing", 1)[1]


def test_key_mentioned_mid_sentence_does_not_count(conn):
    # The literal header token appears, but not at line start: a substring
    # check would accept it; the per-line anchor must not.
    body = (f"{DESIGN_LINE}\nSCOPE: x\n"
            "We will define see ACCEPTANCE: criteria in a follow-up card.\nPROOF: z")
    tid = _raw_card(conn, body)
    assert disp._design_phase_guard(conn, tid, "software-engineer") == "brief_incomplete"
    assert kb.get_task(conn, tid).status == "scheduled"
    reasons = [(e.payload or {}).get("reason") or "" for e in kb.list_events(conn, tid) if e.kind == "scheduled"]
    assert reasons and "ACCEPTANCE:" in reasons[-1].split("missing", 1)[1]


def test_qualified_design_header_with_design_page_passes(conn):
    tid = _card(conn, f"DESIGN (v7.2 §16): https://www.notion.so/{DESIGN_PAGE}")
    assert disp._design_phase_guard(conn, tid, "software-engineer") is None


def test_qualified_design_header_with_non_design_page_is_parked(conn):
    tid = _card(conn, f"DESIGN (v7.2 §16): https://www.notion.so/{GAPS_PAGE}")
    assert disp._design_phase_guard(conn, tid, "software-engineer") == "design_phase_missing"
    assert kb.get_task(conn, tid).assignee == "architect-critic-fable"
