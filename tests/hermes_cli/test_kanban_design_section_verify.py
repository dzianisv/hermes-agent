"""Design gate resolves the DESIGN: citation in Notion before an engineer can claim.

Row #24: naming a configured design page was enough, so a card could cite any
section (t_5598219a). Now the page and the cited section heading must exist.
Notion is a local HTTP server replaying responses recorded from the real API
(tests/hermes_cli/fixtures/notion_design); hermes code is not mocked.
"""
import http.server
import json
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as disp

FIX = Path(__file__).parent / "fixtures" / "notion_design"
PAGE = "3eeac25eb49f814da4d1d8fce929cbf4"          # AgentPod DESIGN page
SEC16 = "db90b4dabfb54581873b82e2a9e095f3"         # heading_2 "16. v7.2 ..."
SEC161 = "7ededce07f5744ff8358da86d3562bc1"        # heading_3 "16.1 v7.2.1 ..."
PARA = "3efac25eb49f81efa2ccd436ac390ad6"          # a paragraph on the page
GONE = "0000000000004000800000000000beef"


class _Notion(http.server.BaseHTTPRequestHandler):
    hits: list = []
    down = False

    def log_message(self, *a):
        pass

    def do_GET(self):
        type(self).hits.append(self.path)
        if type(self).down:
            self.send_response(503); self.end_headers(); return
        assert self.headers["Authorization"] == "Bearer test-token"
        name = self.path.removeprefix("/v1/").replace("/", "_").replace("?", "_").replace("=", "_").replace("&", "_")
        f = FIX / f"{name}.json"
        body = f.read_bytes() if f.exists() else json.dumps({"object": "error", "status": 404}).encode()
        self.send_response(200 if f.exists() else 404)
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def notion():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Notion)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Notion.hits, _Notion.down = [], False
    disp._NOTION_CACHE.clear()
    yield _Notion, f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.fixture
def conn(tmp_path, monkeypatch, notion):
    for v in ("HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB", "HERMES_KANBAN_HOME"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("NOTION_TOKEN", "test-token")
    root = tmp_path / "root"; root.mkdir()
    (root / "config.yaml").write_text(
        "kanban:\n  design_phase:\n    enabled: true\n    impl_assignees: [software-engineer]\n"
        "    architects: [architect-critic-fable]\n    architect: architect-critic-fable\n"
        "    required: ['DESIGN:', 'SCOPE:', 'ACCEPTANCE:', 'PROOF:']\n"
        f"    design_pages: ['{PAGE}']\n    notion_api_base: '{notion[1]}'\n    notion_timeout: 3\n")
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


def _parked(conn, tid, reason):
    assert disp._design_phase_guard(conn, tid, "software-engineer") == reason
    t = kb.get_task(conn, tid)
    assert (t.status, t.assignee) == ("triage", "architect-critic-fable")
    ev = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='respawn_guarded'", (tid,)).fetchone()
    assert json.loads(ev[0])["reason"] == reason
    return json.loads(ev[0])["detail"]


def test_existing_section_number_passes(conn):
    tid = _card(conn, f"DESIGN: v7.2 §16 https://www.notion.so/{PAGE}")
    assert disp._design_phase_guard(conn, tid, "software-engineer") is None
    assert kb.get_task(conn, tid).status == "ready"


def test_existing_subsection_on_second_result_page_passes(conn, notion):
    tid = _card(conn, f"DESIGN: https://app.notion.com/p/{PAGE} §16.1")
    assert disp._design_phase_guard(conn, tid, "software-engineer") is None
    assert any("start_cursor" in h for h in notion[0].hits)  # pagination followed


def test_block_anchor_to_heading_passes(conn):
    tid = _card(conn, f"DESIGN: https://www.notion.so/DESIGN-{PAGE}#{SEC16}")
    assert disp._design_phase_guard(conn, tid, "software-engineer") is None


def test_nonexistent_section_number_is_parked(conn):
    tid = _card(conn, f"DESIGN: https://www.notion.so/{PAGE} §42")
    assert "§42" in _parked(conn, tid, "design_section_missing")


def test_section_prefix_does_not_match_longer_number(conn):
    # "1" must not be satisfied by heading "16. ..." / "10. ..." only by "1. ..."
    tid = _card(conn, f"DESIGN: https://www.notion.so/{PAGE} §16.9")
    assert "§16.9" in _parked(conn, tid, "design_section_missing")


def test_page_without_section_is_parked(conn):
    tid = _card(conn, f"DESIGN: https://www.notion.so/{PAGE}")
    assert "cites no section" in _parked(conn, tid, "design_section_missing")


def test_anchor_to_missing_block_is_parked(conn):
    tid = _card(conn, f"DESIGN: https://www.notion.so/{PAGE}#{GONE}")
    assert "does not exist" in _parked(conn, tid, "design_section_missing")


def test_anchor_to_non_heading_is_parked(conn):
    tid = _card(conn, f"DESIGN: https://www.notion.so/{PAGE}#{PARA}")
    assert "not a section heading" in _parked(conn, tid, "design_section_missing")


def test_notion_unreachable_parks_with_reason(conn, notion):
    notion[0].down = True
    tid = _card(conn, f"DESIGN: https://www.notion.so/{PAGE} §16")
    assert "HTTP 503" in _parked(conn, tid, "design_unverifiable")


def test_missing_token_parks_never_passes(conn, monkeypatch, tmp_path):
    monkeypatch.delenv("NOTION_TOKEN")
    (Path(disp.os.environ["HERMES_ROOT_HOME"]) / "config.yaml").write_text(
        (Path(disp.os.environ["HERMES_ROOT_HOME"]) / "config.yaml").read_text()
        + f"    notion_env_file: '{tmp_path / 'none.env'}'\n")
    tid = _card(conn, f"DESIGN: https://www.notion.so/{PAGE} §16")
    assert "NOTION_TOKEN" in _parked(conn, tid, "design_unverifiable")


def test_token_read_from_env_file(conn, monkeypatch, tmp_path):
    monkeypatch.delenv("NOTION_TOKEN")
    envf = tmp_path / "notion.env"; envf.write_text("export NOTION_TOKEN='test-token'\n")
    cfgp = Path(disp.os.environ["HERMES_ROOT_HOME"]) / "config.yaml"
    cfgp.write_text(cfgp.read_text() + f"    notion_env_file: '{envf}'\n")
    tid = _card(conn, f"DESIGN: https://www.notion.so/{PAGE} §16")
    assert disp._design_phase_guard(conn, tid, "software-engineer") is None


def test_lookups_are_cached_across_cards(conn, notion):
    a = _card(conn, f"DESIGN: https://www.notion.so/{PAGE} §16")
    assert disp._design_phase_guard(conn, a, "software-engineer") is None
    n = len(notion[0].hits)
    b = _card(conn, f"DESIGN: https://www.notion.so/{PAGE} §14")
    assert disp._design_phase_guard(conn, b, "software-engineer") is None
    assert len(notion[0].hits) == n


def test_failure_is_not_cached(conn, notion):
    notion[0].down = True
    a = _card(conn, f"DESIGN: https://www.notion.so/{PAGE} §16")
    assert disp._design_phase_guard(conn, a, "software-engineer") == "design_unverifiable"
    notion[0].down = False
    b = _card(conn, f"DESIGN: https://www.notion.so/{PAGE} §16")
    assert disp._design_phase_guard(conn, b, "software-engineer") is None


def test_parked_card_is_not_dispatched(conn):
    tid = _card(conn, f"DESIGN: https://www.notion.so/{PAGE} §42")
    disp._design_phase_guard(conn, tid, "software-engineer")
    assert kb.get_task(conn, tid).status == "triage"
    assert disp._design_phase_guard(conn, tid, "architect-critic-fable") is None
