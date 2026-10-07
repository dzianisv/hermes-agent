"""A review worker's first prompt carries the design, the card's requirements and the
single-pass instruction; design fetch failures are stated, never dropped.

Real kanban DB under a temp HERMES_HOME, real dispatcher, a local HTTP server standing in
for api.notion.com; only the process spawn is faked (it records the real worker argv).
"""

from __future__ import annotations

import http.server
import json
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_review_brief as krb

PAGE = "a" * 32
SECTION = "b" * 32
OTHER = "c" * 32


def _blk(bid, kind, text):
    return {"id": bid, "type": kind, kind: {"rich_text": [{"plain_text": text}]}}


BLOCKS = [
    _blk("0" * 32, "heading_2", "Intro"),
    _blk("1" * 32, "paragraph", "intro text not in the section"),
    _blk(SECTION, "heading_2", "Exit reasons"),
    _blk("2" * 32, "paragraph", "Record rc and signal on every exit."),
    _blk("3" * 32, "heading_3", "Sub"),
    _blk("4" * 32, "bulleted_list_item", "classify transient exits"),
    _blk("5" * 32, "heading_2", "Next section"),
    _blk("6" * 32, "paragraph", "must not appear"),
]


@pytest.fixture
def notion(monkeypatch):
    calls = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, self.headers.get("Authorization")))
            if f"/v1/blocks/{PAGE}/children" in self.path:
                body, code = {"results": BLOCKS, "has_more": False}, 200
            else:
                body, code = {"message": "nope"}, 404
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("HERMES_NOTION_API_BASE", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setenv("NOTION_TOKEN", "secret-test-token")
    krb._design_cache.clear()
    yield calls
    srv.shutdown()


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kbd, "_resolve_worker_cli_toolsets", lambda _home: None)
    for prof in ("reviewer", "engineer"):
        (h / "profiles" / prof).mkdir(parents=True)
        (h / "profiles" / prof / "config.yaml").write_text("{}\n")
    kb.init_db()
    return h


class _Spawner:
    def __init__(self):
        self.argvs = []

    def __call__(self, task, workspace, board=None):
        self.argvs.append(kbd._worker_argv(task, task.assignee, None))
        return 999_999


def _prompt(conn, tid):
    sp = _Spawner()
    res = kbd.dispatch_once(conn, spawn_fn=sp, failure_limit=50)
    assert sp.argvs, f"task was not spawned: {res}"
    argv = sp.argvs[-1]
    return argv[argv.index("-q") + 1]


CARD = (
    "Fix exits.\n"
    f"DESIGN: https://www.notion.so/Harness-{PAGE}#{SECTION}\n"
    "SCOPE: dispatcher only\n"
    "ACCEPTANCE: every exit has a reason\n"
    "PROOF: test output\n"
    "R1: record rc\n"
    "R2: requeue transient\n"
    "unrelated line R3 not at start\n"
)


def test_reviewer_prompt_has_design_section_requirements_and_single_pass(home, notion):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review exits", assignee="reviewer", body=CARD)
        prompt = _prompt(conn, tid)
    assert prompt.startswith(f"work kanban task {tid}")
    assert "Record rc and signal on every exit." in prompt
    assert "classify transient exits" in prompt          # nested heading stays in the section
    assert "must not appear" not in prompt                # next same-level heading ends it
    assert "intro text" not in prompt
    for line in ("SCOPE: dispatcher only", "ACCEPTANCE: every exit has a reason",
                 "PROOF: test output", "R1: record rc", "R2: requeue transient"):
        assert line in prompt
    assert "R3 not at start" not in prompt
    assert krb.SINGLE_PASS_INSTRUCTION in prompt
    assert notion and notion[0][1] == "Bearer secret-test-token"


def test_design_is_fetched_once_per_run_of_the_dispatcher(home, notion):
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="r1", assignee="reviewer", body=CARD)
        b = kb.create_task(conn, title="r2", assignee="reviewer", body=CARD)
        sp = _Spawner()
        kbd.dispatch_once(conn, spawn_fn=sp, failure_limit=50)
    prompts = [argv[argv.index("-q") + 1] for argv in sp.argvs]
    assert {p.split()[3] for p in prompts} == {a, b}
    assert all("Record rc and signal" in p for p in prompts)
    assert len(notion) == 1


@pytest.mark.parametrize("body, expect", [
    (CARD.replace(PAGE, OTHER), "design unavailable: Notion HTTP 404"),
    ("SCOPE: x\nR1: y\n", "design unavailable: card has no DESIGN: line"),
])
def test_design_failure_is_stated_not_silent(home, notion, body, expect):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="r", assignee="reviewer", body=body)
        prompt = _prompt(conn, tid)
    assert expect in prompt
    assert "R1:" in prompt or "R1: record rc" in prompt
    assert krb.SINGLE_PASS_INSTRUCTION in prompt


def test_missing_token_is_stated(home, monkeypatch):
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    krb._design_cache.clear()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="r", assignee="reviewer", body=CARD)
        prompt = _prompt(conn, tid)
    assert "design unavailable: NOTION_TOKEN not set" in prompt


def test_design_line_in_a_later_comment_wins(home, notion):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="r", assignee="reviewer",
                             body=CARD.replace(PAGE, OTHER))
        kb.add_comment(conn, tid, "architect", f"DESIGN: https://notion.so/x-{PAGE}#{SECTION}")
        prompt = _prompt(conn, tid)
    assert "Record rc and signal" in prompt


def test_non_review_worker_prompt_is_unchanged(home, notion):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="build", assignee="engineer", body=CARD)
        prompt = _prompt(conn, tid)
    assert prompt == f"work kanban task {tid}"
    assert notion == []
