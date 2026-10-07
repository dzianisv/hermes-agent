"""Worker context shows every card comment: specs in full, the rest in full
or as a pointer line — never silently dropped or cut."""
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    c = kb.connect()
    yield c
    c.close()


def _design_body(size: int) -> str:
    head = "DESIGN: rewrite the dispatcher\n"
    body = head + "".join(f"line {i:05d} of the design detail\n" for i in range(size // 30))
    return body + "END-OF-DESIGN-MARKER"


def test_large_design_comment_then_40_routine_comments(conn):
    tid = kb.create_task(conn, title="t", assignee="eng")
    design = _design_body(9 * 1024)
    assert len(design) > 9 * 1024
    kb.add_comment(conn, tid, "architect", design)
    routine_ids = [
        kb.add_comment(conn, tid, "eng", f"routine note {i} " + "x" * 50) for i in range(40)
    ]

    ctx = kb.build_worker_context(conn, tid)

    assert design.strip() in ctx  # full, untruncated
    assert "END-OF-DESIGN-MARKER" in ctx
    # Spec block comes before the recent tail.
    assert ctx.index("END-OF-DESIGN-MARKER") < ctx.index("routine note 39 ")
    # Every routine comment is accounted for: 30 newest in full, 10 listed.
    for i, cid in enumerate(routine_ids):
        assert f"routine note {i} " in ctx
    listed = [ln for ln in ctx.splitlines() if ln.startswith("- #")]
    assert len(listed) == 10
    assert all(f"#{routine_ids[i]} " in "\n".join(listed) for i in range(10))
    assert "kanban_show" in ctx


def test_spec_markers_and_ordering(conn):
    tid = kb.create_task(conn, title="t", assignee="eng")
    kb.add_comment(conn, tid, "em", "context\nEM DECISION: ship option B")
    kb.add_comment(conn, tid, "eng", "just chatting")
    kb.add_comment(conn, tid, "arch", "R1: must be idempotent")
    for i in range(35):
        kb.add_comment(conn, tid, "eng", f"chatter {i}")
    ctx = kb.build_worker_context(conn, tid)
    a, b = ctx.index("EM DECISION: ship"), ctx.index("R1: must be")
    assert a < b < ctx.index("chatter 34")
    # "just chatting" is old routine: listed, not dropped.
    assert any("just chatting" in ln and ln.startswith("- #") for ln in ctx.splitlines())


def test_total_budget_drops_routine_before_specs(conn, tmp_path):
    (tmp_path / ".hermes" / "config.yaml").write_text(
        "kanban:\n  context_max_total_chars: 30000\n  context_max_comment_chars: 8192\n"
    )
    tid = kb.create_task(conn, title="t", assignee="eng")
    design = _design_body(20 * 1024)
    kb.add_comment(conn, tid, "arch", design)
    for i in range(10):
        kb.add_comment(conn, tid, "eng", f"big routine {i} " + "y" * 6000)
    ctx = kb.build_worker_context(conn, tid)
    assert design.strip() in ctx
    assert len(ctx) < 32000
    listed = [ln for ln in ctx.splitlines() if ln.startswith("- #")]
    assert listed, "omitted routine comments must be listed"
    for i in range(10):
        assert f"big routine {i} " in ctx


def test_routine_comment_cap_is_8k(conn):
    tid = kb.create_task(conn, title="t", assignee="eng")
    body = "note\n" + "z" * 7000 + "TAIL-OK"
    kb.add_comment(conn, tid, "eng", body)
    ctx = kb.build_worker_context(conn, tid)
    assert "TAIL-OK" in ctx
