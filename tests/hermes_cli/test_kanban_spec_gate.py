"""Issue #12: implementation claims require a full spec in the card BODY.

Card bodies below are verbatim from the live board (~/.hermes/kanban.db):
t_c0235d09 (no SCOPE/DESIGN, prose "Acceptance", no PROOF command) and
t_89782a19 (has every header, but bulleted ACCEPTANCE and a PROOF with no command).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kcli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

REAL_NO_SPEC_T_C0235D09 = 'After the boot-backfill-removal PR is merged, run a real end-to-end verification against a live environment.\n\nSteps:\n- Create a throwaway tenant through the normal production-path flow.\n- Confirm it provisions correctly with no reliance on the deleted boot backfill (check boot logs for absence of backfill, and that all resources/names/labels are correct at creation time).\n- Delete the tenant and confirm clean teardown with no orphaned resources.\n\nAcceptance: documented evidence that create→delete works identically to before removal. PROOF with no mocks — real tenant id, real timestamps, real log excerpts, real CLI/API output. Explicitly state any anomaly found rather than smoothing it over.'
REAL_PARTIAL_SPEC_T_89782A19 = 'DESIGN: Notion 3a4ac25eb49f807488f9fb854a73feaf block 3eeac25eb49f81adb70ef79893198207 (Tenant placement). This card is a regression fix.\n\nSCOPE: AgentPod, src/commands/plans.ts.\n\n`anyPurchasablePlanOffersHermes()` (around line 317) builds its selections with no hostType. After #5416, a missing hostType resolves to kubernetes. Hermes is only offerable on LXD while HERMES_K8S_ENABLED=false, so the /create picker is hidden for every user, including users who typed `/create lxd`.\n\nEvidence: validation run 37207559339, Phase A job 111458604899, on bot image 9670636ab. `/create lxd` goes straight to "Choose a plan" (plan:*:vps:vmp=lxd).\n\nFix:\n- Pass the /create placement into the gate.\n- `/create lxd` must show the picker with create:engine:hermes:lxd.\n- Bare /create should show Hermes when any plan can carry it on its resolvable placement. If the design says otherwise, confirm the intended behaviour with the EM.\n- Do not loosen telegram.crypto-menu.live.test.ts.\n\nACCEPTANCE:\n- Unit test covering /create lxd showing the Hermes picker with HERMES_K8S_ENABLED=false.\n- PR approved by agentpodreviewer at the exact head.\n- After the tag and deploy, Phase A of integration-telegram passes on the new SHA.\n\nPROOF: PR URL, review URL, and the Phase A run URL showing the :lxd step passing. No mocks in the live proof.\n\nRollback alternative while this is in flight: the EM redeploys v1.2.8.'
COMPLETE = (
    "SCOPE: hermes_cli/kanban_db.py claim path only.\n\n"
    "ACCEPTANCE:\n1. claim refuses incomplete bodies.\n2. existing tests stay green.\n\n"
    "PROOF: `pytest tests/hermes_cli/test_kanban_spec_gate.py -q`\nExpected: all passed.\n\n"
    "DESIGN: https://www.notion.so/3a4ac25eb49f807488f9fb854a73feaf\n"
)
COMMENT_FILL = (
    "SCOPE: x\nACCEPTANCE:\n1. y\nPROOF: `true`\nExpected: exit 0\n"
    "DESIGN: https://notion.so/3a4ac25eb49f807488f9fb854a73feaf"
)


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "kanban:\n  spec_gate:\n    enabled: true\n    impl_assignees: [software-engineer]\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kbc.connect()
    yield conn
    conn.close()


def _card(conn, body, assignee="software-engineer"):
    return kb.create_task(conn, title="impl", body=body, assignee=assignee, created_by="em")


def _assert_sent_back(conn, tid, *fields):
    t = kb.get_task(conn, tid)
    assert t.status == "triage" and t.assignee == "software-engineer"
    notes = [c.body for c in kb.list_comments(conn, tid) if "[spec-gate]" in c.body]
    assert len(notes) == 1
    for f in fields:
        assert f in notes[0]
    assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1  # no replacement card


@pytest.mark.parametrize("body,fields", [
    (REAL_NO_SPEC_T_C0235D09, ("SCOPE:", "ACCEPTANCE:", "PROOF:", "DESIGN:")),
    (REAL_PARTIAL_SPEC_T_89782A19, ("ACCEPTANCE:", "PROOF:")),
], ids=["t_c0235d09", "t_89782a19"])
def test_direct_claim_rejects_real_incomplete_bodies(board, body, fields):
    tid = _card(board, body)
    assert kb.claim_task(board, tid) is None
    _assert_sent_back(board, tid, *fields)


def test_cli_claim_uses_same_gate(board, capsys):
    tid = _card(board, REAL_NO_SPEC_T_C0235D09)
    rc = kcli._cmd_claim(argparse.Namespace(task_id=tid, ttl=None))
    assert rc != 0
    _assert_sent_back(board, tid, "PROOF:")


def test_dispatcher_spawn_uses_same_gate(board, monkeypatch):
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    tid = _card(board, REAL_PARTIAL_SPEC_T_89782A19)
    spawned = []
    kbd.dispatch_once(board, spawn_fn=lambda t, ws, **k: spawned.append(t.id) or 1)
    assert spawned == []
    _assert_sent_back(board, tid, "ACCEPTANCE:", "PROOF:")


def test_comments_never_satisfy_gate(board):
    tid = _card(board, REAL_NO_SPEC_T_C0235D09)
    kb.add_comment(board, tid, "em", COMMENT_FILL)
    assert kb.claim_task(board, tid) is None
    assert kb.get_task(board, tid).status == "triage"


def test_complete_body_and_justified_exemption_claim(board):
    tid = _card(board, COMPLETE)
    assert kb.claim_task(board, tid) is not None
    ex = COMPLETE.replace(
        "DESIGN: https://www.notion.so/3a4ac25eb49f807488f9fb854a73feaf",
        "DESIGN: exempt - one-line config typo, no behaviour change",
    )
    tid2 = _card(board, ex)
    assert kb.claim_task(board, tid2) is not None


def test_bare_exemption_and_other_assignees(board):
    bare = COMPLETE.replace("https://www.notion.so/3a4ac25eb49f807488f9fb854a73feaf", "exempt")
    tid = _card(board, bare)
    assert kb.claim_task(board, tid) is None
    other = _card(board, REAL_NO_SPEC_T_C0235D09, assignee="researcher")
    assert kb.claim_task(board, other) is not None
