"""Implementation-claim spec gate (harness issue #12).

An implementation card may only move ``ready -> running`` when its canonical
BODY (never the title, never comments) carries a full spec:

* ``SCOPE:``      non-empty.
* ``ACCEPTANCE:`` at least one numbered criterion (``1.`` / ``1)``).
* ``PROOF:``      a command (inline `code` or a fenced block) plus an expected
                  result (the word "expect", e.g. ``Expected: 3 passed``).
* ``DESIGN:``     a reference (URL, 32-hex Notion id, or file path) or an
                  explicit exemption with a reason, e.g.
                  ``DESIGN: exempt — one-line typo fix, no behaviour change``.

The single enforcement point is :func:`hermes_cli.kanban_db.claim_task`, so the
dispatcher spawn and ``hermes kanban claim`` run the same validator. An
incomplete card is sent back to its existing owner (status ``triage``, same
assignee) with one comment listing the missing fields; no new card is created.

Config (root ``config.yaml``)::

    kanban:
      spec_gate:
        enabled: true
        impl_assignees: [software-engineer]

When ``spec_gate`` is absent, ``kanban.design_phase.impl_assignees`` (if
enabled) is reused so an existing design-phase setup is gated too.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Optional

SPEC_FIELDS = ("SCOPE", "ACCEPTANCE", "PROOF", "DESIGN")
GATE_AUTHOR = "spec-gate"
_HEADER = re.compile(r"^[ \t>*_#-]*([A-Z][A-Z _/-]{1,30}):", re.M)
_NUMBERED = re.compile(r"^[ \t]*\d+[.)][ \t]+\S", re.M)
_COMMAND = re.compile(r"`[^`\n]+`|```")
_EXPECT = re.compile(r"\bexpect", re.I)
_DESIGN_REF = re.compile(r"https?://\S+|\b[0-9a-f]{32}\b|[\w.-]*/[\w./-]+\.\w+", re.I)
_EXEMPT = re.compile(r"^\W*(exempt(?:ion)?|n/?a|none)\b\W*(.*)$", re.I | re.S)


def _sections(body: str) -> dict[str, str]:
    """Map each spec field to its text (first occurrence, up to the next header)."""
    out: dict[str, str] = {}
    heads = list(_HEADER.finditer(body))
    for i, m in enumerate(heads):
        name = m.group(1).strip()
        if name not in SPEC_FIELDS or name in out:
            continue
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        out[name] = body[m.end():end].strip()
    return out


def validate_card_body(body: Optional[str]) -> list[str]:
    """Return human-readable problems with the card body; empty means complete."""
    sec = _sections(body or "")
    missing: list[str] = []
    if not sec.get("SCOPE"):
        missing.append("SCOPE: (populated scope)")
    if not _NUMBERED.search(sec.get("ACCEPTANCE", "")):
        missing.append("ACCEPTANCE: (numbered criteria: 1. 2. ...)")
    proof = sec.get("PROOF", "")
    if not (_COMMAND.search(proof) and _EXPECT.search(proof)):
        missing.append("PROOF: (a `command` plus Expected: result)")
    design = sec.get("DESIGN", "")
    ex = _EXEMPT.match(design)
    if ex:
        if len(re.findall(r"\w+", ex.group(2))) < 3:
            missing.append("DESIGN: (exemption needs a justification)")
    elif not _DESIGN_REF.search(design):
        missing.append("DESIGN: (design reference or justified exemption)")
    return missing


def _gate_assignees() -> Optional[set[str]]:
    try:
        import yaml
        from hermes_constants import get_default_hermes_root
        raw = yaml.safe_load((get_default_hermes_root() / "config.yaml").read_text()) or {}
    except Exception:
        return None
    kb = raw.get("kanban") or {}
    cfg = kb.get("spec_gate")
    if cfg is None:
        cfg = kb.get("design_phase") or {}
        if not cfg:
            return None
    if not cfg.get("enabled", True):
        return None
    return set(cfg.get("impl_assignees") or ["software-engineer"])


def enforce_spec_gate(conn: sqlite3.Connection, task_id: str,
                      assignees: Optional[set[str]] = None) -> Optional[list[str]]:
    """Run inside the claim txn. Returns the missing fields after sending the card
    back to its owner (``triage`` + comment), or ``None`` when it may be claimed."""
    if assignees is None:
        assignees = _gate_assignees()
    if not assignees:
        return None
    row = conn.execute(
        "SELECT assignee, body, status FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if row is None or row["status"] != "ready" or (row["assignee"] or "") not in assignees:
        return None
    missing = validate_card_body(row["body"])
    if not missing:
        return None
    from hermes_cli import kanban_db as _kb
    conn.execute(
        "UPDATE tasks SET status = 'triage' WHERE id = ? AND status = 'ready'", (task_id,)
    )
    _kb.add_comment(
        conn, task_id, GATE_AUTHOR,
        "[spec-gate] Not claimable: the card body is missing " + "; ".join(missing)
        + ". Comments do not count; edit the body, then move the card out of triage.",
    )
    _kb._append_event(conn, task_id, "claim_rejected",
                      {"reason": "spec_incomplete", "missing": missing})
    return missing
