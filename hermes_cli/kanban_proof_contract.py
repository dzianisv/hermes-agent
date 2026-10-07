"""Live-proof completion contract for kanban cards.

A card whose body carries a ``PROOF: <command>`` line is only done when the
worker actually ran that command: ``kanban_complete`` must carry
``metadata.proof = {"command", "output", "ran_at"}`` with a command matching
the PROOF line (normalized) and non-empty output. Cards without a PROOF line
are unaffected.
"""
from __future__ import annotations

import re

_PROOF_LINE = re.compile(r"^[ \t>*\-]*\**PROOF\**:[ \t]*(.+?)[ \t]*$", re.MULTILINE)


def normalize_command(cmd: str) -> str:
    """Strip markdown fences/backticks, a leading shell prompt, and whitespace runs."""
    s = (cmd or "").strip().strip("`").strip()
    s = re.sub(r"^\$\s+", "", s)
    return " ".join(s.split())


def proof_commands(body: str | None) -> list[str]:
    """Return normalized commands from every PROOF: line in the card body."""
    return [c for c in (normalize_command(m) for m in _PROOF_LINE.findall(body or "")) if c]


def proof_violation(body: str | None, metadata: object) -> str | None:
    """Return a refusal message, or None when the contract is satisfied/not applicable."""
    expected = proof_commands(body)
    if not expected:
        return None
    want = " | ".join(f"`{c}`" for c in expected)
    howto = (f"Run the card's PROOF command ({want}) for real, then retry kanban_complete with "
             f"metadata.proof = {{\"command\": <the PROOF command>, \"output\": <pasted output>, "
             f"\"ran_at\": <ISO timestamp>}}. If you cannot run it, call kanban_block with a reason "
             f"starting 'BLOCKER:EXTERNAL' instead of completing. Your task is still in-flight "
             f"(no state change).")
    proof = metadata.get("proof") if isinstance(metadata, dict) else None
    if not isinstance(proof, dict):
        return f"kanban_complete refused: this card has a PROOF: line but metadata.proof is missing. {howto}"
    missing = [k for k in ("command", "output", "ran_at")
               if not isinstance(proof.get(k), str) or not proof[k].strip()]
    if missing:
        return (f"kanban_complete refused: metadata.proof is missing or has empty "
                f"{', '.join(missing)}. {howto}")
    if normalize_command(proof["command"]) not in expected:
        return (f"kanban_complete refused: metadata.proof.command "
                f"`{normalize_command(proof['command'])}` does not match the card's PROOF "
                f"command. {howto}")
    return None
