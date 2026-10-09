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


# ---------------------------------------------------------------------------
# Harness-run proofs.
#
# Syntax (one command per line, anywhere in the card body):
#   PROOF-CMD: <shell command>         harness runs it on kanban_complete;
#                                      non-zero exit refuses completion.
#   PROOF-CMD-ASYNC: <shell command>   long/live proof: kanban_complete moves the
#                                      card to ``review`` (the verify state) and
#                                      ``rerun_pending_proofs`` completes it later
#                                      once the command exits 0.
# Prose ``PROOF:`` lines keep the worker-pasted contract above and are never executed.
# ---------------------------------------------------------------------------
import os
import subprocess
import time

_PROOF_CMD_LINE = re.compile(
    r"^[ \t>*\-]*\**PROOF-CMD(-ASYNC)?\**:[ \t]*(.+?)[ \t]*$", re.MULTILINE)
DEFAULT_PROOF_TIMEOUT = 300
OUTPUT_TAIL_CHARS = 2000


def proof_cmds(body: str | None) -> list[tuple[str, bool]]:
    """Return ``[(command, is_async), ...]`` for every PROOF-CMD line."""
    out = []
    for is_async, cmd in _PROOF_CMD_LINE.findall(body or ""):
        cmd = normalize_command(cmd)
        if cmd:
            out.append((cmd, bool(is_async)))
    return out


def proof_timeout() -> int:
    try:
        return max(1, int(os.environ.get("HERMES_KANBAN_PROOF_TIMEOUT", DEFAULT_PROOF_TIMEOUT)))
    except ValueError:
        return DEFAULT_PROOF_TIMEOUT


def run_proof(command: str, cwd: str | None, timeout: int | None = None) -> dict:
    """Run one proof command in ``cwd``; return the harness record."""
    timeout = timeout or proof_timeout()
    wd = cwd if cwd and os.path.isdir(cwd) else None
    started = time.time()
    try:
        cp = subprocess.run(command, shell=True, cwd=wd, capture_output=True, text=True,
                            timeout=timeout, stdin=subprocess.DEVNULL)
        exit_code, output, timed_out = cp.returncode, (cp.stdout or "") + (cp.stderr or ""), False
    except subprocess.TimeoutExpired as e:
        raw = (e.stdout or b"") + (e.stderr or b"")
        output = raw.decode(errors="replace") if isinstance(raw, bytes) else str(raw)
        exit_code, timed_out = 124, True
    return {"command": command, "cwd": wd, "exit_code": exit_code, "timed_out": timed_out,
            "output_tail": output[-OUTPUT_TAIL_CHARS:], "duration_s": round(time.time() - started, 3),
            "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runner": "harness"}


def run_sync_proofs(body: str | None, cwd: str | None) -> tuple[list[dict], dict | None]:
    """Run every non-async PROOF-CMD; return (records, first_failure_or_None)."""
    records = []
    for cmd, is_async in proof_cmds(body):
        if is_async:
            continue
        rec = run_proof(cmd, cwd)
        records.append(rec)
        if rec["exit_code"] != 0:
            return records, rec
    return records, None


_IMPL_HINT = re.compile(r"\b(implement|fix|feat|feature|build|add|refactor|bug|patch|ship)\w*\b", re.I)


def missing_proof_warning(title: str | None, body: str | None) -> str | None:
    """Advisory warning for implementation-type cards created without any PROOF line."""
    text = f"{title or ''}\n{body or ''}"
    if proof_commands(body) or proof_cmds(body) or not _IMPL_HINT.search(text):
        return None
    return ("implementation card has no PROOF line; add `PROOF-CMD: <command>` so the harness "
            "re-runs the proof on kanban_complete (or `PROOF: <command>` for an advisory proof).")


def rerun_pending_proofs(conn, kb=None) -> list[dict]:
    """Re-run async proofs for cards parked in ``review``; complete those that pass.

    A card is pending when its last ``proof_async_pending`` event is newer than any
    ``proof_async_result`` event. Failures are recorded and the card stays in review.
    """
    if kb is None:
        from hermes_cli import kanban_db as kb
    results = []
    for task in kb.list_tasks(conn, status="review"):
        events = kb.list_events(conn, task.id)
        pend = [e for e in events if e.kind == "proof_async_pending"]
        if not pend:
            continue
        done = [e for e in events if e.kind == "proof_async_result" and e.id > pend[-1].id
                and (e.payload or {}).get("passed")]
        if done:
            continue
        cmds = [c for c, a in proof_cmds(task.body) if a]
        recs = [run_proof(c, task.workspace_path) for c in cmds]
        passed = all(r["exit_code"] == 0 for r in recs)
        kb._append_event(conn, task.id, "proof_async_result", {"passed": passed, "proof_run": recs})
        if passed:
            kb.complete_task(conn, task.id, summary="async PROOF-CMD passed (harness re-run)",
                             metadata={"proof_run": recs}, force=True)
        results.append({"task_id": task.id, "passed": passed, "proof_run": recs})
    return results
