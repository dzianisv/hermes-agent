"""Single-pass reviewer brief: what a review-stage worker gets in its first prompt.

A reviewer that only sees ``work kanban task <id>`` reads the card, finds the first
defect, hands back, and the card loops one defect per round. The dispatcher therefore
inlines (a) the DESIGN section the card's ``DESIGN:`` line links to (fetched from Notion),
(b) the card's SCOPE / ACCEPTANCE / PROOF / R1..Rn lines, and (c) an explicit
one-complete-pass instruction. Every failure to fetch the design is stated in the brief as
``design unavailable: <reason>`` — never dropped silently. Never raises.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

SINGLE_PASS_INSTRUCTION = (
    "Do ONE complete pass: list every defect you find in this run, grouped by requirement; "
    "do not stop at the first issue."
)

REVIEW_ASSIGNEES = frozenset({"reviewer"})
REVIEW_STEP_KEYS = frozenset({"review"})

_NOTION_VERSION = "2022-06-28"
_DESIGN_MAX_CHARS = 12_000
_REQ_LINE_RE = re.compile(r"^\s*(?:[-*>]\s*)?\**(SCOPE|ACCEPTANCE|PROOF|R\d+)\**\s*[:.)\-–—]", re.I)
_DESIGN_LINE_RE = re.compile(r"DESIGN:[ \t]*(\S.*)")
_HEX32 = re.compile(r"[0-9a-f]{32}")
_CACHE_TTL_SECONDS = 600
# (page_id, block_id) -> (fetched_at, text-or-None, reason)
_design_cache: dict[tuple[str, str], tuple[float, Optional[str], str]] = {}


def is_review_spawn(task, lane: str) -> bool:
    """A worker whose job is review: the review lane, the reviewer profile, or a review step."""
    return (lane == "review"
            or (task.assignee or "") in REVIEW_ASSIGNEES
            or (task.current_step_key or "") in REVIEW_STEP_KEYS)


def _card_texts(conn: sqlite3.Connection, task) -> list[str]:
    from hermes_cli import kanban_db as kb

    texts = [task.body or ""]
    try:
        texts += [c.body or "" for c in kb.list_comments(conn, task.id)]
    except Exception:
        pass
    return texts


def requirement_lines(texts: list[str]) -> list[str]:
    """SCOPE/ACCEPTANCE/PROOF/R1..Rn lines, first occurrence of each text, card order."""
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        for line in text.splitlines():
            if _REQ_LINE_RE.match(line):
                s = line.strip()
                if s not in seen:
                    seen.add(s)
                    out.append(s)
    return out


_R_ID_RE = re.compile(r"^\s*(?:[-*>]\s*)?\**(R\d+|ACCEPTANCE)\**\s*[:.)\-–—]", re.I)
FINDING_VERDICTS = frozenset({"PASS", "FAIL"})
_DIFF_MAX_CHARS = 40_000


def requirement_ids(texts: list[str]) -> list[str]:
    """Ids a change request must cover: ``R<n>`` per R-line, ``ACCEPTANCE`` (or
    ``ACCEPTANCE-<k>`` when the card has several) per ACCEPTANCE line. Card order."""
    ids: list[str] = []
    acceptance = 0
    lines = requirement_lines(texts)
    n_acc = sum(1 for ln in lines if (_R_ID_RE.match(ln) or [None, ""])[1].upper() == "ACCEPTANCE")
    for line in lines:
        m = _R_ID_RE.match(line)
        if not m:
            continue
        tag = m.group(1).upper()
        if tag == "ACCEPTANCE":
            acceptance += 1
            tag = "ACCEPTANCE" if n_acc == 1 else f"ACCEPTANCE-{acceptance}"
        if tag not in ids:
            ids.append(tag)
    return ids


def check_findings(texts: list[str], metadata) -> Optional[str]:
    """None when ``metadata.findings`` covers every required id with a valid
    ``{id, verdict PASS|FAIL, evidence}``; otherwise the refusal text. Cards with no
    R/ACCEPTANCE lines need no findings."""
    required = requirement_ids(texts)
    if not required:
        return None
    findings = (metadata or {}).get("findings") if isinstance(metadata, dict) else None
    if not isinstance(findings, list):
        return ("metadata.findings is required: a list of {id, verdict: PASS|FAIL, evidence} "
                f"covering every card requirement. Missing ids: {', '.join(required)}")
    covered: set[str] = set()
    bad: list[str] = []
    for f in findings:
        if not isinstance(f, dict):
            bad.append(f"non-object finding {f!r}"[:80])
            continue
        fid = str(f.get("id") or "").strip().upper()
        verdict = str(f.get("verdict") or "").strip().upper()
        evidence = str(f.get("evidence") or "").strip()
        if verdict not in FINDING_VERDICTS:
            bad.append(f"{fid or '?'}: verdict must be PASS or FAIL")
        elif not evidence:
            bad.append(f"{fid or '?'}: evidence is empty")
        elif fid:
            covered.add(fid)
    missing = [i for i in required if i not in covered]
    if not missing and not bad:
        return None
    msg = []
    if missing:
        msg.append(f"metadata.findings is missing ids: {', '.join(missing)}")
    if bad:
        msg.append("invalid findings: " + "; ".join(bad))
    return ". ".join(msg) + f". Required ids: {', '.join(required)}"


def prior_review(conn: sqlite3.Connection, task_id: str) -> Optional[dict]:
    """Payload of the latest ``changes_requested`` event (round 2+), else None."""
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'changes_requested' "
        "ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    if row is None:
        return None
    try:
        data = json.loads(row[0] or "{}")
    except (TypeError, ValueError):
        data = {}
    return data if isinstance(data, dict) else {}


def rereview_model() -> tuple[Optional[str], Optional[str]]:
    """``(kanban.rereview_model, kanban.rereview_provider)`` from config; both optional."""
    try:
        from hermes_cli.kanban_db import _kanban_cfg
        cfg = _kanban_cfg()
    except Exception:
        return None, None
    m = str(cfg.get("rereview_model") or "").strip() or None
    p = str(cfg.get("rereview_provider") or "").strip() or None
    return m, (p if m else None)


def _git_diff_since(workspace: Optional[str], sha: str) -> tuple[Optional[str], str]:
    import subprocess
    if not workspace or not Path(workspace).is_dir():
        return None, "no workspace to diff in"
    try:
        r = subprocess.run(["git", "-C", workspace, "diff", f"{sha}..HEAD"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    except Exception as exc:
        return None, f"git diff failed: {type(exc).__name__}"
    if r.returncode != 0:
        return None, f"git diff {sha[:12]}..HEAD failed: {(r.stderr or '').strip()[:160]}"
    text = r.stdout
    if not text.strip():
        return "", ""
    if len(text) > _DIFF_MAX_CHARS:
        text = text[:_DIFF_MAX_CHARS] + f"\n… [diff truncated at {_DIFF_MAX_CHARS} chars]"
    return text, ""


def _incremental_section(prior: dict, workspace: Optional[str], diff_fn) -> str:
    rnd = prior.get("round")
    out = [f"RE-REVIEW (round {int(rnd) + 1 if isinstance(rnd, int) else '2+'}). "
           "Prior review findings:"]
    findings = prior.get("findings")
    if isinstance(findings, list) and findings:
        for f in findings:
            if isinstance(f, dict):
                out.append(f"- {f.get('id')}: {f.get('verdict')} — {f.get('evidence')}")
    else:
        out.append(f"(no structured findings recorded) reason: {prior.get('reason') or '-'}")
    sha = str(prior.get("reviewed_sha") or "").strip()
    if not sha:
        out.append("No reviewed sha was recorded for the prior round; incremental diff "
                   "unavailable — fall back to a full review.")
    else:
        diff, why = diff_fn(workspace, sha)
        if diff is None:
            out.append(f"Incremental diff since {sha} unavailable ({why}); fall back to a full review.")
        elif not diff:
            out.append(f"No changes since last reviewed sha {sha}.")
        else:
            out.append(f"Review ONLY the changes since last reviewed sha {sha}, re-checking each "
                       f"FAIL above; PASS items need re-checking only if the diff touches them:\n{diff}")
    return "\n".join(out)


def design_link(texts: list[str]) -> Optional[str]:
    """The newest ``DESIGN:`` line that carries a Notion id (later comments override the body)."""
    found = None
    for text in texts:
        for value in _DESIGN_LINE_RE.findall(text):
            if _HEX32.search(value.replace("-", "").lower()):
                found = value.strip()
    return found


def _parse_link(link: str) -> tuple[str, str]:
    """(page_id, block_id) from a Notion URL; block_id is the ``#anchor`` section or ''."""
    base, _, anchor = link.partition("#")
    page_ids = _HEX32.findall(base.replace("-", "").lower())
    block_ids = _HEX32.findall(anchor.replace("-", "").lower())
    page = page_ids[-1] if page_ids else (block_ids[0] if block_ids else "")
    return page, (block_ids[0] if block_ids else "")


def _notion_token() -> Optional[str]:
    tok = (os.environ.get("NOTION_TOKEN") or "").strip()
    if tok:
        return tok
    env_file = Path.home() / ".env.d" / "notion.env"
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            if line.startswith("NOTION_TOKEN="):
                return line.split("=", 1)[1].strip().strip("'\"") or None
    except OSError:
        return None
    return None


def _api_base() -> str:
    return (os.environ.get("HERMES_NOTION_API_BASE") or "https://api.notion.com").rstrip("/")


def _get_json(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}", "Notion-Version": _NOTION_VERSION,
    })
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 - fixed https base
        return json.loads(resp.read().decode("utf-8"))


def _children(block_id: str, token: str) -> list[dict]:
    out: list[dict] = []
    cursor = None
    for _ in range(20):  # 2,000 blocks is far past any design section
        url = f"{_api_base()}/v1/blocks/{block_id}/children?page_size=100"
        if cursor:
            url += f"&start_cursor={cursor}"
        data = _get_json(url, token)
        out += data.get("results") or []
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
    return out


def _block_text(block: dict) -> str:
    kind = block.get("type") or ""
    body = block.get(kind) or {}
    text = "".join(rt.get("plain_text", "") for rt in body.get("rich_text") or [])
    prefix = {"heading_1": "# ", "heading_2": "## ", "heading_3": "### ",
              "bulleted_list_item": "- ", "numbered_list_item": "1. ", "to_do": "- [ ] "}.get(kind, "")
    return prefix + text if text else ""


def _heading_level(block: dict) -> Optional[int]:
    kind = block.get("type") or ""
    return int(kind[-1]) if kind.startswith("heading_") and kind[-1].isdigit() else None


def _section(blocks: list[dict], anchor: str) -> Optional[list[dict]]:
    """The anchored heading plus every following block up to the next same-or-higher heading."""
    for i, b in enumerate(blocks):
        if (b.get("id") or "").replace("-", "").lower() != anchor:
            continue
        level = _heading_level(b)
        if level is None:
            return [b]
        out = [b]
        for nxt in blocks[i + 1:]:
            nl = _heading_level(nxt)
            if nl is not None and nl <= level:
                break
            out.append(nxt)
        return out
    return None


def _fetch_design(link: str) -> tuple[Optional[str], str]:
    page, anchor = _parse_link(link)
    if not page:
        return None, "DESIGN: line names no Notion page id"
    key = (page, anchor)
    hit = _design_cache.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL_SECONDS:
        return hit[1], hit[2]
    token = _notion_token()
    if not token:
        return None, "NOTION_TOKEN not set (env or ~/.env.d/notion.env)"
    try:
        blocks = _children(page, token)
        if anchor:
            picked = _section(blocks, anchor)
            if picked is None:
                text, reason = None, f"section {anchor} not found on page {page}"
                _design_cache[key] = (time.time(), text, reason)
                return text, reason
            blocks = picked
        lines = [t for t in (_block_text(b) for b in blocks) if t]
        if not lines:
            text, reason = None, f"Notion page {page} has no readable text"
        else:
            text = "\n".join(lines)
            if len(text) > _DESIGN_MAX_CHARS:
                text = text[:_DESIGN_MAX_CHARS] + f"\n… [design truncated at {_DESIGN_MAX_CHARS} chars]"
            reason = ""
    except urllib.error.HTTPError as exc:
        text, reason = None, f"Notion HTTP {exc.code} for {page}"
    except Exception as exc:  # network, JSON, timeout
        text, reason = None, f"Notion fetch failed: {type(exc).__name__}: {exc}"[:200]
    _design_cache[key] = (time.time(), text, reason)
    return text, reason


def build_review_brief(
    conn: sqlite3.Connection, task, *, fetch: Optional[Callable[[str], tuple]] = None,
    workspace: Optional[str] = None, diff_fn: Optional[Callable] = None,
) -> str:
    """The reviewer brief appended to the worker's first prompt. Never raises."""
    try:
        texts = _card_texts(conn, task)
        parts = [f"REVIEW BRIEF for kanban task {task.id}."]
        link = design_link(texts)
        if link is None:
            parts.append("design unavailable: card has no DESIGN: line with a Notion link")
        else:
            text, reason = (fetch or _fetch_design)(link)
            if text:
                parts.append(f"DESIGN section ({link}):\n{text}")
            else:
                parts.append(f"design unavailable: {reason or 'unknown error'}")
        reqs = requirement_lines(texts)
        parts.append("Card requirements:\n" + ("\n".join(reqs) if reqs
                                               else "(card has no SCOPE/ACCEPTANCE/PROOF/R lines)"))
        ids = requirement_ids(texts)
        if ids:
            parts.append("kanban_request_changes requires metadata.findings: one "
                         "{id, verdict: PASS|FAIL, evidence} per id: " + ", ".join(ids)
                         + "; also pass metadata.reviewed_sha (the head you reviewed).")
        prior = prior_review(conn, task.id)
        if prior is not None:
            parts.append(_incremental_section(
                prior, workspace or getattr(task, "workspace_path", None),
                diff_fn or _git_diff_since))
        parts.append(SINGLE_PASS_INSTRUCTION)
        return "\n\n".join(parts)
    except Exception as exc:
        return (f"REVIEW BRIEF for kanban task {getattr(task, 'id', '?')}.\n\n"
                f"design unavailable: brief build failed ({type(exc).__name__})\n\n{SINGLE_PASS_INSTRUCTION}")
