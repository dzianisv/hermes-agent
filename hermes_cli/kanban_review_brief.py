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
        parts.append(SINGLE_PASS_INSTRUCTION)
        return "\n\n".join(parts)
    except Exception as exc:
        return (f"REVIEW BRIEF for kanban task {getattr(task, 'id', '?')}.\n\n"
                f"design unavailable: brief build failed ({type(exc).__name__})\n\n{SINGLE_PASS_INSTRUCTION}")
