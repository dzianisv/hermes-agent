"""Deterministic gate in front of kanban notifier wakes (#17).

Every subscribed card event used to wake the supervisor's session at once, so
cron cards, alerts and unrelated cards kept pulling the EM off the outcome it
was working on. With ``kanban.wake_gate.enabled: true`` a wake is delivered
immediately only when it is about the active outcome or a production incident;
everything else is persisted and delivered as ONE digest wake per session at
the next checkpoint, with identical events (same card, same event kinds)
folded into a counted line.

Config (``kanban.wake_gate``):
  enabled            bool, default False (old behaviour: every wake is immediate)
  active_task_ids    card ids of the active outcome; a card whose ancestor is
                     listed also counts
  active_outcome_keys  ``tasks.outcome_key`` values of the active outcome
  incident_markers   case-insensitive substrings in title/event text that tag
                     a production incident (default below)
  digest_interval_seconds  checkpoint interval, default 1800
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_INCIDENT_MARKERS = ("[incident]", "incident:", "production incident", "prod incident", "sev1", "p0")
DEFAULT_DIGEST_INTERVAL = 1800


@dataclass
class WakeGateConfig:
    enabled: bool = False
    active_task_ids: frozenset = frozenset()
    active_outcome_keys: frozenset = frozenset()
    incident_markers: tuple = DEFAULT_INCIDENT_MARKERS
    digest_interval_seconds: int = DEFAULT_DIGEST_INTERVAL

    @classmethod
    def from_kanban_cfg(cls, kanban_cfg: Any) -> "WakeGateConfig":
        raw = (kanban_cfg or {}).get("wake_gate") if isinstance(kanban_cfg, dict) else None
        if not isinstance(raw, dict):
            return cls()

        def _set(key: str) -> frozenset:
            v = raw.get(key) or []
            if isinstance(v, str):
                v = [v]
            return frozenset(str(x).strip() for x in v if str(x).strip())

        markers = raw.get("incident_markers")
        markers = tuple(str(m).lower() for m in markers) if isinstance(markers, list) else DEFAULT_INCIDENT_MARKERS
        try:
            interval = max(0, int(raw.get("digest_interval_seconds", DEFAULT_DIGEST_INTERVAL)))
        except (TypeError, ValueError):
            interval = DEFAULT_DIGEST_INTERVAL
        return cls(enabled=bool(raw.get("enabled", False)), active_task_ids=_set("active_task_ids"),
                   active_outcome_keys=_set("active_outcome_keys"), incident_markers=markers,
                   digest_interval_seconds=interval)


def task_lineage(conn: sqlite3.Connection, task_id: str, *, max_depth: int = 8) -> list[str]:
    """``task_id`` plus its ancestors via ``task_links`` (bounded BFS)."""
    seen, frontier = [task_id], [task_id]
    for _ in range(max_depth):
        if not frontier:
            break
        q = ",".join("?" * len(frontier))
        rows = conn.execute(f"SELECT parent_id FROM task_links WHERE child_id IN ({q})", frontier).fetchall()
        frontier = [r[0] for r in rows if r[0] not in seen]
        seen.extend(frontier)
    return seen


def is_incident(cfg: WakeGateConfig, task: Any, events: Iterable[Any]) -> bool:
    texts = [str(getattr(task, "title", "") or "")]
    for ev in events:
        payload = getattr(ev, "payload", None)
        if isinstance(payload, dict):
            texts.extend(str(payload.get(k) or "") for k in ("reason", "summary", "tag", "severity"))
    hay = "\n".join(texts).lower()
    return any(m and m in hay for m in cfg.incident_markers)


def is_immediate(cfg: WakeGateConfig, d: dict, events: Iterable[Any]) -> bool:
    """True when this wake must not wait for the digest."""
    if not cfg.enabled:
        return True
    events = list(events)
    task = d.get("task")
    lineage = d.get("lineage") or [d["sub"]["task_id"]]
    if cfg.active_task_ids.intersection(lineage):
        return True
    if task is not None and getattr(task, "outcome_key", None) in cfg.active_outcome_keys:
        return True
    return is_incident(cfg, task, events)


@dataclass
class DigestStore:
    """Durable per-session queue of coalesced wakes (``<kanban_home>/kanban/wake_digest.db``)."""
    path: Path
    _ready: bool = field(default=False, init=False)

    def _conn(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=10)
        conn.row_factory = sqlite3.Row
        if not self._ready:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS wake_digest ("
                " route_key TEXT NOT NULL, dedup_key TEXT NOT NULL, route TEXT NOT NULL,"
                " line TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 1,"
                " first_at REAL NOT NULL, last_at REAL NOT NULL,"
                " PRIMARY KEY (route_key, dedup_key))")
            conn.commit()
            self._ready = True
        return conn

    def add(self, *, route_key: str, route: dict, dedup_key: str, line: str, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO wake_digest (route_key, dedup_key, route, line, count, first_at, last_at) "
                "VALUES (?, ?, ?, ?, 1, ?, ?) ON CONFLICT(route_key, dedup_key) DO UPDATE SET "
                "count = count + 1, last_at = excluded.last_at, line = excluded.line, route = excluded.route",
                (route_key, dedup_key, json.dumps(route, default=str), line, now, now))
            conn.commit()
        finally:
            conn.close()

    def due(self, interval: int, now: Optional[float] = None) -> list[tuple[str, dict, list[sqlite3.Row]]]:
        """Routes whose oldest pending item is at least ``interval`` old."""
        now = time.time() if now is None else now
        conn = self._conn()
        try:
            keys = [r[0] for r in conn.execute(
                "SELECT route_key FROM wake_digest GROUP BY route_key HAVING MIN(first_at) <= ?",
                (now - interval,))]
            out = []
            for key in keys:
                rows = conn.execute("SELECT * FROM wake_digest WHERE route_key = ? ORDER BY first_at", (key,)).fetchall()
                out.append((key, json.loads(rows[-1]["route"]), rows))
            return out
        finally:
            conn.close()

    def settle(self, route_key: str, rows: list[sqlite3.Row]) -> None:
        """Remove delivered rows; a row bumped after the read stays for the next digest."""
        conn = self._conn()
        try:
            for r in rows:
                conn.execute("DELETE FROM wake_digest WHERE route_key = ? AND dedup_key = ? AND last_at = ?",
                             (route_key, r["dedup_key"], r["last_at"]))
            conn.commit()
        finally:
            conn.close()


def default_store() -> DigestStore:
    from hermes_cli.kanban_db import kanban_home
    return DigestStore(kanban_home() / "kanban" / "wake_digest.db")


def route_key(sub: dict, session_key: str) -> str:
    return "|".join([sub.get("notifier_profile") or "", (sub.get("platform") or "").lower(),
                     str(sub.get("chat_id") or ""), str(sub.get("thread_id") or ""), session_key or ""])


def digest_text(rows: list[sqlite3.Row]) -> str:
    lines = [f"[kanban digest] {len(rows)} coalesced card update(s) since the last checkpoint "
             "(not about your active outcome; no incident). Review once, act only if needed:"]
    for r in rows:
        n = int(r["count"])
        lines.append(f"- {r['line']}" + (f" (x{n})" if n > 1 else ""))
    return "\n".join(lines)
