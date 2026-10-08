"""Replay live kanban history against the run-loop limits (read-only).

Usage: python scripts/kanban_replay_loop_limits.py [path/to/kanban.db]
For each task, walks its runs in order and reports the first point where
review_round_limit (3 change requests) or active_seconds_limit (24h summed run
time) would have parked it, and how many later runs that avoids.
"""
from __future__ import annotations

import collections
import sqlite3
import sys
from pathlib import Path

ROUNDS, ACTIVE = 3, 86400


def replay(db: str) -> dict:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    runs = collections.defaultdict(list)
    for tid, rid, s, e, outcome in c.execute(
        "SELECT task_id, id, started_at, ended_at, outcome FROM task_runs ORDER BY id"
    ):
        runs[tid].append((rid, s, e, outcome))
    out = {"tasks": len(runs), "parked_by_rounds": 0, "parked_by_active": 0, "runs_avoided": 0, "worst": []}
    for tid, rs in runs.items():
        spent = cr = 0
        for i, (rid, s, e, outcome) in enumerate(rs):
            if outcome == "changes_requested":
                cr += 1
                if cr >= ROUNDS:
                    out["parked_by_rounds"] += 1
                    out["runs_avoided"] += len(rs) - i - 1
                    out["worst"].append((len(rs) - i - 1, tid, "rounds"))
                    break
            if s and e:
                spent += e - s
            if spent >= ACTIVE and i + 1 < len(rs):
                out["parked_by_active"] += 1
                out["runs_avoided"] += len(rs) - i - 1
                out["worst"].append((len(rs) - i - 1, tid, "active"))
                break
    out["worst"] = sorted(out["worst"], reverse=True)[:5]
    return out


if __name__ == "__main__":
    print(replay(sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / ".hermes/kanban.db")))
