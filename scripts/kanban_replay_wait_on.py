"""Replay the last N days of a READ-ONLY kanban.db copy against the wait_on /
no-progress rules (hermes_cli/kanban_wait_on.py).

A "red-CI hold run" is a run that ended without moving the card forward
(blocked/scheduled/crashed/reclaimed/gave_up) whose summary or error says CI /
checks are red, failing or pending. For each card:

  R1 wait_on: the first hold run names its handle (gh run id, PR number + sha,
  or card id). A later hold run on the same card that re-reports the SAME
  handle would have been skipped: the card stays parked until the handle
  changes, and an identical handle means it had not.

  R2 no-progress: after 2 consecutive non-forward runs the card goes to the
  architect; further non-forward runs by the implementer before a forward
  outcome would not have been claimed.

Usage: python scripts/kanban_replay_wait_on.py <kanban.db> [days]
"""

from __future__ import annotations

import collections
import re
import sqlite3
import sys
import time

HOLD_RX = re.compile(
    r"(?i)(\bCI\b|checks?|run \d{9,}|workflow|release-smoke|deploy)[^.\n]{0,80}\b(red|fail\w*|pending|in.progress|queued)"
    r"|\b(red|failing)\b[^.\n]{0,40}\b(CI|checks?)")
NON_FORWARD = {"blocked", "scheduled", "crashed", "reclaimed", "gave_up", "timed_out"}
FORWARD = {"completed", "review_requested", "approved", "changes_requested"}
RUN_RX = re.compile(r"(?:run[s]?\s*(?:id\s*)?#?|actions/runs/)(\d{9,12})", re.I)
PR_RX = re.compile(r"(?:PR\s*#?|pull/|#)(\d{1,6})\b")
SHA_RX = re.compile(r"\b([0-9a-f]{7,40})\b")
CARD_RX = re.compile(r"\b(t_[0-9a-f]{8})\b")


def handles(text: str, own: str) -> frozenset:
    hs = {f"run:{m}" for m in RUN_RX.findall(text)}
    shas = SHA_RX.findall(text)
    for n in PR_RX.findall(text):
        hs.add(f"pr:{n}@{shas[0][:7] if shas else ''}")
    hs |= {f"card:{c}" for c in CARD_RX.findall(text) if c != own}
    return frozenset(hs)


def main(path: str, days: int = 30) -> None:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    since = int(time.time()) - days * 86400
    rows = conn.execute(
        "SELECT task_id, profile, outcome, COALESCE(summary,'') || ' ' || COALESCE(error,'') "
        "FROM task_runs WHERE started_at > ? AND ended_at IS NOT NULL ORDER BY task_id, started_at, id",
        (since,)).fetchall()
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r[0]].append(r)

    hold_runs = repeated = r1_skipped = r2_skipped = both = 0
    cards_with_repeats = set()
    for tid, runs in by_task.items():
        parked: frozenset | None = None   # handle set the card is parked on
        prev_hold = False
        streak = 0
        for _tid, _prof, outcome, text in runs:
            outcome = outcome or ""
            is_hold = outcome in NON_FORWARD and bool(HOLD_RX.search(text))
            hs = handles(text, tid) if is_hold else frozenset()
            s1 = s2 = False
            if is_hold:
                hold_runs += 1
                if prev_hold:
                    repeated += 1
                    cards_with_repeats.add(tid)
                    # R1: same handle re-reported -> the card would still be parked.
                    s1 = parked is not None and bool(hs) and bool(hs & parked)
            # R2: third+ consecutive non-forward run would go to the architect.
            if outcome in NON_FORWARD:
                s2 = streak >= 2
                streak += 1
            else:
                streak = 0
            if is_hold and prev_hold:
                r1_skipped += s1
                r2_skipped += s2
                both += (s1 or s2)
            if is_hold:
                parked = hs or parked
                prev_hold = True
            elif outcome in FORWARD:
                parked = None
                prev_hold = False
    print(f"window_days={days} runs={len(rows)} cards={len(by_task)}")
    print(f"red_ci_hold_runs={hold_runs}")
    print(f"repeated_red_ci_hold_runs={repeated} on {len(cards_with_repeats)} cards")
    print(f"skipped_by_R1_wait_on_same_handle={r1_skipped}")
    print(f"skipped_by_R2_no_progress_respec={r2_skipped}")
    print(f"skipped_by_R1_or_R2={both} ({(100 * both / repeated) if repeated else 0:.0f}% of repeats)")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30)
