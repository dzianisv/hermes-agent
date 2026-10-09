"""Facts-hash gate: skip the agent run and delivery when a job's facts are unchanged.

Opt-in per job with ``facts_hash: true``. The job's existing pre-run ``script`` output (success
or failure text, exactly what the agent would see) is the "facts". Volatile fragments (clocks,
ages) can be stripped before hashing with ``facts_ignore``: a list of regexes, each match
removed. If the hash equals ``facts_state.last_hash`` — the hash of the last run that completed
and delivered (or deliberately stayed silent) — the tick is skipped with no LLM call and no
delivery. The hash is stored only after a successful run, so a failed run or failed delivery
retries the same facts next tick. A manual run (``cronjob run`` / ``hermes cron run``) always
runs. Jobs without ``facts_hash`` are unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

SKIP_LOG = "skipped: facts unchanged"
PENDING_KEY = "_facts_pending_hash"


def job_uses_facts_hash(job: dict) -> bool:
    return bool(job.get("facts_hash")) and bool(str(job.get("script") or "").strip())


def is_manual_run(job: dict, extra_prompt: Optional[str]) -> bool:
    return bool(extra_prompt) or bool(job.get("manual_run_at"))


def facts_hash(output: str, ignore: Optional[list] = None) -> str:
    text = output or ""
    for pattern in ignore or []:
        try:
            text = re.sub(pattern, "", text)
        except re.error as exc:
            logger.warning("facts_ignore pattern %r invalid: %s", pattern, exc)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def last_hash(job: dict) -> Optional[str]:
    state = job.get("facts_state")
    return state.get("last_hash") if isinstance(state, dict) else None


def check(job: dict, script_result: tuple, extra_prompt: Optional[str]) -> bool:
    """Return True when the tick must be skipped. Otherwise stash the new hash on the in-memory
    job so ``record_delivered`` can persist it after a successful run."""
    if not job_uses_facts_hash(job):
        return False
    ran_ok, output = script_result
    new = facts_hash(f"{'ok' if ran_ok else 'fail'}\n{output}", job.get("facts_ignore"))
    if not is_manual_run(job, extra_prompt) and new == last_hash(job):
        return True
    job[PENDING_KEY] = new
    return False


def record_delivered(job: dict) -> None:
    """Persist the pending hash after a run that succeeded and delivered (or chose silence)."""
    new = job.pop(PENDING_KEY, None)
    if not new:
        return
    try:
        from cron.jobs import _hermes_now, update_job

        update_job(job["id"], {"facts_state": {
            "last_hash": new, "last_delivered_at": _hermes_now().isoformat()}})
    except Exception as exc:
        logger.warning("facts_hash: failed to persist state for %r: %s", job.get("id"), exc)
