"""Exact scheduled identities, independent of mutable jobs.json dispatch stamps."""
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)


def scheduled_instant(value):
    """Canonicalize aware instants; legacy/ambiguous values carry no exact identity."""
    if not isinstance(value, str):
        return None
    try:
        instant = datetime.fromisoformat(value)
        if instant.tzinfo is None:
            return None
        return instant.astimezone(timezone.utc).isoformat()
    except ValueError:
        return None


def completed_occurrence(job, instant):
    """Unknown/failed/pruned attempts cannot prove completion: keep them eligible."""
    from cron.constants import FIRE_CLAIM_SKEW_SECONDS
    from cron.executions import _transaction

    instant = scheduled_instant(instant)
    if instant is None:
        return False
    # A skewed early fire (see claim_job_for_fire) legitimately completes just before its slot.
    earliest_real = datetime.fromisoformat(instant) - timedelta(seconds=FIRE_CLAIM_SKEW_SECONDS)
    try:
        with _transaction() as conn:
            rows = conn.execute(
                "SELECT id, finished_at, claimed_at FROM executions "
                "WHERE job_id=? AND scheduled_instant=? "
                "AND status='completed'", (str(job['id']), instant)
            ).fetchall()
        for row in rows:
            completed_at = scheduled_instant(row["finished_at"] or row["claimed_at"])
            # Legacy or malformed timestamps remain proof; only positively identified poison
            # rows — completions recorded before their claimed occurrence — are ignored.
            if completed_at is None or datetime.fromisoformat(completed_at) >= earliest_real:
                # Both dedup gates (due scan and fire claim) consume the slot on True without a
                # run or a ledger row, so this line is the only trace the skip leaves (#111414).
                logger.warning(
                    "Job '%s' (%s): scheduled occurrence %s was already completed by execution "
                    "%s (finished %s); skipping the due slot without a new run",
                    job.get("name", job.get("id")), job.get("id"), instant, row["id"],
                    row["finished_at"] or row["claimed_at"])
                return True
        return False
    except Exception:
        logger.warning("Cannot check completed occurrence for job %s", job['id'], exc_info=True)
        return False


# --- Pending slot: the occurrence a tick took off the schedule but has not yet claimed ---
#
# The tick advances a recurring job's ``next_run_at`` BEFORE dispatch (at-most-once: a crash
# mid-run must not re-fire on every restart). That leaves a window — advance done, fire claim
# not yet taken (interpreter finalizing, executor refusing work, process killed) — in which the
# process exiting loses the occurrence silently: the restarted scan sees a future ``next_run_at``
# and nothing ever ran (#107485). ``pending_slot`` is the durable record of that window: the due
# scan stamps it with the exact stored instant plus the stamping owner, ``claim_job_for_fire``
# (the point after which side effects may exist) clears it, and any explicit rewrite of the
# schedule drops it. A slot still pending once its owner is provably gone (or its lease has
# expired) was never claimed, so it is restored ONCE as the due instant and then flows through
# the ordinary late / fast-forward / ``cron.catch_up_missed`` policy — never N replays.

def pending_slot_stamp(next_run, now):
    """Store value for a recurring occurrence about to be handed to the dispatcher."""
    from cron.jobs import _machine_id

    return {"scheduled_at": next_run, "at": now.isoformat(), "by": _machine_id()}


def unclaimed_pending_slot(job, now):
    """Stored instant of a slot the dispatcher never claimed, or None.

    None for non-recurring jobs and malformed stamps (never a fire), for a job still in this
    process's running set (its queued worker will claim and clear the slot itself), and for a job
    whose own fire claim is live (a run of it is in flight in ANOTHER process sharing this store).
    A stamp by THIS process on a job not running here is orphaned (dispatch refused). A stamp by
    another process is honoured while that owner may still be alive within the fire-claim lease — a
    second live gateway on the same store is mid-dispatch, not dead."""
    from cron.constants import FIRE_CLAIM_TTL_SECONDS
    from cron.jobs import _claim_is_live, _job_running_in_this_process, _machine_id

    pending = job.get("pending_slot")
    if not isinstance(pending, dict):
        return None
    slot = pending.get("scheduled_at")
    if job.get("schedule", {}).get("kind") not in {"cron", "interval"} or not isinstance(slot, str):
        return None
    try:
        datetime.fromisoformat(slot)
    except ValueError:
        return None
    if _job_running_in_this_process(str(job.get("id", ""))):
        return None
    # A run of THIS job is in flight. The in-process running set above only sees THIS
    # process; the job's own fire claim is the durable cross-process "a run is happening
    # right now" marker — the running job heartbeats it every 60s (heartbeat_fire_claim),
    # mark_job_run clears it when the run ends, and a claim whose owner is provably dead
    # expires at once (_claim_owner_is_dead). Restoring the slot while that claim is live
    # makes the job due again on EVERY tick: the tick re-dispatches it, the live claim
    # refuses it, and the refusal is recorded as a failed execution row ("Fire claim
    # lost") — one bogus row per tick for as long as the run lasts, plus a dispatch record
    # whose lateness grows with every tick ("catch-up after missed fire … 51m late"). The
    # in-flight run re-anchors next_run_at when it completes, so the occurrence is not lost
    # by not restoring it here.
    if _claim_is_live(job.get("fire_claim"), now, FIRE_CLAIM_TTL_SECONDS):
        return None
    if pending.get("by") != _machine_id() and _claim_is_live(pending, now, FIRE_CLAIM_TTL_SECONDS):
        return None
    return slot
