"""A recurring job whose run is in flight in ANOTHER process must not be re-dispatched by the
ticking process — and the refusal must never land in the ledger as a FAILED execution row.

Live incident (profile ``product-lead-vibebrowser``): 819 ledger rows carrying
``error = "Fire claim lost; execution was not started."`` — one per 60s tick, for the whole
~50 minutes the run was in flight — while the job's ``fire_claim`` was owned by a DIFFERENT
process (PID 18620) than the one stamping its ``pending_slot`` (PID 7209).

Two mechanisms conspired:

* ``cron.occurrences.unclaimed_pending_slot`` treated a slot stamped by THIS process as
  orphaned and restored it as the due instant on EVERY tick, so the job was due again
  immediately, forever, while the run was in flight.
* ``cron.scheduler._submit_with_guard`` only deduplicated against THIS process's running set,
  so the cross-process dispatch went through, ``create_execution`` wrote the ledger row, and
  ``claim_job_for_fire`` refused it ("Fire claim lost") — one failed row per tick.

The same restore had a second, quieter symptom: it re-stamped the job's dispatch record
(``last_dispatch`` — ``scheduled_at`` / ``dispatched_at`` / ``lateness_seconds`` / ``kind``) on
EVERY tick, always against the same stale occurrence, so the lateness grew with every tick and
``hermes cron status`` / ``hermes cron doctor`` reported the healthy in-flight job as perpetually
late ("catch-up after missed fire: scheduled 19:52:35, ran 20:44:02 - 51m late").

The fire claim is the durable cross-process "a run is happening right now" marker: the running
job heartbeats it every 60s, ``mark_job_run`` clears it when the run ends, and a claim whose
owner is provably dead expires at once, so the genuine #107485 recovery (a slot lost because
the process died before claiming) keeps working. The in-flight run re-anchors ``next_run_at``
when it completes, so the occurrence is not lost by not restoring the slot.

Drives the REAL store, the REAL tick and the REAL ledger under a throwaway HERMES_HOME — no
mocks of the code under test.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest

TICKS = 5


@pytest.fixture
def cron_home(tmp_path, monkeypatch):
    """Throwaway HERMES_HOME so jobs.json, the ledger and the scripts never touch the real store."""
    home = tmp_path / ".hermes"
    (home / "cron" / "output").mkdir(parents=True)
    (home / "scripts").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MACHINE_ID", raising=False)
    return home


@pytest.fixture(autouse=True)
def _clean_running_state():
    """The in-process running set is module-global; a leaked key would fake an in-flight run."""
    import cron.scheduler as sched

    sched._running_job_ids.clear()
    sched._running_since.clear()
    sched._running_futures.clear()
    yield
    sched._running_job_ids.clear()
    sched._running_since.clear()
    sched._running_futures.clear()


@pytest.fixture
def other_process():
    """A real, live process that is NOT this one — the owner of a cross-process fire claim."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - terminate already asked it to go
            proc.kill()
            proc.wait(timeout=10)


def _create_script_job(home: Path, name: str = "inflight"):
    """A recurring ``no_agent`` job whose script appends one line per fire.

    ``no_agent`` means the script IS the job, so nothing can call an LLM even if the job does
    get dispatched — the counter file is the ground truth for "the job ran".
    """
    import cron.jobs as J

    counter = home / "fires.txt"
    script = home / "scripts" / "fire.sh"
    script.write_text(f"#!/bin/sh\necho fired >> {counter}\necho fired\n", encoding="utf-8")
    script.chmod(0o755)
    job = J.create_job(prompt=None, schedule="every 1m", name=name, script="fire.sh",
                       no_agent=True, deliver="local")
    return job["id"], counter


def _stamp_live_fire_claim(job_id: str, pid: int) -> dict:
    """Stamp the job's fire claim as owned by a LIVE process on this host.

    ``by`` uses the same shape the store writes (``host:pid:token``); the pid belongs to a real
    running process, so ``_claim_owner_is_dead`` cannot call the claim stale.
    """
    import cron.jobs as J

    claim = {"at": J._hermes_now().isoformat(),
             "by": f"{socket.gethostname()}:{pid}:live-claim-token"}
    _patch_job(job_id, fire_claim=claim)
    return claim


def _patch_job(job_id: str, **fields) -> None:
    """Write *fields* onto the stored record for *job_id* (the real store, no mocks)."""
    import cron.jobs as J

    jobs = J.load_jobs()
    for record in jobs:
        if record.get("id") == job_id:
            record.update(fields)
    J.save_jobs(jobs)


def _make_due(job_id: str, minutes_ago: int = 5) -> str:
    """Push the job's next occurrence into the past so every tick sees it due."""
    import cron.jobs as J

    instant = (J._hermes_now() - timedelta(minutes=minutes_ago)).replace(
        microsecond=0).isoformat()
    _patch_job(job_id, next_run_at=instant)
    return instant


def _tick():
    """Run the REAL tick (admission, scan, advance, dispatch)."""
    from cron.scheduler_tick import tick

    return tick(verbose=False)


def _ledger_rows(job_id: str, settle_seconds: float = 1.0) -> list:
    """Poll the ledger until a row shows up or the window closes; return every row for *job_id*.

    The dispatch is asynchronous (``pool.submit``), so a refusal only shows up in the ledger a
    moment after the tick returns.
    """
    import cron.executions as E

    deadline = time.monotonic() + settle_seconds
    rows: list = []
    while True:
        time.sleep(0.2)
        rows = E.list_executions(job_id=job_id, limit=50)
        if rows or time.monotonic() >= deadline:
            return rows


def _fire_claim_lost_rows(rows) -> list:
    """Ledger rows whose error is the "the dispatch lost its claim" refusal."""
    return [row for row in rows if "Fire claim lost" in (row.get("error") or "")]


class TestCrossProcessInFlightRun:
    def test_live_claim_in_another_process_creates_no_failed_rows(self, cron_home, other_process):
        """The regression: a run in flight in another process must not be re-dispatched here.

        Before the fix every tick re-dispatched the job (the pending slot was restored as the
        due instant), the claim was refused inside the worker, and the refusal was recorded as a
        FAILED execution row — one per tick, for the whole life of the run.

        The other half of the live symptom: the same restore re-stamped the job's dispatch
        record on every tick, so the healthy in-flight job reported itself perpetually late
        ("catch-up after missed fire … 51m late") in ``hermes cron status`` / ``hermes cron doctor``.
        """
        from cron.jobs import get_job

        job_id, counter = _create_script_job(cron_home)
        _stamp_live_fire_claim(job_id, other_process.pid)
        _make_due(job_id)

        _tick()
        _ledger_rows(job_id, settle_seconds=0.5)
        # WHY snapshot the record here: the due scan stamps it ONCE, at the real due instant.
        # Before the fix the pending-slot restore re-stamped it on EVERY tick with a lateness
        # that grew with every tick (the restored slot was always the same stale occurrence) —
        # that growing lateness is exactly what made cron status/doctor call a healthy in-flight
        # job perpetually late.
        after_first = get_job(job_id).get("last_dispatch")
        assert after_first is not None, (
            "the due scan must stamp a dispatch record for the due occurrence; "
            f"got {after_first!r}"
        )

        for _ in range(TICKS - 1):
            _tick()
            _ledger_rows(job_id, settle_seconds=0.5)

        after_last = get_job(job_id).get("last_dispatch")
        assert after_last == after_first, (
            "the dispatch record must not be re-stamped on every tick: that is what made "
            "hermes cron status/doctor report a healthy in-flight job as 'catch-up after "
            f"missed fire'; first={after_first} last={after_last}")

        rows = _ledger_rows(job_id, settle_seconds=1.5)
        assert not _fire_claim_lost_rows(rows), (
            "a job with a live fire claim in another process must not be re-dispatched; "
            f"got {len(_fire_claim_lost_rows(rows))} bogus 'Fire claim lost' row(s)"
        )
        assert not rows, f"no execution row at all may be created for an in-flight job: {rows}"
        assert not counter.exists(), "the job must not run while another process holds its claim"

    def test_same_process_inflight_run_creates_no_failed_rows(self, cron_home):
        """The in-process dedup must keep working: a run held by THIS process is not re-dispatched."""
        import cron.scheduler as sched

        job_id, counter = _create_script_job(cron_home, name="same-process")
        _make_due(job_id)
        assert sched.try_register_running_job(job_id) is True
        try:
            for _ in range(TICKS):
                _tick()
                _ledger_rows(job_id, settle_seconds=0.5)
        finally:
            sched.release_running_job(job_id, home=sched._get_hermes_home())

        rows = _ledger_rows(job_id, settle_seconds=1.5)
        assert not _fire_claim_lost_rows(rows), (
            f"a job already running in this process must not be re-dispatched: {rows}"
        )
        assert not counter.exists(), "the job must not run twice"


class TestPendingSlotRestore:
    def test_unclaimed_slot_is_restored_once_without_a_live_claim(self, cron_home):
        """#107485 keeps working: a slot nobody claimed is restored as the due instant.

        The stamp is owned by THIS process (the dispatch was refused) and no run of the job is
        in flight, so the slot must come back — restoring it is how the occurrence survives.
        """
        import cron.jobs as J
        from cron.occurrences import unclaimed_pending_slot

        job_id, _counter = _create_script_job(cron_home, name="orphan-slot")
        slot = _make_due(job_id)
        _patch_job(job_id, pending_slot={
            "scheduled_at": slot, "at": J._hermes_now().isoformat(), "by": J._machine_id()})

        job = J.get_job(job_id)
        assert unclaimed_pending_slot(job, J._hermes_now()) == slot

    def test_unclaimed_slot_is_not_restored_while_a_live_claim_holds(self, cron_home, other_process):
        """The other branch: while a run of the job is in flight elsewhere, the slot stays put.

        Restoring it would make the job due again on every tick — the dispatch would be refused
        and recorded as a failed "Fire claim lost" row.
        """
        import cron.jobs as J
        from cron.occurrences import unclaimed_pending_slot

        job_id, _counter = _create_script_job(cron_home, name="live-slot")
        slot = _make_due(job_id)
        _patch_job(job_id, pending_slot={
            "scheduled_at": slot, "at": J._hermes_now().isoformat(), "by": J._machine_id()})
        _stamp_live_fire_claim(job_id, other_process.pid)

        job = J.get_job(job_id)
        assert unclaimed_pending_slot(job, J._hermes_now()) is None
