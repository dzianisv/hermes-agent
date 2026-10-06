"""Disk-aware kanban dispatch guard (t_7938d84b).

Free disk at or below ``floor + one swapfile quantum`` means the next macOS
swap growth can fill the disk, so the dispatcher spawns nothing; within two
quanta (plus one per freshly-created swapfile) it spawns at most one. With no
quantum (Linux) the thresholds are ``floor`` / ``2*floor``. Unknown free space
imposes no restriction (fail-open).
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import config as hermes_config

_REAL_LOAD_CONFIG_READONLY = hermes_config.load_config_readonly

GIB = 1024 ** 3
FLOOR = GIB // 2  # default 0.5 GiB
Q = GIB  # 1 GiB swapfile quantum


@pytest.fixture(autouse=True)
def _default_floor(monkeypatch):
    monkeypatch.delenv("DISK_GUARD_FLOOR_GI", raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _sample(free, q=Q, fresh=0, files=None):
    return {
        "free_bytes": free, "quantum_bytes": q, "swap_fresh_files": fresh,
        "swap_files": fresh if files is None else files,
    }


def _dispatch_three(monkeypatch, sample):
    monkeypatch.setattr(kbd, "_disk_sample", lambda *a, **k: sample)
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        ids = [kb.create_task(conn, title=t, assignee="alice") for t in ("a", "b", "c")]
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        statuses = {tid: kb.get_task(conn, tid).status for tid in ids}
    return res, spawns, statuses


# ---------------------------------------------------------------------------
# _disk_pressure_level thresholds
# ---------------------------------------------------------------------------


def test_level_unknown_without_sample():
    assert kbd._disk_pressure_level({}) == "unknown"


def test_level_with_quantum_boundaries():
    assert kbd._disk_pressure_level(_sample(FLOOR + Q // 2)) == "critical"
    assert kbd._disk_pressure_level(_sample(FLOOR + Q)) == "critical"
    assert kbd._disk_pressure_level(_sample(FLOOR + Q + 1)) == "elevated"
    assert kbd._disk_pressure_level(_sample(FLOOR + 2 * Q)) == "elevated"
    assert kbd._disk_pressure_level(_sample(FLOOR + 2 * Q + 1)) == "ok"
    # Fresh swapfiles widen the elevated band by one quantum each.
    assert kbd._disk_pressure_level(_sample(FLOOR + Q + Q // 2, fresh=2)) == "elevated"
    assert kbd._disk_pressure_level(_sample(FLOOR + 3 * Q, fresh=2)) == "elevated"
    assert kbd._disk_pressure_level(_sample(FLOOR + 4 * Q + 1, fresh=2)) == "ok"


def test_level_without_quantum_uses_floor_only():
    assert kbd._disk_pressure_level(_sample(FLOOR, q=0)) == "critical"
    assert kbd._disk_pressure_level(_sample(FLOOR + 1, q=0)) == "elevated"
    assert kbd._disk_pressure_level(_sample(2 * FLOOR, q=0)) == "elevated"
    assert kbd._disk_pressure_level(_sample(2 * FLOOR + 1, q=0)) == "ok"


def test_level_linux_shaped_sample_uses_floor_only():
    linux = {"free_bytes": FLOOR + 1, "swap_used_bytes": 3 * GIB}
    assert kbd._disk_pressure_level(linux) == "elevated"
    linux["free_bytes"] = FLOOR
    assert kbd._disk_pressure_level(linux) == "critical"
    linux["free_bytes"] = 2 * FLOOR + 1
    assert kbd._disk_pressure_level(linux) == "ok"


def test_floor_default_and_env(monkeypatch):
    assert kbd._disk_floor_bytes() == FLOOR
    monkeypatch.setenv("DISK_GUARD_FLOOR_GI", "2")
    assert kbd._disk_floor_bytes() == 2 * GIB
    # 1.5 GiB free: ok at the 0.5 GiB default floor, critical at a 2 GiB floor.
    assert kbd._disk_pressure_level(_sample(GIB + GIB // 2, q=0)) == "critical"
    for bad in ("0", "-1", "abc", "1e308", "inf", "nan"):
        monkeypatch.setenv("DISK_GUARD_FLOOR_GI", bad)
        assert kbd._disk_floor_bytes() == FLOOR


def test_floor_overflowing_config_falls_through_without_raising(monkeypatch):
    for bad in (1e308, float("inf"), float("nan"), "1e308"):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", lambda bad=bad: {"kanban": {"disk_floor_gi": bad}}
        )
        assert kbd._disk_floor_bytes() == FLOOR


def test_floor_read_from_real_config_yaml(kanban_home, monkeypatch):
    """``kanban.disk_floor_gi`` in the profile's config.yaml reaches the guard
    through the real loader (registered in DEFAULT_CONFIG, read at call time)."""
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _REAL_LOAD_CONFIG_READONLY)
    hermes_config._LOAD_CONFIG_CACHE.clear()
    (kanban_home / "config.yaml").write_text("kanban:\n  disk_floor_gi: 2\n")
    assert kbd._disk_floor_bytes() == 2 * GIB


def test_floor_config_wins_over_env(monkeypatch):
    monkeypatch.setenv("DISK_GUARD_FLOOR_GI", "2")
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: {"kanban": {"disk_floor_gi": 3}}
    )
    assert kbd._disk_floor_bytes() == 3 * GIB
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: {"kanban": {"disk_floor_gi": 0}}
    )
    assert kbd._disk_floor_bytes() == 2 * GIB


# ---------------------------------------------------------------------------
# dispatch_once under disk pressure
# ---------------------------------------------------------------------------


def test_dispatch_unknown_disk_spawns_full_budget(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    res, spawns, _ = _dispatch_three(monkeypatch, {})
    assert len(spawns) == 3
    assert res.disk_pressure is None


def test_dispatch_critical_disk_spawns_nothing_and_defers(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    res, spawns, statuses = _dispatch_three(monkeypatch, _sample(FLOOR + Q // 2))
    assert not spawns
    assert not res.spawned
    assert res.disk_pressure == "critical"
    assert set(statuses.values()) == {"ready"}
    assert "disk_pressure=critical" in kbd.describe_suppression([res])


def test_dispatch_critical_disk_log_names_swap_used_total_and_fresh(
    kanban_home, all_assignees_spawnable, monkeypatch, caplog,
):
    sample = _sample(FLOOR + Q // 2, fresh=2, files=4)
    sample["swap_used_bytes"] = int(3.51 * GIB)
    with caplog.at_level(logging.WARNING):
        _dispatch_three(monkeypatch, sample)
    assert (
        "disk pressure critical (free 1.00 GiB, floor 0.50 GiB, swapfile quantum 1.00 GiB, "
        "swap used 3.51 GiB in 4 swapfiles (2 created in last hour))"
    ) in caplog.text


def test_detail_without_swap_used_still_counts_swapfiles():
    detail = kbd._disk_pressure_detail(_sample(FLOOR, fresh=1, files=3))
    assert "swap used" not in detail
    assert detail.endswith("3 swapfiles (1 created in last hour)")


def test_dispatch_elevated_disk_spawns_exactly_one(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    res, spawns, _ = _dispatch_three(monkeypatch, _sample(FLOOR + Q + Q // 2, fresh=2))
    assert len(spawns) == 1
    assert res.disk_pressure == "elevated"


# ---------------------------------------------------------------------------
# _disk_sample probes
# ---------------------------------------------------------------------------


def _fake_swapfiles(swap_dir: Path):
    now = time.time()
    for name, size, age in (
        ("swapfile0", 1 * GIB, 7200),
        ("swapfile1", 2 * GIB, 600),
        ("swapfile2", 1 * GIB, 60),
    ):
        path = swap_dir / name
        with open(path, "wb") as fh:
            fh.truncate(size)  # sparse: no real disk used
        os.utime(path, (now - age, now - age))
    (swap_dir / "sleepimage").write_bytes(b"x")


@pytest.mark.real_disk_guard
def test_disk_sample_macos_parse(kanban_home, tmp_path, monkeypatch):
    swap_dir = tmp_path / "vm"
    swap_dir.mkdir()
    _fake_swapfiles(swap_dir)
    monkeypatch.setattr(kbd, "_SWAP_DIR", swap_dir)

    def fake_run(cmd, **kwargs):
        assert cmd == ["sysctl", "-n", "vm.swapusage"]
        assert kwargs.get("timeout") == 5
        return subprocess.CompletedProcess(
            cmd, 0, stdout="total = 4096.00M  used = 2741.50M  free = 1354.50M  (encrypted)\n",
        )

    monkeypatch.setattr(kbd.subprocess, "run", fake_run)
    sample = kbd._disk_sample()
    assert sample["free_bytes"] > 0
    assert sample["quantum_bytes"] == 2 * GIB
    assert sample["swap_fresh_files"] == 2
    assert sample["swap_used_bytes"] == int(2741.50 * 1024 ** 2)


@pytest.mark.real_disk_guard
def test_disk_sample_linux_swap_path(kanban_home, tmp_path, monkeypatch):
    monkeypatch.setattr(kbd, "_SWAP_DIR", tmp_path / "absent")

    def no_sysctl(cmd, **kwargs):
        raise FileNotFoundError("sysctl")

    monkeypatch.setattr(kbd.subprocess, "run", no_sysctl)
    monkeypatch.setattr(
        "gateway.lifecycle_ledger.sample_memory", lambda: {"swap_used_kib": 2048}
    )
    sample = kbd._disk_sample()
    assert sample["free_bytes"] > 0
    assert sample["quantum_bytes"] == 0
    assert sample["swap_fresh_files"] == 0
    assert sample["swap_used_bytes"] == 2048 * 1024


@pytest.mark.real_disk_guard
def test_disk_sample_omits_swap_when_unavailable(kanban_home, tmp_path, monkeypatch):
    monkeypatch.setattr(kbd, "_SWAP_DIR", tmp_path / "absent")
    monkeypatch.setattr(
        kbd.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout=""),
    )
    monkeypatch.setattr("gateway.lifecycle_ledger.sample_memory", lambda: {})
    assert "swap_used_bytes" not in kbd._disk_sample()


@pytest.mark.real_disk_guard
def test_disk_sample_empty_when_free_unknown(kanban_home, monkeypatch):
    def boom(path):
        raise OSError("no fs")

    monkeypatch.setattr(kbd.shutil, "disk_usage", boom)
    assert kbd._disk_sample() == {}
    assert kbd._disk_pressure_level() == "unknown"


@pytest.mark.real_disk_guard
def test_disk_sample_swap_probe_failure_fails_open(
    kanban_home, all_assignees_spawnable, tmp_path, monkeypatch, caplog,
):
    """An unreadable swap dir is a probe failure, not "no swapfiles": the
    quantum is unknown, so the guard fails open explicitly and says so."""
    swap_dir = tmp_path / "vm"
    swap_dir.mkdir()
    _fake_swapfiles(swap_dir)
    monkeypatch.setattr(kbd, "_SWAP_DIR", swap_dir)
    monkeypatch.setattr(
        kbd.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout=""),
    )
    monkeypatch.setattr("gateway.lifecycle_ledger.sample_memory", lambda: {})
    # Free space that would be critical with ANY quantum or a zero quantum.
    monkeypatch.setattr(kbd.shutil, "disk_usage", lambda p: _Usage(FLOOR // 2))
    real_stat = Path.stat

    def failing_stat(self, *a, **k):
        if self.name.startswith("swapfile"):
            raise PermissionError("operation not permitted")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", failing_stat)
    sample = kbd._disk_sample()
    assert sample["swap_probe_failed"] is True
    assert sample["free_bytes"] == FLOOR // 2
    assert "quantum_bytes" not in sample
    assert kbd._disk_pressure_level(sample) == "unknown"

    with caplog.at_level(logging.WARNING):
        res, spawns, _ = _dispatch_three(monkeypatch, sample)
    assert len(spawns) == 3
    assert res.disk_pressure is None
    probe_logs = [r for r in caplog.records if "swapfile probe" in r.getMessage()]
    assert len(probe_logs) == 1
    assert probe_logs[0].levelno == logging.WARNING
    assert "free 0.25 GiB" in probe_logs[0].getMessage()


@pytest.mark.real_disk_guard
def test_swap_listing_failure_is_probe_failure_not_absent(tmp_path, monkeypatch):
    def boom(path):
        raise PermissionError("listing failed")

    monkeypatch.setattr(kbd, "_SWAP_DIR", tmp_path)
    monkeypatch.setattr(kbd.os, "scandir", boom)
    assert kbd._swapfiles() is None
    monkeypatch.undo()
    monkeypatch.setattr(kbd, "_SWAP_DIR", tmp_path / "absent")
    assert kbd._swapfiles() == []


class _Usage:
    def __init__(self, free):
        self.free = free
        self.total = self.used = 0


@pytest.mark.real_disk_guard
def test_disk_sample_and_dispatch_gate_on_the_named_board(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Free space is read on the dispatched board's workspaces filesystem,
    not the current board's."""
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACES_ROOT", raising=False)
    monkeypatch.setattr(kbd, "_SWAP_DIR", kanban_home / "no-swap")
    monkeypatch.setattr(
        kbd.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout=""),
    )
    monkeypatch.setattr("gateway.lifecycle_ledger.sample_memory", lambda: {})
    kb.create_board("b")
    root_default = kb.workspaces_root("default")
    root_b = kb.workspaces_root("b")
    assert root_default != root_b
    root_default.mkdir(parents=True, exist_ok=True)
    root_b.mkdir(parents=True, exist_ok=True)
    free_by_root = {str(root_default): 100 * GIB, str(root_b): FLOOR // 2}
    monkeypatch.setattr(kbd.shutil, "disk_usage", lambda p: _Usage(free_by_root[str(p)]))

    assert kb.get_current_board() == "default"
    assert kbd._disk_sample().get("free_bytes") == 100 * GIB
    assert kbd._disk_sample(board="b")["free_bytes"] == FLOOR // 2

    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect(board="b") as conn:
        kb.create_task(conn, title="on b", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn, board="b")
    assert not spawns
    assert res.disk_pressure == "critical"


@pytest.mark.real_disk_guard
def test_unreadable_swap_dir_is_probe_failure(kanban_home, tmp_path, monkeypatch):
    """A real chmod-000 swap dir must fail the probe, not read as zero quantum."""
    swap_dir = tmp_path / "vm"
    swap_dir.mkdir()
    with open(swap_dir / "swapfile0", "wb") as fh:
        fh.truncate(GIB)
    monkeypatch.setattr(kbd, "_SWAP_DIR", swap_dir)
    monkeypatch.setattr(
        kbd.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout=""),
    )
    monkeypatch.setattr("gateway.lifecycle_ledger.sample_memory", lambda: {})
    os.chmod(swap_dir, 0)
    try:
        try:
            with os.scandir(swap_dir):
                readable = True
        except PermissionError:
            readable = False
        if readable:
            pytest.skip("chmod 000 does not block listing here (root?)")
        assert kbd._swapfiles() is None
        sample = kbd._disk_sample()
        assert sample["swap_probe_failed"] is True
        assert "quantum_bytes" not in sample
    finally:
        os.chmod(swap_dir, 0o755)


@pytest.mark.real_disk_guard
def test_absent_swap_dir_is_zero_quantum(kanban_home, tmp_path, monkeypatch):
    monkeypatch.setattr(kbd, "_SWAP_DIR", tmp_path / "absent")
    monkeypatch.setattr(
        kbd.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout=""),
    )
    monkeypatch.setattr("gateway.lifecycle_ledger.sample_memory", lambda: {})
    assert kbd._swapfiles() == []
    sample = kbd._disk_sample()
    assert sample["quantum_bytes"] == 0
    assert "swap_probe_failed" not in sample


@pytest.mark.real_disk_guard
def test_sysctl_overflowing_used_value_is_rejected(kanban_home, tmp_path, monkeypatch):
    out = "total = 1.00M  used = " + "9" * 400 + "M  free = 1.00M"
    monkeypatch.setattr(
        kbd.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=out),
    )
    monkeypatch.setattr(kbd, "_SWAP_DIR", tmp_path / "absent")
    monkeypatch.setattr("gateway.lifecycle_ledger.sample_memory", lambda: {})
    assert kbd._sysctl_swap_used_bytes() is None
    assert "swap_used_bytes" not in kbd._disk_sample()


@pytest.mark.parametrize("order", [("critical", "elevated"), ("elevated", "critical")])
def test_describe_suppression_keeps_most_severe_pressure(order):
    disk = [kbd.DispatchResult(disk_pressure=lvl) for lvl in order]
    assert kbd.describe_suppression(disk) == "disk_pressure=critical"
    mem = [kbd.DispatchResult(memory_pressure=lvl) for lvl in order]
    assert kbd.describe_suppression(mem) == "memory_pressure=critical"


# ---------------------------------------------------------------------------
# Bounded filesystem probes
# ---------------------------------------------------------------------------


def _alive_probe_threads():
    return [t for t in threading.enumerate() if t.name == "kanban-disk-probe" and t.is_alive()]


def _quiet_swap_used(monkeypatch):
    monkeypatch.setattr(
        kbd.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout=""),
    )
    monkeypatch.setattr("gateway.lifecycle_ledger.sample_memory", lambda: {})


def _assert_stall_degrades_then_recovers(monkeypatch, release, recovered_key="free_bytes"):
    monkeypatch.setattr(kbd, "DISK_GUARD_PROBE_TIMEOUT_SECONDS", 0.2)
    start = time.monotonic()
    sample = kbd._disk_sample()
    assert sample == {}
    assert time.monotonic() - start < 1.0
    assert kbd._disk_pressure_level(sample) == "unknown"

    start = time.monotonic()
    assert kbd._disk_sample() == {}
    assert time.monotonic() - start < 1.0
    assert len(_alive_probe_threads()) == 1

    release.set()
    for t in _alive_probe_threads():
        t.join(5)
    assert not _alive_probe_threads()
    assert recovered_key in kbd._disk_sample()


@pytest.mark.real_disk_guard
def test_stalled_free_probe_degrades_to_unknown(kanban_home, tmp_path, monkeypatch):
    monkeypatch.setattr(kbd, "_SWAP_DIR", tmp_path / "absent")
    _quiet_swap_used(monkeypatch)
    release = threading.Event()

    def blocking_usage(path):
        release.wait(10)
        return _Usage(100 * GIB)

    monkeypatch.setattr(kbd.shutil, "disk_usage", blocking_usage)
    try:
        _assert_stall_degrades_then_recovers(monkeypatch, release)
    finally:
        release.set()


@pytest.mark.real_disk_guard
def test_stalled_swap_listing_degrades_to_unknown(kanban_home, tmp_path, monkeypatch):
    monkeypatch.setattr(kbd, "_SWAP_DIR", tmp_path)
    _quiet_swap_used(monkeypatch)
    monkeypatch.setattr(kbd.shutil, "disk_usage", lambda p: _Usage(100 * GIB))
    release = threading.Event()
    real_scandir = os.scandir

    def blocking_scandir(path):
        release.wait(10)
        return real_scandir(path)

    monkeypatch.setattr(kbd.os, "scandir", blocking_scandir)
    try:
        _assert_stall_degrades_then_recovers(monkeypatch, release)
    finally:
        release.set()


@pytest.mark.real_disk_guard
def test_dispatch_with_stalled_probe_does_not_hang(
    kanban_home, all_assignees_spawnable, tmp_path, monkeypatch,
):
    """A stalled mount must not block the serial dispatcher tick: the guard
    reads "unknown" and imposes no restriction."""
    monkeypatch.setattr(kbd, "DISK_GUARD_PROBE_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(kbd, "_SWAP_DIR", tmp_path / "absent")
    _quiet_swap_used(monkeypatch)
    release = threading.Event()

    def blocking_usage(path):
        release.wait(10)
        return _Usage(FLOOR // 2)

    monkeypatch.setattr(kbd.shutil, "disk_usage", blocking_usage)
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    try:
        with kbc.connect() as conn:
            for t in ("a", "b", "c"):
                kb.create_task(conn, title=t, assignee="alice")
            start = time.monotonic()
            res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
            elapsed = time.monotonic() - start
        assert elapsed < 2.0
        assert len(spawns) == 3
        assert res.disk_pressure is None
    finally:
        release.set()
        for t in _alive_probe_threads():
            t.join(5)


def _blocking_meminfo_fallback(monkeypatch, release):
    """No sysctl; the Linux /proc fallback stalls until ``release``."""
    monkeypatch.setattr(kbd, "_sysctl_swap_used_bytes", lambda: None)

    def blocking_sample_memory():
        release.wait(10)
        return {"swap_used_kib": 2048}

    monkeypatch.setattr("gateway.lifecycle_ledger.sample_memory", blocking_sample_memory)


@pytest.mark.real_disk_guard
def test_stalled_linux_swap_fallback_degrades_to_unknown(kanban_home, tmp_path, monkeypatch):
    monkeypatch.setattr(kbd, "_SWAP_DIR", tmp_path / "absent")
    monkeypatch.setattr(kbd.shutil, "disk_usage", lambda p: _Usage(100 * GIB))
    release = threading.Event()
    _blocking_meminfo_fallback(monkeypatch, release)
    try:
        _assert_stall_degrades_then_recovers(monkeypatch, release, "swap_used_bytes")
    finally:
        release.set()


@pytest.mark.real_disk_guard
def test_stalled_sysctl_degrades_to_unknown(kanban_home, tmp_path, monkeypatch):
    monkeypatch.setattr(kbd, "_SWAP_DIR", tmp_path / "absent")
    monkeypatch.setattr(kbd.shutil, "disk_usage", lambda p: _Usage(100 * GIB))
    release = threading.Event()

    def blocking_sysctl():
        release.wait(10)
        return 3 * GIB

    monkeypatch.setattr(kbd, "_sysctl_swap_used_bytes", blocking_sysctl)
    try:
        _assert_stall_degrades_then_recovers(monkeypatch, release, "swap_used_bytes")
    finally:
        release.set()


@pytest.mark.real_disk_guard
def test_dispatch_with_stalled_swap_fallback_does_not_hang(
    kanban_home, all_assignees_spawnable, tmp_path, monkeypatch,
):
    monkeypatch.setattr(kbd, "DISK_GUARD_PROBE_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(kbd, "_SWAP_DIR", tmp_path / "absent")
    monkeypatch.setattr(kbd.shutil, "disk_usage", lambda p: _Usage(FLOOR // 2))
    release = threading.Event()
    _blocking_meminfo_fallback(monkeypatch, release)
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    try:
        with kbc.connect() as conn:
            for t in ("a", "b", "c"):
                kb.create_task(conn, title=t, assignee="alice")
            start = time.monotonic()
            res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
            elapsed = time.monotonic() - start
        assert elapsed < 2.0
        assert len(spawns) == 3
        assert res.disk_pressure is None
    finally:
        release.set()
        for t in _alive_probe_threads():
            t.join(5)
