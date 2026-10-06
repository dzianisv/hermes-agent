"""Disk-aware kanban dispatch guard (t_7938d84b).

Free disk at or below ``floor + one swapfile quantum`` means the next macOS
swap growth can fill the disk, so the dispatcher spawns nothing; within two
quanta (plus one per freshly-created swapfile) it spawns at most one. With no
quantum (Linux) the thresholds are ``floor`` / ``2*floor``. Unknown free space
imposes no restriction (fail-open).
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc

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


def _sample(free, q=Q, fresh=0):
    return {"free_bytes": free, "quantum_bytes": q, "swap_fresh_files": fresh}


def _dispatch_three(monkeypatch, sample):
    monkeypatch.setattr(kbd, "_disk_sample", lambda: sample)
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
    for bad in ("0", "-1", "abc"):
        monkeypatch.setenv("DISK_GUARD_FLOOR_GI", bad)
        assert kbd._disk_floor_bytes() == FLOOR


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
