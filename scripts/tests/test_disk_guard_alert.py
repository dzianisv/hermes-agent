"""Tests for disk_guard_alert.py — the alert path of the disk guard.

CLASS under test: the guard was RED (launchctl exit 1, 501 runs) for days and
nobody heard about it, because the only output channel was a launchd log file.
An alert that goes nowhere is not an alert.

Run: python3 -m pytest ~/.hermes/scripts/tests/test_disk_guard_alert.py -q
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

# Bind to THIS checkout; DISK_GUARD_ALERT_SCRIPT may point at another copy.
SCRIPT = Path(os.environ.get("DISK_GUARD_ALERT_SCRIPT")
              or Path(__file__).resolve().parents[1] / "disk_guard_alert.py")
spec = importlib.util.spec_from_file_location("disk_guard_alert", SCRIPT)
dga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dga)


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Every test gets its own dedupe state; otherwise a real state file on
    this host silently suppresses the alert the test is asserting."""
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "state.json"))
    # The RED path measures swap; keep the 2s `top` probe out of every test.
    monkeypatch.setenv("DISK_GUARD_SKIP_TOP", "1")


# ------------------------------------------------------------- free space ---

def test_free_gi_parses_df(monkeypatch):
    df = (
        "Filesystem 1M-blocks Used Avail Capacity iused ifree %iused Mounted on\n"
        "/dev/disk3s1 233472 195000 7600 97% 1 1 1% /System/Volumes/Data\n"
    )
    monkeypatch.setattr(dga, "_run", lambda *a, **k: df)
    assert dga.free_gi() == pytest.approx(7600 / 1024)


def test_free_gi_returns_none_when_df_unparseable(monkeypatch):
    monkeypatch.setattr(dga, "_run", lambda *a, **k: "")
    assert dga.free_gi() is None


# ------------------------------------------------------------- threshold ---

def test_below_floor_is_alertable():
    assert dga.should_alert(free=7, floor=10) is True


def test_at_floor_is_alertable():
    """Owner decision (Den, 2026-09-21): page at the floor OR LESS, not strictly
    below it. With the floor pinned at 1Gi, `free < floor` would mean the only
    alertable state is 0Gi — i.e. the host is already dead. The comparison is
    `<=` so 1Gi free still pages."""
    assert dga.should_alert(free=10, floor=10) is True
    assert dga.should_alert(free=1, floor=1) is True
    assert dga.should_alert(free=0, floor=1) is True


def test_above_floor_is_silent():
    assert dga.should_alert(free=2, floor=1) is False
    assert dga.should_alert(free=7, floor=1) is False


def test_unknown_free_alerts_rather_than_staying_silent():
    """df failing is itself an anomaly; silence is the failure mode we fix."""
    assert dga.should_alert(free=None, floor=10) is True
    assert dga.should_alert(free=None, floor=1) is True


# ------------------------------------------------------------ pinned floor --
# CLASS: the floor is a DECISION, not a tunable. It is pinned in three places
# that must agree, and a disagreement is how an operator ends up believing the
# guard is quiet at 7Gi while one copy still pages at 10Gi. Each assertion below
# DERIVES the value from the file rather than trusting a comment.

EXPECTED_FLOOR_GI = 0.5

# The floor is a DECIMAL GiB figure since 2026-09-22 (1 -> 0.5). Every pattern
# below therefore matches `\d+(?:\.\d+)?`: an int-only pattern does not read a
# 0.5 pin as "wrong", it reads it as "absent", and the assertion that fires is
# the "no longer declares" one -- which the next person fixes by loosening the
# ASSERTION instead of the number, leaving the copy unguarded.
_NUM = r'(\d+(?:\.\d+)?)'


def test_default_floor_is_the_owner_decision():
    assert dga.DEFAULT_FLOOR_GI == EXPECTED_FLOOR_GI


def test_every_deployed_floor_agrees_with_the_decision():
    """Derived, not a hand-maintained list: each source of truth is parsed."""
    import re as _re
    root = Path(os.path.expanduser("~/.hermes"))
    found = {}

    found["disk_guard_alert.py"] = float(dga.DEFAULT_FLOOR_GI)

    sh = (root / "scripts" / "disk-guard.sh").read_text(encoding="utf-8-sig")
    m = _re.search(r'FLOOR_GI="\$\{DISK_GUARD_FLOOR_GI:-' + _NUM + r'\}"', sh)
    assert m, "disk-guard.sh no longer declares FLOOR_GI the way this guard parses"
    found["disk-guard.sh"] = float(m.group(1))

    cron = (root / "profiles" / "software-engineer" / "scripts"
            / "disk-guard-cron.sh").read_text(encoding="utf-8-sig")
    m = _re.search(r'^FLOOR_GI_PINNED=' + _NUM, cron, _re.M)
    assert m, "disk-guard-cron.sh no longer declares FLOOR_GI_PINNED"
    found["disk-guard-cron.sh"] = float(m.group(1))

    # Every deployed copy of the heartbeat, globbed — a new copy cannot be
    # silently unguarded.
    hbs = sorted(root.glob("scripts/agentpod_em_heartbeat.py")) + \
        sorted(root.glob("profiles/*/scripts/agentpod_em_heartbeat.py"))
    assert hbs, "no heartbeat copy found — the glob is wrong, not the floor"
    for p in hbs:
        m = _re.search(r'^DISK_FLOOR_GI\s*=\s*' + _NUM, p.read_text(encoding="utf-8-sig"), _re.M)
        assert m, f"{p} no longer declares DISK_FLOOR_GI"
        found[str(p)] = float(m.group(1))

    bad = {k: v for k, v in found.items() if v != EXPECTED_FLOOR_GI}
    assert not bad, f"floors disagree with the {EXPECTED_FLOOR_GI}Gi decision: {bad}"


def test_reclaim_target_is_higher_than_the_paging_floor_and_never_pages():
    """CLASS (2026-09-24 incident, card t_b8d0aaeb): the host went from above
    the floor to ENOSPC inside one 900s tick, so a paging floor was being asked
    to do capacity work. The fix separates the two numbers:

      * RECLAIM_TARGET_GI -- how much free space we try to HOLD, enforced by
        evicting more sanctioned scratch, SILENTLY.
      * FLOOR_GI          -- the only thing that pages. Owner-pinned at 0.5Gi.

    The target must be strictly above the floor (otherwise it enforces nothing)
    and must not appear on any paging path (otherwise it silently re-raises the
    alert rate the owner cut on 2026-09-22). Both are DERIVED from the file.
    """
    import re as _re
    sh = Path(os.path.expanduser("~/.hermes/scripts/disk-guard.sh")).read_text(encoding="utf-8-sig")

    m = _re.search(r'RECLAIM_TARGET_GI="\$\{DISK_GUARD_RECLAIM_TARGET_GI:-'
                   + _NUM + r'\}"', sh)
    assert m, "disk-guard.sh no longer declares RECLAIM_TARGET_GI"
    target = float(m.group(1))

    m = _re.search(r'FLOOR_GI="\$\{DISK_GUARD_FLOOR_GI:-' + _NUM + r'\}"', sh)
    assert m, "disk-guard.sh no longer declares FLOOR_GI"
    floor = float(m.group(1))

    assert floor == EXPECTED_FLOOR_GI, (
        "the reclaim target must not be implemented by moving the paging floor"
    )
    assert target > floor, (
        f"reclaim target {target}Gi must sit above the paging floor {floor}Gi"
    )

    # The exit-1 (paging) decision is `below_floor`, never `below_target`.
    tail = sh.split("after=$(free_gi)")[-1]
    assert "below_floor" in tail
    assert "below_target" not in tail, (
        "the reclaim target leaked into the paging decision; it must stay silent"
    )


def test_clamp_cannot_lower_the_floor_below_the_decision(monkeypatch):
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "3")
    boom = lambda t: (_ for _ in ()).throw(AssertionError("paged at 3Gi"))
    monkeypatch.setattr(dga, "deliver_kanban_cto",
                        lambda t, key=None: boom(t))
    monkeypatch.setattr(dga, "deliver_telegram", boom)
    assert dga.main(["--floor", "0"]) == 0, "--floor 0 must clamp up to the pin"


# ----------------------------------------------------------- both channels --

def test_alert_delivers_to_the_cto_and_to_den(monkeypatch):
    """Owner decision (card t_b8d0aaeb, 2026-09-24): host disk is CTO scope,
    not EM scope. The page must reach the CTO route and Den, and must NOT wake
    the EM (ceo) session -- an EM paged for a host problem can only re-delegate
    it, which is why this incident reached an operator late."""
    calls = []
    monkeypatch.setattr(dga, "deliver_kanban_cto",
                        lambda text, key=None: calls.append(("cto", text)) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda text: calls.append(("tg", text)) or True)
    monkeypatch.setattr(dga, "deliver_wake",
                        lambda text: (_ for _ in ()).throw(AssertionError("EM was paged")))
    ok = dga.alert("DISK GUARD RED: 7Gi free")
    assert ok is True
    assert [c[0] for c in calls] == ["cto", "tg"]
    assert all("DISK GUARD RED" in c[1] for c in calls)


def test_cto_card_is_keyed_to_the_incident(monkeypatch):
    """Board dedupe must match page dedupe: one incident, one card, however
    many ticks observe it. The key is passed through to the idempotency key."""
    seen = {}
    monkeypatch.setattr(dga, "deliver_kanban_cto",
                        lambda text, key=None: seen.update(key=key) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda text: True)
    dga.alert("DISK GUARD RED", key="abc123")
    assert seen["key"] == "abc123"


def test_cto_card_uses_idempotency_key_and_cto_assignee(monkeypatch, tmp_path):
    """The card is raised via the kanban CLI with the incident key as the
    idempotency key, so a repeated observation returns the same card instead of
    littering the board."""
    fake = tmp_path / "hermes"
    fake.write_text("#!/bin/sh\necho '{\"id\": \"t_dead1234\"}'\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(dga, "KANBAN_BIN", str(fake))
    captured = {}
    real_run = dga.subprocess.run

    def spy(argv, **kw):
        captured["argv"] = argv
        return real_run(argv, **kw)

    monkeypatch.setattr(dga.subprocess, "run", spy)
    assert dga.deliver_kanban_cto("DISK GUARD RED\nbody", key="k9") is True
    argv = captured["argv"]
    assert "--assignee" in argv and argv[argv.index("--assignee") + 1] == "cto"
    assert argv[argv.index("--idempotency-key") + 1] == "disk-guard-k9"


def test_cto_card_failure_does_not_swallow_the_telegram_page(monkeypatch):
    """The two channels must have INDEPENDENT failure modes; a broken board
    (the very thing ENOSPC corrupts) must not silence the page."""
    calls = []

    def boom(text, key=None):
        raise RuntimeError("kanban.db disk I/O error")

    monkeypatch.setattr(dga, "deliver_kanban_cto", boom)
    monkeypatch.setattr(dga, "deliver_telegram", lambda text: calls.append("tg") or True)
    assert dga.alert("DISK GUARD RED: 0Gi free") is True
    assert calls == ["tg"]


def test_telegram_is_attempted_even_when_board_channel_fails(monkeypatch):
    """One dead channel must not swallow the page — that is this bug's shape."""
    calls = []

    def boom(text, key=None):
        calls.append(("cto", text))
        raise RuntimeError("board down")

    monkeypatch.setattr(dga, "deliver_kanban_cto", boom)
    monkeypatch.setattr(dga, "deliver_telegram", lambda text: calls.append(("tg", text)) or True)
    ok = dga.alert("DISK GUARD RED: 0Gi free")
    assert [c[0] for c in calls] == ["cto", "tg"]
    assert ok is True  # at least one channel landed


def test_alert_returns_false_when_every_channel_fails(monkeypatch):
    def boom(text, key=None):
        raise RuntimeError("down")

    monkeypatch.setattr(dga, "deliver_kanban_cto", boom)
    monkeypatch.setattr(dga, "deliver_telegram", lambda text: boom(text))
    assert dga.alert("x") is False


# ------------------------------------------------------------------ wake ---

def test_wake_signs_v2_over_timestamp_dot_body(monkeypatch, tmp_path):
    import hashlib
    import hmac

    secret_file = tmp_path / "secret"
    secret_file.write_text("s3cr3t\n", encoding="utf-8")
    sent = {}

    class FakeResp:
        def read(self):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=10):
        sent["url"] = req.full_url
        sent["headers"] = {k.lower(): v for k, v in req.headers.items()}
        sent["body"] = req.data
        return FakeResp()

    monkeypatch.setattr(dga, "wake_config", lambda: {
        "url": "http://127.0.0.1:8645/webhooks/kanban-wake",
        "secret_file": str(secret_file),
    })
    monkeypatch.setattr(dga.urllib.request, "urlopen", fake_urlopen)

    assert dga.deliver_wake("hello disk") is True
    ts = sent["headers"]["x-webhook-timestamp"]
    expected = hmac.new(b"s3cr3t", f"{ts}.".encode() + sent["body"],
                        hashlib.sha256).hexdigest()
    assert sent["headers"]["x-webhook-signature-v2"] == expected
    assert json.loads(sent["body"])["event"] == "disk_guard"
    assert "hello disk" in json.loads(sent["body"])["text"]


# ----------------------------------------------------------- forced fixture -

def test_forced_low_disk_fixture_drives_the_whole_path(monkeypatch):
    """DISK_GUARD_FAKE_FREE_GI forces the low-disk branch end to end, so the
    alert path is provable without actually filling the volume."""
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "1")
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    rc = dga.main(["--floor", "10"])
    assert rc == 1
    assert delivered, "forced low-disk fixture must reach the alert channels"
    assert "1.0Gi" in delivered[0]


def test_healthy_disk_is_silent(monkeypatch):
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "99")
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    rc = dga.main(["--floor", "10"])
    assert rc == 0
    assert delivered == []


# ----------------------------------------------------------------- dedupe --
# Dedupe is exercised with a MEASURED figure (free_gi patched), not the
# DISK_GUARD_FAKE_FREE_GI fixture: a fixture run never writes dedupe state, so
# it cannot be the first tick of a repeat.

def _measured(monkeypatch, free):
    monkeypatch.delenv("DISK_GUARD_FAKE_FREE_GI", raising=False)
    monkeypatch.delenv("DISK_GUARD_FAKE_SWAP_GI", raising=False)
    monkeypatch.setattr(dga, "free_gi", lambda: float(free))
    monkeypatch.setattr(dga, "swap_info", lambda: {
        "used_gi": None, "files": None, "quantum_gi": None,
        "grew_24h": None, "top": None})

def test_repeat_alert_within_window_is_suppressed(monkeypatch, tmp_path):
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "state.json"))
    _measured(monkeypatch, "1")
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    assert dga.main(["--floor", "10"]) == 1
    first = len(delivered)
    assert dga.main(["--floor", "10"]) == 1
    assert len(delivered) == first, "identical alert inside the window must not re-page"


def test_worsening_alert_is_not_suppressed(monkeypatch, tmp_path):
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "state.json"))
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    _measured(monkeypatch, "5")
    dga.main(["--floor", "10"])
    n = len(delivered)
    _measured(monkeypatch, "1")
    dga.main(["--floor", "10"])
    assert len(delivered) > n, "a worse figure is new information and must page"


def test_small_wobble_inside_window_does_not_repage(monkeypatch, tmp_path):
    """Two pages in five minutes for the same incident is the anti-pattern.
    Under hash dedupe the incident key includes the free figure, so a drifting
    number with the SAME remediation list must still not re-page — the operator
    would read an identical actionable list."""
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "s.json"))
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    body = "Top reclaimable:\n  2337MB  /a"
    _measured(monkeypatch, "7")
    dga.main(["--floor", "10", "--detail", body])
    n = len(delivered)
    assert n > 0
    _measured(monkeypatch, "7")  # same state, next tick
    dga.main(["--floor", "10", "--detail", body])
    assert len(delivered) == n


def test_material_drop_repages(monkeypatch, tmp_path):
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "s.json"))
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    _measured(monkeypatch, "7")
    dga.main(["--floor", "10"])
    n = len(delivered)
    _measured(monkeypatch, "2")
    dga.main(["--floor", "10"])
    assert len(delivered) > n


# ----------------------------------------------- hash-of-content dedupe ----
# CLASS: five pages were sent for one incident on 2026-09-21; the fourth and
# fifth were BYTE-IDENTICAL (14Gi/25Gi, same six paths). Dedupe keyed on the
# free-GiB number alone cannot see that the remediation list is unchanged, and
# any reset of the state file re-pages. The key must be a hash of what the
# operator would actually read: (free, floor, top paths).

def _fire(monkeypatch, delivered, free, detail, floor="25"):
    _measured(monkeypatch, free)
    return dga.main(["--floor", floor, "--detail", detail])


def test_identical_red_does_not_repage(monkeypatch, tmp_path):
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "s.json"))
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    body = "Top reclaimable:\n  2337MB  /a\n  1746MB  /b"
    _fire(monkeypatch, delivered, "14", body)
    n = len(delivered)
    assert n > 0, "first RED must page"
    _fire(monkeypatch, delivered, "14", body)   # byte-identical second tick
    assert len(delivered) == n, "identical RED must not re-page"


def test_changed_top_paths_repages(monkeypatch, tmp_path):
    """The number is the same but the remediation list changed — that is new
    information for the operator and must page."""
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "s.json"))
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    _fire(monkeypatch, delivered, "14", "Top reclaimable:\n  2337MB  /a")
    n = len(delivered)
    _fire(monkeypatch, delivered, "14", "Top reclaimable:\n  9000MB  /zzz-new")
    assert len(delivered) > n


def test_identical_red_repages_after_ttl(monkeypatch, tmp_path):
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "s.json"))
    delivered = []
    monkeypatch.setattr(dga, "deliver_kanban_cto", lambda t, key=None: delivered.append(t) or True)
    monkeypatch.setattr(dga, "deliver_telegram", lambda t: delivered.append(t) or True)
    body = "Top reclaimable:\n  2337MB  /a"
    _fire(monkeypatch, delivered, "14", body)
    n = len(delivered)
    st = json.loads((tmp_path / "s.json").read_text(encoding="utf-8-sig"))
    st["last_at"] = int(time.time()) - (dga.REPEAT_WINDOW_SECONDS + 60)
    (tmp_path / "s.json").write_text(json.dumps(st), encoding="utf-8")
    _fire(monkeypatch, delivered, "14", body)
    assert len(delivered) > n, "TTL expiry must re-page even when unchanged"


def test_dry_run_never_delivers_but_reports(monkeypatch, tmp_path, capsys):
    """Iterating on the guard must not page the CEO. --dry-run prints the page
    and touches no channel and no state."""
    state = tmp_path / "s.json"
    monkeypatch.setenv("DISK_GUARD_STATE", str(state))
    boom = lambda t: (_ for _ in ()).throw(AssertionError("delivered in dry-run"))
    monkeypatch.setattr(dga, "deliver_kanban_cto",
                        lambda t, key=None: boom(t))
    monkeypatch.setattr(dga, "deliver_telegram", boom)
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "3")
    rc = dga.main(["--floor", "10", "--detail", "x", "--dry-run"])
    assert rc == 1
    assert "DISK GUARD RED" in capsys.readouterr().out
    assert not state.exists(), "dry-run must not write dedupe state"


# ------------------------------------------------------------- swap term ---
# CLASS (card t_515b8493): ~30GiB of macOS swap shared the APFS container with
# the data volume, so free space decayed while the page blamed scratch. The page
# must name swap when swap explains at least half of the shortfall to target.

TOP_OUT = ("Processes: 1\n\nCMPRS COMMAND\n20G   fseventsd\n7680M  com.apple.Virtua\n")


def _fake_run(swap_dir_rows):
    def run(argv, timeout=30):
        if argv[0] == "top":
            return TOP_OUT
        if argv[0] == "/bin/sh" and "stat" in argv[2]:
            return swap_dir_rows
        return ""
    return run


def _swap_page(monkeypatch, capsys, tmp_path, swap_gi, free="0.3"):
    monkeypatch.delenv("DISK_GUARD_SKIP_TOP", raising=False)
    monkeypatch.delenv("DISK_GUARD_RECLAIM_TARGET_GI", raising=False)
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", free)
    monkeypatch.setenv("DISK_GUARD_FAKE_SWAP_GI", swap_gi)
    d = tmp_path / "vm"
    monkeypatch.setenv("DISK_GUARD_SWAP_DIR", str(d))
    now = int(time.time())
    rows = (f"1073741824 {now - 100} {d}/swapfile0\n"
            f"1073741824 {now - 200} {d}/swapfile1\n"
            f"1073741824 {now - 200000} {d}/swapfile2\n"
            # a sibling the guard must ignore (not a swapfile proper)
            f"9999999999 {now} {d}/swapfile.lock\n")
    monkeypatch.setattr(dga, "_run", _fake_run(rows))
    assert dga.main(["--dry-run"]) == 1
    return capsys.readouterr().out


def test_swap_dominated_shortfall_names_swap_as_cause(monkeypatch, capsys, tmp_path):
    out = _swap_page(monkeypatch, capsys, tmp_path, "20")
    first = out.strip().splitlines()[0]
    assert "cause: swap pressure, not scratch" in first
    assert first[:120].startswith("DISK GUARD RED")
    swap = [ln for ln in out.splitlines() if ln.startswith("Swap:")]
    assert swap == ["Swap: used 20.0 Gi in 3 swapfiles (grew by 2 in 24h); "
                    "top compressed: fseventsd:20.0G,com.apple.Virtua:7.5G"]


def test_no_swap_means_no_swap_line_and_no_cause(monkeypatch, capsys, tmp_path):
    out = _swap_page(monkeypatch, capsys, tmp_path, "0")
    assert "Swap:" not in out
    assert "cause: swap pressure" not in out


def test_minor_swap_is_reported_but_not_blamed(monkeypatch, capsys, tmp_path):
    out = _swap_page(monkeypatch, capsys, tmp_path, "5")
    assert any(ln.startswith("Swap: used 5.0 Gi") for ln in out.splitlines())
    assert "cause:" not in out.strip().splitlines()[0]


def test_fake_swap_alone_is_a_fixture_that_cannot_page(monkeypatch):
    monkeypatch.delenv("DISK_GUARD_FAKE_FREE_GI", raising=False)
    monkeypatch.setenv("DISK_GUARD_FAKE_SWAP_GI", "20")
    assert dga.fixture_active() is True
    with pytest.raises(RuntimeError, match="DISK_GUARD_FAKE_SWAP_GI"):
        dga.deliver_telegram("x")
    with pytest.raises(RuntimeError, match="DISK_GUARD_FAKE_SWAP_GI"):
        dga.deliver_kanban_cto("x", key="k")


def test_swap_info_degrades_to_none_when_every_probe_fails(monkeypatch, tmp_path):
    monkeypatch.delenv("DISK_GUARD_FAKE_SWAP_GI", raising=False)
    monkeypatch.delenv("DISK_GUARD_SKIP_TOP", raising=False)
    monkeypatch.setenv("DISK_GUARD_SWAP_DIR", str(tmp_path))
    (tmp_path / "swapfile0").write_text("", encoding="utf-8")
    monkeypatch.setattr(dga, "_run", lambda *a, **k: "")
    assert dga.swap_info() == {"used_gi": None, "files": None, "quantum_gi": None,
                               "grew_24h": None, "top": None}


def test_swap_used_parses_sysctl_swapusage(monkeypatch):
    monkeypatch.delenv("DISK_GUARD_FAKE_SWAP_GI", raising=False)
    real = "total = 4096.00M  used = 2741.50M  free = 1354.50M  (encrypted)\n"
    monkeypatch.setattr(dga, "_run",
                        lambda argv, timeout=30: real if argv[0] == "sysctl" else "")
    assert dga.swap_info()["used_gi"] == pytest.approx(2741.5 / 1024, abs=0.01)


def test_swap_files_reports_the_average_of_actual_sizes(monkeypatch):
    rows = ("1073741824 100 /vm/swapfile0\n"
            "3221225472 200 /vm/swapfile1\n")
    monkeypatch.setattr(dga, "_run", lambda argv, timeout=30: rows)
    assert dga._swap_files(300) == (2, pytest.approx(2.0), 2)


def test_swap_files_timeout_degrades_and_page_is_still_composed(
        monkeypatch, capsys, tmp_path):
    """_run returns None on a timed-out probe: the swap line degrades, nothing
    raises, and the RED page is still rendered."""
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "0.3")
    monkeypatch.setenv("DISK_GUARD_FAKE_SWAP_GI", "20")
    seen = []

    def run(argv, timeout=30):
        seen.append((argv, timeout))
        return None
    monkeypatch.setattr(dga, "_run", run)
    assert dga._swap_files(time.time()) == (None, None, None)
    assert dga.main(["--dry-run"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("DISK GUARD RED")
    assert "Swap: used 20.0 Gi in unknown swapfiles (grew by unknown in 24h)" in out
    assert any(a[0] == "/bin/sh" and t == 5 for a, t in seen), seen


def test_swap_files_missing_dir_degrades(monkeypatch, tmp_path):
    """Real probe, nonexistent dir: the unmatched glob fails stat -> None."""
    monkeypatch.setenv("DISK_GUARD_SWAP_DIR", str(tmp_path / "nope"))
    assert dga._swap_files(time.time()) == (None, None, None)


def test_swap_files_never_enumerates_outside_the_bounded_seam(
        monkeypatch, tmp_path):
    """os.listdir has no timeout; enumeration must go through _run only."""
    def boom(*a, **k):
        raise AssertionError("os.listdir called outside the _run seam")
    monkeypatch.setattr(dga.os, "listdir", boom)
    d = tmp_path / "vm"
    d.mkdir()
    (d / "swapfile0").write_bytes(b"x" * 2048)
    (d / "swapfile0.lock").write_text("", encoding="utf-8")
    monkeypatch.setenv("DISK_GUARD_SWAP_DIR", str(d))
    n, avg, grew = dga._swap_files(time.time())
    assert (n, grew) == (1, 1)
    assert avg == pytest.approx(2048 / 1073741824)
    monkeypatch.delenv("DISK_GUARD_FAKE_SWAP_GI", raising=False)
    info = dga.swap_info()
    assert info["files"] == 1


# ------------------------------------------- undelivered pages never dedupe -

def _channels(monkeypatch, delivered, ok=True):
    monkeypatch.setattr(dga, "deliver_kanban_cto",
                        lambda t, key=None: delivered.append(t) or ok)
    monkeypatch.setattr(dga, "deliver_telegram",
                        lambda t: delivered.append(t) or ok)


def test_refused_fixture_red_does_not_suppress_the_identical_real_red(
        monkeypatch, tmp_path):
    state = tmp_path / "s.json"
    monkeypatch.setenv("DISK_GUARD_STATE", str(state))
    monkeypatch.setattr(dga, "swap_info", lambda: {
        "used_gi": None, "files": None, "quantum_gi": None,
        "grew_24h": None, "top": None})
    body = "Top reclaimable:\n  2337MB  /a"
    # Tick 1: fixture figure, REAL channels -> both refuse (FIXTURE fence).
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "0.3")
    monkeypatch.setattr(dga.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("fixture reached the board")))
    monkeypatch.setattr(dga.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("fixture reached the network")))
    assert dga.main(["--detail", body]) == 1
    st = json.loads(state.read_text(encoding="utf-8-sig")) if state.exists() else {}
    assert "last_at" not in st and "last_key" not in st, st
    # Tick 2: identical incident, now MEASURED -> a real delivery is attempted.
    delivered = []
    monkeypatch.delenv("DISK_GUARD_FAKE_FREE_GI")
    monkeypatch.setattr(dga, "free_gi", lambda: 0.3)
    _channels(monkeypatch, delivered)
    assert dga.main(["--detail", body]) == 1
    assert len(delivered) == 2, "the real RED was suppressed by a refused fixture"


def test_undelivered_real_page_is_reattempted(monkeypatch, tmp_path):
    state = tmp_path / "s.json"
    monkeypatch.setenv("DISK_GUARD_STATE", str(state))
    _measured(monkeypatch, 0.3)
    body = "Top reclaimable:\n  2337MB  /a"
    attempts = []
    _channels(monkeypatch, attempts, ok=False)
    dga.main(["--detail", body])
    assert json.loads(state.read_text(encoding="utf-8-sig"))["landed"] is False
    n = len(attempts)
    assert n == 2
    dga.main(["--detail", body])
    assert len(attempts) == 2 * n, "a page that never landed must not dedupe"


def test_landed_page_still_suppresses_the_identical_page(monkeypatch, tmp_path):
    state = tmp_path / "s.json"
    monkeypatch.setenv("DISK_GUARD_STATE", str(state))
    _measured(monkeypatch, 0.3)
    body = "Top reclaimable:\n  2337MB  /a"
    delivered = []
    _channels(monkeypatch, delivered)
    dga.main(["--detail", body])
    first_at = json.loads(state.read_text(encoding="utf-8-sig"))["last_at"]
    n = len(delivered)
    assert n == 2
    dga.main(["--detail", body])
    assert len(delivered) == n
    assert json.loads(state.read_text(encoding="utf-8-sig"))["last_at"] == first_at
