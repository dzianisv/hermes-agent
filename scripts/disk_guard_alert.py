#!/usr/bin/env python3
"""disk_guard_alert.py — the alert half of the host disk guard.

CLASS this fixes: `com.agentlabs.hermes-disk-guard` sat RED (launchd exit 1)
for 501 runs and nobody heard it, because its only output was a log file.
The volume then hit 100% and the agentpod-ceo gateway took ENOSPC on every
write while still reporting "running". An alert with no delivery channel is
indistinguishable from no alert.

So this module:
  * measures free space on the data volume (derived from df, not a literal),
  * pages BOTH channels independently — the EM (ceo) session via the
    kanban-wake webhook AND Den directly via Telegram sendMessage — so one
    dead channel cannot swallow the page,
  * dedupes identical alerts inside a window, but always re-pages when the
    figure gets worse,
  * supports DISK_GUARD_FAKE_FREE_GI, a forced low-disk fixture that drives
    the whole path end to end without filling the volume.

Exit codes: 0 = healthy, 1 = below floor (alert attempted).
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import uuid

HERMES = os.path.expanduser("~/.local/bin/hermes")
DATA_VOLUME = "/System/Volumes/Data"
# THE FLOOR IS A DECISION, NOT A TUNABLE. Pinned at 0.5Gi by the owner (Den,
# 2026-09-22): page ONLY at 0.5Gi free or less -- i.e. only where writes are
# actually about to fail. The previous 1Gi pin produced 67 RED ticks in ~13h on
# 2026-09-22, all at 0.8-0.9Gi, none of them next to a failed write; an alert
# nobody can act on trains the operator to ignore the channel that has to
# survive a real ENOSPC. The derivation for the old 10Gi capacity figure lives
# in disk-guard-cron.sh; the owner accepted the reduced headroom explicitly.
# Change it there and here together, or not at all
# (test_every_deployed_floor_agrees_with_the_decision enforces it).
DEFAULT_FLOOR_GI = 0.5
DEFAULT_STATE = "~/.hermes/scripts/state/disk-guard-alert.json"
REPEAT_WINDOW_SECONDS = 6 * 3600
# A host hovering near the floor wobbles by a few hundred MB every tick. Only a
# materially worse figure is new information; anything else inside the window is
# the same incident and must not re-page (two pages in five minutes for one
# state is the anti-pattern this window exists to prevent).
MATERIAL_DROP_GI = 2
DEN_TELEGRAM_ID = "1916982742"


def _run(argv, timeout=30):
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout
        ).stdout
    except Exception:
        return ""


# ----------------------------------------------------------- measurement ---

def free_gi():
    """Free GiB on the data volume, or None when df cannot be parsed."""
    forced = os.environ.get("DISK_GUARD_FAKE_FREE_GI")
    if forced not in (None, ""):
        try:
            return float(forced)
        except ValueError:
            return None
    out = _run(["df", "-m", DATA_VOLUME])
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    parts = lines[1].split()
    if len(parts) < 4:
        return None
    try:
        # FLOAT GiB. `// 1024` floored 1708 MiB to 1Gi and paged a host that
        # actually had 1.7Gi free (2026-09-22).
        return int(parts[3]) / 1024.0
    except ValueError:
        return None


# ------------------------------------------------------------------ swap ---
# MEASURED (card t_515b8493): ~30GiB of /System/Volumes/VM/swapfile* shared
# the APFS container with the data volume while every scratch class reported
# nothing to reclaim. Swap is reported so the page names the real cause; it is
# never reclaimed and never changes the paging decision.
SWAP_DIR = "/System/Volumes/VM"
_SWAP_USED = re.compile(r"used\s*=\s*([0-9.]+)([KMG])")
_CMPRS = re.compile(r"^([0-9.]+)([BKMG]?)[+-]?$")
_UNIT_GI = {"G": 1.0, "M": 1 / 1024, "K": 1 / 1048576, "B": 1 / 1073741824,
            "": 1 / 1073741824}


def _swap_used_gi():
    forced = os.environ.get("DISK_GUARD_FAKE_SWAP_GI")
    if forced not in (None, ""):
        try:
            return float(forced)
        except ValueError:
            return None
    m = _SWAP_USED.search(_run(["sysctl", "-n", "vm.swapusage"], timeout=5) or "")
    return float(m.group(1)) * _UNIT_GI[m.group(2)] if m else None


def _swap_files(now):
    """(count, average GiB, files written in the last 24h) or (None,)*3.

    Enumeration lives INSIDE the _run seam with a single 5s bound: an
    os.listdir on a wedged volume blocks with no timeout, and a per-file stat
    loop multiplies the bound by the file count. The shell expands the glob;
    an unmatched glob is passed through literally and fails stat (rc!=0).
    """
    d = os.environ.get("DISK_GUARD_SWAP_DIR") or SWAP_DIR
    out = _run(["/bin/sh", "-c", 'stat -f "%z %m %N" "$1"/swapfile*', "_", d],
               timeout=5)
    if not out:
        return None, None, None
    rows = []
    for line in out.splitlines():
        parts = line.split(" ", 2)
        if len(parts) != 3 or not (parts[0].isdigit() and parts[1].isdigit()):
            continue
        name = os.path.basename(parts[2])
        if not name.startswith("swapfile") or "." in name:
            continue
        rows.append((int(parts[0]), int(parts[1])))
    if not rows:
        return None, None, None
    avg = sum(sz for sz, _ in rows) / len(rows) / 1073741824
    grew = sum(1 for _, mt in rows if now - mt <= 86400)
    return len(rows), avg, grew


def _compressor_top():
    if os.environ.get("DISK_GUARD_SKIP_TOP") == "1":
        return None
    out = _run(["top", "-l1", "-stats", "cmprs,command", "-o", "cmprs",
                "-n", "2"], timeout=10) or ""
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^CMPRS\s+COMMAND", line.strip()):
            items = []
            for row in lines[i + 1:]:
                parts = row.split(None, 1)
                if len(parts) < 2:
                    continue
                m = _CMPRS.match(parts[0])
                if not m:
                    return None
                gi = float(m.group(1)) * _UNIT_GI[m.group(2)]
                items.append(f"{'_'.join(parts[1].split())}:{gi:.1f}G")
                if len(items) == 2:
                    break
            return ",".join(items) or None
    return None


def swap_info():
    """Measured swap term. Every field degrades to None; never raises."""
    info = {"used_gi": None, "files": None, "quantum_gi": None,
            "grew_24h": None, "top": None}
    for field, fn in (("used_gi", _swap_used_gi), ("top", _compressor_top)):
        try:
            info[field] = fn()
        except Exception:
            pass
    try:
        info["files"], info["quantum_gi"], info["grew_24h"] = \
            _swap_files(time.time())
    except Exception:
        pass
    return info


def swap_line(info):
    """'Swap: ...' page line, or None when there is no swap to report."""
    used = info.get("used_gi")
    if not used:
        return None
    fmt = lambda v: "unknown" if v is None else v
    return (f"Swap: used {used:.1f} Gi in {fmt(info.get('files'))} swapfiles "
            f"(grew by {fmt(info.get('grew_24h'))} in 24h); "
            f"top compressed: {fmt(info.get('top'))}")


def swap_is_cause(info, free, target):
    used = info.get("used_gi")
    if not used or used <= 0 or free is None:
        return False
    shortfall = max(0.0, float(target) - float(free))
    return shortfall > 0 and used >= 0.5 * shortfall


FIXTURE_VARS = ("DISK_GUARD_FAKE_FREE_GI", "DISK_GUARD_FAKE_SWAP_GI")


def _active_fixtures():
    return {v: os.environ.get(v) for v in FIXTURE_VARS
            if os.environ.get(v) not in (None, "")}


def fixture_active():
    """True when the free-space figure is a FIXTURE, not a measurement.

    CLASS this fences (measured 2026-09-25): a pytest run of
    tests/test_disk_guard_incident_key.py created two real CTO cards on the
    shared board (t_849189ed, t_0fbf5b8d, 19:04:47/49 — the same second the
    test bytecode was rewritten) reporting "8.0Gi free (floor 10.0Gi)" while
    the host actually had 23Gi free and the production floor is 0.5Gi. One
    test in that file drives dga.main() while patching only the OTHER
    channels, so the unpatched deliver_kanban_cto ran for real. A synthetic
    incident that pages a human and consumes a worker is worse than no alert:
    it spends the credibility of the channel that has to survive a real ENOSPC.

    The fence lives HERE, with the side effect, not in the tests. Requiring
    every present and future test to remember to patch every channel is a
    hand-maintained invariant; deriving "this figure is not real" from the
    fixture env var is self-detecting and cannot be forgotten. --dry-run stays
    the supported way to exercise the rendering.

    A fake SWAP figure is a fixture too: it changes the page's stated cause.
    """
    return bool(_active_fixtures())


def _refuse_if_fixture(channel):
    if fixture_active():
        raise RuntimeError(
            f"{channel}: refusing to page from a FIXTURE measurement ("
            + ", ".join(f"{k}={v!r}" for k, v in _active_fixtures().items())
            + "). Patch the channel in the test, or use --dry-run.")


def should_alert(free, floor):
    # Unknown free space is itself an anomaly: silence is the failure mode
    # this module exists to remove.
    if free is None:
        return True
    # `<=`, not `<`: with the floor at 1Gi a strict comparison would only ever
    # fire at 0Gi -- i.e. after the host is already dead.
    return float(free) <= float(floor)


# -------------------------------------------------------------- channels ---

def _config_get(key):
    for argv in ([HERMES, "--profile", "default", "config", "get", key],
                 [HERMES, "config", "get", key]):
        out = _run(argv).strip()
        if out and not out.lower().startswith("config key not set"):
            return out
    return ""


def wake_config():
    return {
        "url": _config_get("kanban_wake.url")
               or "http://127.0.0.1:8645/webhooks/kanban-wake",
        "secret_file": _config_get("kanban_wake.secret_file")
                       or "~/.secrets/kanban-wake-secret",
    }


def deliver_wake(text):
    """DEPRECATED ROUTE — kept only so an old caller cannot silently lose a page.

    This POSTed into the kanban-wake webhook, which is served by the
    agentpod-ceo gateway (port 8645, platforms.webhook in that profile's
    config.yaml): every disk page woke the EM. Host disk capacity is not EM
    scope -- it is the CTO's (owner decision, card t_b8d0aaeb, 2026-09-24), and
    an EM woken for a host problem can only re-delegate it, which is how this
    incident reached an operator late. The CTO route is deliver_kanban_cto.
    """
    cfg = wake_config()
    _refuse_if_fixture("deliver_wake")
    with open(os.path.expanduser(cfg["secret_file"])) as fh:
        secret = fh.read().strip()
    body = json.dumps({
        "text": text,
        "task_id": "",
        "event": "disk_guard",
    }).encode()
    ts = str(int(time.time()))
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + body,
                   hashlib.sha256).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": str(uuid.uuid4()),
        "X-Webhook-Timestamp": ts,
        "X-Webhook-Signature-V2": sig,
    }
    req = urllib.request.Request(cfg["url"], data=body, headers=headers,
                                 method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()
    return True


# The profile that OWNS host infrastructure. Not a routing preference: the EM
# cannot act on an ENOSPC host, and a page whose only recipient must re-delegate
# it is a page that arrives late.
CTO_PROFILE = os.environ.get("DISK_GUARD_OWNER_PROFILE", "cto")
KANBAN_BIN = os.path.expanduser(
    os.environ.get("DISK_GUARD_KANBAN_BIN",
                   "~/.hermes/hermes-agent/venv/bin/hermes"))


def deliver_kanban_cto(text, key=None):
    """Raise a CTO-assigned card on the shared board.

    Chosen over waking a gateway session because this channel SURVIVES the
    failure it reports: a card is a row in kanban.db that the dispatcher picks
    up whenever a worker is next free, whereas a gateway wake is lost if the
    gateway is the thing ENOSPC broke.

    The incident key is the idempotency key, so the board-level dedupe matches
    the page-level dedupe exactly: one incident is one card no matter how many
    ticks observe it. A NEW incident key (state materially changed) gets a new
    card, which is the same rule the Telegram channel uses.
    """
    _refuse_if_fixture("deliver_kanban_cto")
    if not os.path.exists(KANBAN_BIN):
        raise RuntimeError(f"kanban cli not found at {KANBAN_BIN}")
    title = text.strip().splitlines()[0][:120]
    argv = [
        KANBAN_BIN, "kanban", "create", title,
        "--assignee", CTO_PROFILE,
        "--body", text,
        "--created-by", "disk-guard",
        "--idempotency-key", f"disk-guard-{key or 'unkeyed'}",
        "--json",
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(
            f"kanban create rc={proc.returncode}: {proc.stderr.strip()[:200]}")
    try:
        tid = json.loads(proc.stdout)["id"]
    except Exception:
        raise RuntimeError("kanban create returned no task id")
    print(f"disk-guard: raised CTO card {tid}")
    return True


def _bot_token():
    tok = os.environ.get("DISK_GUARD_BOT_TOKEN", "").strip()
    if tok:
        return tok
    env = os.path.expanduser("~/.hermes/.env")
    with open(env) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("TELEGRAM_BOT_TOKEN=") and len(line) > 19:
                return line.split("=", 1)[1].strip()
    raise RuntimeError("no TELEGRAM_BOT_TOKEN available")


def deliver_telegram(text):
    """Direct Telegram sendMessage to Den — the out-of-band channel that
    still works when the gateway itself is the thing that is broken."""
    _refuse_if_fixture("deliver_telegram")
    token = _bot_token()
    chat = os.environ.get("DISK_GUARD_TELEGRAM_CHAT", DEN_TELEGRAM_ID)
    body = json.dumps({"chat_id": chat, "text": text}).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        payload = json.loads(r.read() or b"{}")
    if not payload.get("ok", False):
        raise RuntimeError(f"sendMessage failed: {payload}")
    return True


def alert(text, key=None):
    """Page every channel independently. True if at least one landed.

    Channels are deliberately of DIFFERENT KINDS: a durable board card (works
    when a gateway is dead) and a direct Telegram send (works when the board or
    the disk is dead). Two instances of the same mechanism would share a failure
    mode, which is exactly how a RED guard went unheard for 501 runs.
    """
    landed = False
    for channel in (deliver_kanban_cto, deliver_telegram):
        try:
            ok = channel(text, key) if channel is deliver_kanban_cto \
                else channel(text)
            if ok:
                landed = True
        except Exception as exc:  # a dead channel must not swallow the page
            print(f"disk-guard alert: {channel.__name__} failed: {exc}",
                  file=sys.stderr)
    return landed


# ----------------------------------------------------------------- state ---

def state_path():
    return os.environ.get("DISK_GUARD_STATE") or os.path.expanduser(DEFAULT_STATE)


def load_state():
    try:
        with open(state_path()) as fh:
            st = json.load(fh)
        return st if isinstance(st, dict) else {}
    except Exception:
        return {}


def save_state(st):
    p = state_path()
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    tmp = f"{p}.tmp"
    with open(tmp, "w") as fh:
        json.dump(st, fh, indent=1, sort_keys=True)
    os.replace(tmp, p)


_PATH_LINE = re.compile(r"^\s*(\d+)\s*MB\s+(\S.*?)\s*$")
# A leaf that is an instance id, not a place: uuid, long hex, pure digits, or a
# kanban/worktree task id. Two ticks of the same incident must not look like two
# incidents because a session directory has a fresh uuid.
_INSTANCE_LEAF = re.compile(
    r"^(?:[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}|[0-9a-fA-F]{12,}|\d+|t_[0-9a-fA-F]+)$"
)
# A size is the same size for keying purposes unless it moves by a quarter.
SIZE_CHANGE_RATIO = 0.25


def normalize_path(path):
    """Collapse a ranked path to its STABLE identity.

    ~ for $HOME (the key must not change when the guard runs as a different
    user in a harness) and an instance-shaped leaf folded into its parent, so
    `.copilot/session-state/<uuid>` is one place, not one place per session.
    """
    p = (path or "").strip().rstrip("/")
    home = os.path.expanduser("~")
    if p.startswith(home + "/"):
        p = "~" + p[len(home):]
    head, _, leaf = p.rpartition("/")
    if head and _INSTANCE_LEAF.match(leaf):
        p = head
    return p


def parse_paths(detail):
    """{normalized path: MB} parsed out of a rendered remediation list.

    DERIVED from the rendered detail, not from a hand-kept list of the paths we
    expect to see: a newly ranked root is picked up with no edit here.
    """
    out = {}
    for line in (detail or "").splitlines():
        m = _PATH_LINE.match(line)
        if not m:
            continue
        mb, path = int(m.group(1)), normalize_path(m.group(2))
        out[path] = max(out.get(path, 0), mb)
    return out


def _sizes_changed_materially(prev_paths, paths):
    """True when a shared path crossed +/-25%. Sizes drive the MESSAGE; only a
    material move is allowed to drive the KEY."""
    for name, mb in paths.items():
        old = (prev_paths or {}).get(name)
        if old is None:
            continue
        if old <= 0:
            return mb > 0
        if abs(mb - old) / float(old) > SIZE_CHANGE_RATIO:
            return True
    return False


def resolve_incident(free, floor, detail, prev=None):
    """(key, paths_to_store, generation) for this tick.

    CLASS this fixes — measured on 2026-09-21: the guard paged 17 times in one
    day for ONE unchanging state (8Gi free, floor 10Gi) because the key hashed
    the rendered detail text, and that text is not a function of the state. Two
    renderings alternated on consecutive ticks:
      (a) "No single reclaimable path over 100MB ..."
      (b) "1746MB pnpm/store/v10 + 44xMB .copilot/session-state/<uuid>"
    (a) was NOT a different state and NOT "everything is under the threshold":
    reproduced by holding an open fd on those two directories, which makes
    disk-guard.sh's lsof-based `path_in_use` drop them from the ranking. Any
    unrelated process touching a scratch dir erased the evidence for that tick.

    So the key is built from stable identity only:
      * the floor and the free-GiB figure,
      * the SET of normalized path names (no sizes, no uuid leaves),
      * a generation counter bumped only when a path crosses +/-25%.
    And, decisively: an EMPTY ranking is absence of evidence, not evidence of
    change. It inherits the running incident's key instead of minting a new one
    (and so does the first tick that regains evidence for an incident whose
    previous tick had none).
    """
    prev = prev or {}
    paths = parse_paths(detail)
    prev_paths = prev.get("last_paths") or {}
    prev_key = prev.get("last_key")
    gen = int(prev.get("last_gen") or 0)

    same_state = (
        prev_key
        and prev.get("last_floor") == floor
        and prev.get("last_free") == free
    )

    if same_state:
        # The incident's IDENTITY is the path set we last stored for it. The
        # key stays pinned to prev_key for as long as that identity holds, so
        # an inherited key survives the next tick instead of being replaced by
        # a freshly computed hash (that leak re-paged the incident exactly once
        # more whenever it began on an lsof-suppressed, empty-ranking tick).
        if not paths:
            # Absence of evidence: carry the incident AND its identity.
            return prev_key, prev_paths, gen
        if not prev_paths:
            # Regaining evidence for a running incident is not new information:
            # adopt the ranking as the identity, keep the key.
            return prev_key, paths, gen
        if sorted(paths) == sorted(prev_paths):
            if not _sizes_changed_materially(prev_paths, paths):
                return prev_key, paths, gen
            gen += 1

    names = "|".join(sorted(paths))
    blob = f"{free}|{floor}|{gen}|{names}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16], paths, gen


def incident_key(free, floor, detail, prev=None):
    """Stable identity of this incident (see resolve_incident)."""
    return resolve_incident(free, floor, detail, prev)[0]


def suppressed(free, now=None, key=None):
    """True when this exact incident already paged inside the TTL.

    Three independent reasons to re-page: the incident CHANGED (different key),
    the TTL expired, or the previous page never LANDED. An undelivered page
    (every channel failed, or a fixture was refused) informed nobody, so it
    must never dedupe the next attempt. Anything else is a repeat.
    """
    now = now or int(time.time())
    st = load_state()
    last_at = int(st.get("last_at") or 0)
    if not st:
        return False
    if st.get("landed") is False:
        return False          # nobody heard it: try again
    if (now - last_at) >= REPEAT_WINDOW_SECONDS:
        return False          # TTL expired: re-page even if unchanged
    if key is not None:
        if st.get("last_key") != key:
            return False      # content changed: new information
        return True
    # Legacy numeric path (no key supplied): a materially worse figure pages.
    last_free = st.get("last_free")
    if last_free is None or free is None:
        return False
    if free <= last_free - MATERIAL_DROP_GI:
        return False
    return True


# ------------------------------------------------------------------ main ---

def main(argv=None):
    ap = argparse.ArgumentParser(description="Host disk guard alert channel")
    # float, not int: the floor is a decimal GiB decision (0.5Gi since
    # 2026-09-22). `type=int` raised SystemExit 2 on `--floor 0.5`, and the
    # cron wrapper passes exactly that -- an int parser here turns the page
    # path into a crash at the moment it is needed.
    ap.add_argument("--floor", type=float,
                    default=float(os.environ.get("DISK_GUARD_FLOOR_GI",
                                                 DEFAULT_FLOOR_GI)))
    ap.add_argument("--detail", default="", help="extra body appended to the page")
    ap.add_argument("--no-alert", action="store_true",
                    help="measure and report only; never page")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the page that WOULD be sent; touch no channel "
                         "and no dedupe state. Use this for iteration — manual "
                         "test fires are what paged the CEO five times on "
                         "2026-09-21.")
    args = ap.parse_args(argv)

    # The floor is the whole point; never trust it downward.
    floor = max(args.floor, DEFAULT_FLOOR_GI)
    free = free_gi()

    if not should_alert(free, floor):
        print(f"disk-guard: OK {float(free):.1f}Gi free (floor {floor}Gi)")
        return 0

    shown = "unknown" if free is None else f"{float(free):.1f}Gi"
    swap = swap_info()
    try:
        target = float(os.environ.get("DISK_GUARD_RECLAIM_TARGET_GI") or 25)
    except ValueError:
        target = 25.0
    cause = (" (cause: swap pressure, not scratch)"
             if swap_is_cause(swap, free, target) else "")
    text = (
        f"DISK GUARD RED on this supervisor host{cause}: {shown} free "
        f"(floor {floor}Gi, pages at or below it).\n"
        "An ENOSPC host breaks every gateway write, kanban dispatch and "
        "watchdog while the board still reports green."
    )
    sline = swap_line(swap)
    if sline:
        text += "\n" + sline
    if args.detail:
        text += "\n" + args.detail

    if args.no_alert or args.dry_run:
        print(text)
        return 1

    prev = load_state()
    key, paths, gen = resolve_incident(free, floor, args.detail, prev)
    if suppressed(free, key=key):
        print(f"disk-guard: still RED at {shown}; page suppressed (identical "
              f"incident {key} inside the {REPEAT_WINDOW_SECONDS // 3600}h TTL)")
        # Refresh the evidence WITHOUT touching last_at: a suppressed tick must
        # not extend the TTL (that would make a genuine 6h re-page never fire),
        # but a tick that regained a ranking must record it, or the next tick
        # compares against an empty set and mints a new key.
        if (paths and paths != (prev.get("last_paths") or {})
                and not fixture_active()):
            prev.update({"last_paths": paths, "last_gen": gen,
                         "last_floor": floor})
            save_state(prev)
        return 1

    landed = alert(text, key=key)
    if fixture_active():
        # A fixture never mutates dedupe state: a refused synthetic page that
        # wrote last_at/last_key would silence the identical REAL incident.
        print("disk-guard: fixture run; dedupe state left untouched",
              file=sys.stderr)
    else:
        save_state({"last_free": free, "last_at": int(time.time()),
                    "last_key": key, "last_paths": paths, "last_gen": gen,
                    "last_floor": floor,
                    "landed": bool(landed)})
    print(text)
    if not landed:
        print("disk-guard: NO alert channel accepted the page", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
