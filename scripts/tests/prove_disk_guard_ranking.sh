#!/usr/bin/env bash
# Proof harness for the disk-guard "top reclaimable" ranking.
#
# CLASS proven: the alert of 2026-09-21T02:01 listed six /private/tmp files of
# ONE du block each (wake.out = 30 bytes, hb.bak = 18KB) as the top reclaimable
# paths. The scan covered the wrong roots (kanban workspaces + /private/tmp only)
# and ranked by du blocks, so an operator following it would delete nothing.
# An alert whose remediation list is noise is worse than no alert.
#
# Requirements encoded here:
#   R1 entries are ordered bytes-descending
#   R2 sub-100MB trivia never appears
#   R3 the real scratch roots (worktrees, pnpm store, hermes logs/sessions) are
#      covered — proven by planting a large dir under a real root and requiring
#      it to surface
#   R4 a path with a LIVE process in it is never listed as reclaimable
#   R5 a path the guard's OWN reclaim classes are forbidden to delete is never
#      listed. CLASS (measured 2026-10-05, card t_4b6e5ab2): the RED page named
#      ~/.cache/huggingface (guard_hf_cache protects it) and
#      ~/Library/pnpm/store/v10 (reclaim_stale_pnpm_stores logged
#      "skip (referenced)" every tick). An operator following the list deleted
#      both: local STT model gone, three projects forced to reinstall. A
#      remediation list that recommends what the automation refuses to do is
#      worse than no list.
#   R6 the swap term is REPORTED (card t_515b8493): swap competes for the same
#      APFS container, so a check-only run below target must name swap in its
#      shortfall line, and reclaim_to_target's exhausted line must too. Fixture
#      figures only; nothing is deleted.
#   R7 every swap probe degrades silently: failing sysctl/top, a hanging top
#      (bounded by the 5s timeout) and an unreadable swap dir still exit 0;
#      and with sysctl, top AND stat all hanging 30s (stat over real-looking
#      swapfiles) one tick still finishes in <20s, rc 0, reporting unknown.
#
# Run: bash scripts/tests/prove_disk_guard_ranking.sh   (guards THIS checkout;
#      DISK_GUARD_SCRIPT=<path> proves another copy, e.g. a mutation)
#      PROVE_SWAP_ONLY=1 (or --only-swap) runs just R6/R7, skipping the slow du scan.
set -uo pipefail

GUARD="${DISK_GUARD_SCRIPT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/disk-guard.sh}"
echo "guard under test: $GUARD"
[ "${1:-}" = "--only-swap" ] && PROVE_SWAP_ONLY=1
SWAP_ONLY="${PROVE_SWAP_ONLY:-0}"
fail=0
check() { if [ "$2" = "$3" ]; then echo "PASS $1 ($3)"; else echo "FAIL $1: expected '$2' got '$3'"; fail=1; fi; }

SANDBOX=$(mktemp -d "$HOME/.hermes/logs/ranking-proof.XXXXXX")
cleanup() { kill "${live_pid:-}" 2>/dev/null; rm -rf "$SANDBOX"; }
trap cleanup EXIT

if [ "$SWAP_ONLY" != 1 ]; then
# Plant a big dir under a REAL scratch root, plus trivia that must not rank.
BIG="$HOME/workspace/AgentPod-worktrees/zzz-ranking-proof-big"
SMALL="$HOME/workspace/AgentPod-worktrees/zzz-ranking-proof-small"
INUSE="$HOME/workspace/AgentPod-worktrees/zzz-ranking-proof-inuse"
mkdir -p "$BIG" "$SMALL" "$INUSE"
cleanup() { kill "${live_pid:-}" 2>/dev/null; rm -rf "$SANDBOX" "$BIG" "$SMALL" "$INUSE"; }
trap cleanup EXIT

# Real bytes, not sparse: `mkfile -n` / `dd` from /dev/zero can produce a
# sparse or compressed file that du reports as 1 block, which would make R3
# fail for a reason that has nothing to do with the ranking. /dev/urandom is
# incompressible, so the size du sees is the size on disk.
dd if=/dev/urandom of="$BIG/blob" bs=1m count=300 >/dev/null 2>&1
dd if=/dev/urandom of="$INUSE/blob" bs=1m count=220 >/dev/null 2>&1
printf 'tiny\n' > "$SMALL/note.txt"

# A live process holding the path as its cwd (what path_in_use derives on).
# It must outlive the whole du scan across every scratch root, which takes
# minutes on a full disk — a 120s holder dies mid-scan and R4 then fails for
# a harness reason rather than a guard reason.
( cd "$INUSE" && exec sleep 1800 ) &
live_pid=$!
sleep 1
cleanup() { kill "${live_pid:-}" 2>/dev/null; rm -rf "$SANDBOX" "$BIG" "$SMALL" "$INUSE"; }
trap cleanup EXIT

# Fail loudly rather than vacuously if the holder is not actually live.
kill -0 "$live_pid" 2>/dev/null || { echo "FAIL harness: in-use holder never started"; exit 1; }

out=$(bash "$GUARD" --top-reclaimable 2>/dev/null)

if [ -z "$out" ]; then
  echo "FAIL harness: --top-reclaimable produced no output"; exit 1
fi
echo "--- ranking output ---"; printf '%s\n' "$out"; echo "----------------------"

# Anti-vacuity: R1/R2/R4 are all trivially satisfiable by an empty or log-only
# list, which is exactly how a broken ranking would sneak through. Every line
# must be "<integer MB>\t<absolute path>" and there must be at least two.
lines=$(printf '%s\n' "$out" | wc -l | tr -d ' ')
shaped=$(printf '%s\n' "$out" | grep -cE '^[0-9]+[[:space:]]+/')
check "R0 output is a real ranking (>=2 well-formed rows)" "yes" \
      "$([ "$lines" -ge 2 ] && [ "$shaped" = "$lines" ] && echo yes || echo no)"

# R1: bytes-descending. First field is MB.
sizes=$(printf '%s\n' "$out" | awk '{print $1}')
sorted=$(printf '%s\n' "$sizes" | sort -rn)
check "R1 entries are bytes-descending" "$sorted" "$sizes"

# R2: nothing under 100MB
tiny=$(printf '%s\n' "$out" | awk '$1 < 100' | wc -l | tr -d ' ')
check "R2 no sub-100MB trivia listed" "0" "$tiny"

# R3: a large dir under a real scratch root surfaces
hit=$(printf '%s\n' "$out" | grep -c "zzz-ranking-proof-big")
check "R3 large dir under a real scratch root is listed" "1" "$hit"

# R4: an in-use path is never offered for deletion
inuse_hit=$(printf '%s\n' "$out" | grep -c "zzz-ranking-proof-inuse")
check "R4 in-use path is not listed" "0" "$inuse_hit"

# R5: a protected path (one a reclaim class refuses to delete) is never listed.
# Derived from the guard itself, not a copy of the predicate: source the library
# seam, ask protected_paths() what it protects, and require that NO ranked row
# sits at or under any of them. Anti-vacuity: protected_paths must be non-empty
# (the HF cache path is unconditional), or R5 would pass on a broken guard.
prot=$(DISK_GUARD_LIB=1 . "$GUARD" >/dev/null 2>&1; protected_paths 2>/dev/null | sort -u)
nprot=$(printf '%s\n' "$prot" | grep -c '^/')
check "R5a guard exposes a non-empty protected set" "yes" \
      "$([ "${nprot:-0}" -ge 1 ] && echo yes || echo no)"

prot_hit=0
while read -r p; do
  [ -n "$p" ] || continue
  printf '%s\n' "$out" | awk -v pp="$p" '{ $1=""; sub(/^[ \t]+/,""); \
        if ($0 == pp || index($0, pp "/") == 1) exit 0 } END { exit 1 }' \
    && { echo "  protected path offered for deletion: $p"; prot_hit=1; }
done <<EOF
$prot
EOF
check "R5b no protected path is listed as reclaimable" "0" "$prot_hit"
fi

# --- R6: swap is reported as a measured term (fixture figures, no deletion).
r6_out=$(DISK_GUARD_FAKE_FREE_GI=10 DISK_GUARD_FAKE_SWAP_GI=28.8 DISK_GUARD_SKIP_TOP=1 \
         bash "$GUARD" 2>&1); r6_rc=$?
echo "--- R6 guard output ---"; printf '%s\n' "$r6_out"; echo "----------------------"
check "R6a free= line reports swap_used=28.8Gi" "yes" \
      "$(printf '%s\n' "$r6_out" | grep -Eq ' free=10\.00Gi floor=[0-9.]+Gi target=[0-9.]+Gi swap_used=28\.8Gi ' && echo yes || echo no)"
r6_bt=$(printf '%s\n' "$r6_out" | grep 'below_target:')
check "R6b below_target line names swap" "yes" \
      "$(printf '%s\n' "$r6_bt" | grep -q 'of which swap accounts for' && echo yes || echo no)"
check "R6b' below_target line carries swap_used=28.8Gi unclipped" "yes" \
      "$(printf '%s\n' "$r6_bt" | grep -q 'swap_used=28\.8Gi measured' && echo yes || echo no)"
check "R6c swap term never changes the exit code" "0" "$r6_rc"

# R6d: the reclaim_to_target exhausted line, driven through the library seam
# with the deleting classes stubbed out so NOTHING is removed.
r6d_out=$(
  export DISK_GUARD_FAKE_FREE_GI=10 DISK_GUARD_FAKE_SWAP_GI=28.8 DISK_GUARD_SKIP_TOP=1
  DISK_GUARD_LIB=1 . "$GUARD" >/dev/null 2>&1
  reclaim_user_tmp() { :; }
  reclaim_tmp() { :; }
  reclaim_to_target 2>&1
)
echo "--- R6d reclaim_to_target output ---"; printf '%s\n' "$r6d_out"; echo "----------------------"
check "R6d reclaim_to_target attributes min(swap, shortfall) and reports swap unclipped" "yes" \
      "$(printf '%s\n' "$r6d_out" | grep -q 'shortfall to target 15\.0Gi, of which swap accounts for 15\.0Gi (swap_used=28\.8Gi measured' && echo yes || echo no)"

# --- R7: failure paths degrade silently and stay bounded.
STUBS="$SANDBOX/stubs"; mkdir -p "$STUBS"
printf '#!/bin/sh\nexit 1\n' > "$STUBS/sysctl"
printf '#!/bin/sh\nexec sleep 30\n' > "$STUBS/top"
chmod +x "$STUBS/sysctl" "$STUBS/top"
r7_start=$(date +%s)
r7_out=$(env -u DISK_GUARD_FAKE_SWAP_GI -u DISK_GUARD_SKIP_TOP PATH="$STUBS:$PATH" \
         DISK_GUARD_SWAP_DIR=/nonexistent DISK_GUARD_FAKE_FREE_GI=10 \
         bash "$GUARD" 2>&1); r7_rc=$?
r7_secs=$(( $(date +%s) - r7_start ))
echo "--- R7 guard output (${r7_secs}s) ---"; printf '%s\n' "$r7_out"; echo "----------------------"
check "R7a failing probes never change the exit code" "0" "$r7_rc"
check "R7b failed sysctl reports swap_used=unknown" "yes" \
      "$(printf '%s\n' "$r7_out" | grep -q 'swap_used=unknown' && echo yes || echo no)"
check "R7c unreadable swap dir reports swapfiles=unreadable" "yes" \
      "$(printf '%s\n' "$r7_out" | grep -q 'swapfiles=unreadable' && echo yes || echo no)"
check "R7d failed/hung top omits compressor_top" "0" \
      "$(printf '%s\n' "$r7_out" | grep -c 'compressor_top=')"
# One top probe per check-only run, bounded at 5s, plus slack.
check "R7e hung top is bounded by the timeout" "yes" \
      "$([ "$r7_secs" -le 15 ] && echo yes || echo no)"

# R7f: EVERY probe hangs. sysctl, top and stat sleep 30s; the swap dir holds
# real-looking swapfiles so the single stat probe actually runs (a missing dir
# never reaches stat). The tick must stay bounded, exit 0 and say unknown.
HANG="$SANDBOX/hang"; VMDIR="$SANDBOX/vm"; mkdir -p "$HANG" "$VMDIR"
for c in sysctl top stat; do printf '#!/bin/sh\nexec sleep 30\n' > "$HANG/$c"; chmod +x "$HANG/$c"; done
: > "$VMDIR/swapfile0"; : > "$VMDIR/swapfile1"
r7f_start=$(date +%s)
r7f_out=$(env -u DISK_GUARD_FAKE_SWAP_GI -u DISK_GUARD_SKIP_TOP PATH="$HANG:$PATH" \
          DISK_GUARD_SWAP_DIR="$VMDIR" DISK_GUARD_FAKE_FREE_GI=10 \
          bash "$GUARD" 2>&1); r7f_rc=$?
r7f_secs=$(( $(date +%s) - r7f_start ))
echo "--- R7f all-probes-hang output (${r7f_secs}s) ---"; printf '%s\n' "$r7f_out"; echo "----------------------"
check "R7f all-hung tick exits 0" "0" "$r7f_rc"
check "R7f all-hung tick finishes in <20s" "yes" "$([ "$r7f_secs" -lt 20 ] && echo yes || echo no)"
check "R7f hung sysctl -> swap_used=unknown" "yes" \
      "$(printf '%s\n' "$r7f_out" | grep -q 'swap_used=unknown' && echo yes || echo no)"
check "R7f hung stat -> swapfiles=unreadable" "yes" \
      "$(printf '%s\n' "$r7f_out" | grep -q 'swapfiles=unreadable' && echo yes || echo no)"
check "R7f hung top -> compressor_top omitted" "0" \
      "$(printf '%s\n' "$r7f_out" | grep -c 'compressor_top=')"

exit "$fail"
