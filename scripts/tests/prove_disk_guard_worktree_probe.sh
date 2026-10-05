#!/usr/bin/env bash
# prove_disk_guard_worktree_probe.sh
#
# CLASS: disk-guard's worktree capability probe treated every non-zero
# `hermes kanban reclaim --dry-run` as "upgrade the installed hermes". Under
# HERMES_DELEGATED_CHILD_CONTEXT that refusal is the Kanban write fence
# ("delegate_task child contexts cannot mutate Kanban tasks via the CLI"),
# not a stale binary. A fenced child must SKIP. A real probe failure must
# log rc + stderr, not a generic upgrade hint.
#
# Drives the REAL reclaim_hermes_worktrees via the DISK_GUARD_LIB seam against
# a stub hermes and a fixture HOME, so the harness cannot reclaim live
# worktrees and cannot drift from the script it protects.
#
# Run: env -u HERMES_DELEGATED_CHILD_CONTEXT bash scripts/tests/prove_disk_guard_worktree_probe.sh
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
GUARD="${DISK_GUARD:-$HERE/../disk-guard.sh}"
FAIL=0
ok()  { printf 'PASS  %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; FAIL=1; }

[ -f "$GUARD" ] && ok "guard exists ($GUARD)" || { bad "guard missing: $GUARD"; echo "=== RESULT: FAIL ==="; exit 1; }
bash -n "$GUARD" && ok "guard parses" || bad "guard has a syntax error"
grep -q 'reclaim_hermes_worktrees' "$GUARD" \
  && ok "guard contains reclaim_hermes_worktrees" \
  || bad "guard has no reclaim_hermes_worktrees — probe fix is gone"

FIX=$(mktemp -d -t dgwtprobe)
BIN="$FIX/bin"
mkdir -p "$BIN" "$FIX/home"
cat > "$BIN/hermes" << 'EOF'
#!/usr/bin/env bash
# Stub: fence refusal, broken probe, or a successful reclaim summary.
if [ -n "${HERMES_DELEGATED_CHILD_CONTEXT:-}" ]; then
  echo "delegate_task child contexts cannot mutate Kanban tasks via the CLI" >&2
  exit 1
fi
if [ "${STUB_BROKEN:-0}" = 1 ]; then
  echo "unrecognized arguments: --dry-run" >&2
  exit 2
fi
printf '%s\n' "  -> 2 removed, 1 kept"
exit 0
EOF
chmod +x "$BIN/hermes"

# Source the real script, then replace log() so assertions see the message
# without the timestamp prefix the production logger adds.
drive() {
  env "$@" bash -c '
    set -uo pipefail
    export HOME="'"$FIX/home"'"
    export DISK_GUARD_HERMES_BIN="'"$BIN/hermes"'"
    export PATH="'"$BIN"':$PATH"
    DISK_GUARD_LIB=1 source "'"$GUARD"'"
    log() { echo "$*"; }
    reclaim_hermes_worktrees
  '
}

SKIP_LINE="worktrees: SKIP fenced child context -- run disk-guard from an operator shell/launchd"

# 1. Fenced child: exact SKIP, no upgrade hint, no shell-fallback FAIL.
out=$(drive -u STUB_BROKEN HERMES_DELEGATED_CHILD_CONTEXT=1)
printf '%s\n' "$out" | grep -qx "$SKIP_LINE" \
  && ok "fenced child logs SKIP and returns" \
  || { bad "fenced child did not log the exact SKIP line"; printf '    got: %s\n' "$out"; }
printf '%s\n' "$out" | grep -q 'upgrade' \
  && bad "fenced child told the operator to upgrade hermes" \
  || ok "fenced child does not say upgrade"
printf '%s\n' "$out" | grep -q 'FAIL' \
  && bad "fenced child logged FAIL" \
  || ok "fenced child does not FAIL"

# 2. Clean operator shell: success path unchanged.
out=$(drive -u HERMES_DELEGATED_CHILD_CONTEXT -u STUB_BROKEN)
printf '%s\n' "$out" | grep -F "worktrees:   -> 2 removed, 1 kept" >/dev/null \
  && ok "clean probe reclaims and logs the hermes summary" \
  || { bad "clean probe did not log the reclaim summary"; printf '    got: %s\n' "$out"; }

# 3. Broken probe: FAIL carries rc and stderr, not a bare "unavailable/upgrade".
out=$(drive -u HERMES_DELEGATED_CHILD_CONTEXT STUB_BROKEN=1)
want="worktrees: FAIL hermes reclaim probe rc=2 at $BIN/hermes: unrecognized arguments: --dry-run"
printf '%s\n' "$out" | grep -F "$want" >/dev/null \
  && ok "broken probe logs rc=2 and stderr" \
  || { bad "broken probe did not log rc=2 with stderr"; printf '    got: %s\n' "$out"; }

rm -rf "$FIX"

[ "$FAIL" = 0 ] && { echo "=== RESULT: PASS ==="; exit 0; }
echo "=== RESULT: FAIL ==="
exit 1
