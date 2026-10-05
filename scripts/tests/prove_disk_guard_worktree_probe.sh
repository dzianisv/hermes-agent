#!/usr/bin/env bash
# prove_disk_guard_worktree_probe.sh
#
# CLASS: disk-guard's worktree capability probe treated every non-zero
# `hermes kanban reclaim --dry-run` as "upgrade the installed hermes". Under
# HERMES_DELEGATED_CHILD_CONTEXT that refusal is the Kanban write fence
# ("delegate_task child contexts cannot mutate Kanban tasks via the CLI"),
# not a stale binary. A fenced child must SKIP every worktree/workspace
# deletion in the run, quoting the real refusal, ONLY when the probe exits 1
# AND a stderr line is exactly that sentence (with or without the "kanban: "
# prefix; "Command helper:" preamble lines are ignored). A substring is not
# enough. Any other fenced probe is FAIL with rc + the first non-preamble
# stderr line, and still deletes nothing. The env var is never cleared.
#
# Drives the REAL functions via the DISK_GUARD_LIB seam against a stub hermes
# and a fixture HOME, so the harness cannot reclaim live worktrees and cannot
# drift from the script it protects. HOME is exported to the fixture BEFORE
# source; worktree_repos derives from HOME.
#
# Run:
#   env -u HERMES_DELEGATED_CHILD_CONTEXT bash scripts/tests/prove_disk_guard_worktree_probe.sh
#   HERMES_DELEGATED_CHILD_CONTEXT=1 bash scripts/tests/prove_disk_guard_worktree_probe.sh
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
GUARD="${DISK_GUARD:-$HERE/../disk-guard.sh}"
FAIL=0
ok()  { printf 'PASS  %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; FAIL=1; }

[ -f "$GUARD" ] && ok "guard exists ($GUARD)" || { bad "guard missing: $GUARD"; echo "=== RESULT: FAIL ==="; exit 1; }
bash -n "$GUARD" && ok "guard parses" || bad "guard has a syntax error"

FIX=$(mktemp -d -t dgwtprobe)
BIN="$FIX/bin"
HOME_FIX="$FIX/home"
WT="$HOME_FIX/workspace/wt-t_deadbeef"
REPO="$HOME_FIX/workspace/repo"
mkdir -p "$BIN" "$HOME_FIX/workspace"

cat > "$BIN/hermes" << 'EOF'
#!/usr/bin/env bash
# Stub: preamble on stderr (live hermes does this), then explicit stderr,
# fence refusal, broken probe, or a successful summary on stdout.
preamble() { echo "  Command helper: applied 1 secret" >&2; }
if [ -n "${STUB_STDERR:-}" ]; then
  preamble
  printf '%s\n' "$STUB_STDERR" >&2
  exit "${STUB_RC:-1}"
fi
if [ -n "${HERMES_DELEGATED_CHILD_CONTEXT:-}" ]; then
  preamble
  echo "delegate_task child contexts cannot mutate Kanban tasks via the CLI" >&2
  exit 1
fi
if [ "${STUB_BROKEN:-0}" = 1 ]; then
  preamble
  echo "unrecognized arguments: --dry-run" >&2
  exit 2
fi
printf '%s\n' "  -> 2 removed, 1 kept"
exit 0
EOF
chmod +x "$BIN/hermes"

git -C "$HOME_FIX/workspace" init -q repo
git -C "$REPO" config user.email "disk-guard-test@example.com"
git -C "$REPO" config user.name "disk-guard-test"
git -C "$REPO" config commit.gpgsign false
git -C "$REPO" commit -q --allow-empty -m init

reset_wt() {
  git -C "$REPO" worktree prune >/dev/null 2>&1 || true
  git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || rm -rf "$WT"
  git -C "$REPO" branch -D card-t_deadbeef >/dev/null 2>&1 || true
  git -C "$REPO" worktree add -q "$WT" -b card-t_deadbeef
}

# Board workspace reclaim_workspaces deletes on a clean run: the dir sits under
# boards/<slug>/workspaces/<task-id>/ and that board's kanban.db marks the
# task done. All of this is under the fixture HOME, never the real one.
WS_BOARD="$HOME_FIX/.hermes/kanban/boards/proof"
WS_ROOT="$WS_BOARD/workspaces/t_done0001"
WS_DB="$WS_BOARD/kanban.db"
MISLEAD='unrelated plugin error: message catalog contains cannot mutate Kanban tasks, init failed'
reset_ws() {
  mkdir -p "$WS_ROOT"
  echo fixture > "$WS_ROOT/scratch.txt"
  sqlite3 "$WS_DB" "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, status TEXT);
    INSERT INTO tasks (id, status) VALUES ('t_done0001', 'done')
    ON CONFLICT(id) DO UPDATE SET status='done';"
}

# Isolate each case from the outer environment. The proof must pass whether or
# not the parent shell is itself a fenced child.
base_env() {
  env -u HERMES_DELEGATED_CHILD_CONTEXT -u STUB_STDERR -u STUB_BROKEN -u STUB_RC "$@"
}

# Source the real script, then replace log() so assertions see the message
# without the timestamp prefix the production logger adds.
drive() {
  base_env "$@" bash -c '
    set -uo pipefail
    export HOME="'"$HOME_FIX"'"
    export DISK_GUARD_HERMES_BIN="'"$BIN/hermes"'"
    export PATH="'"$BIN"':$PATH"
    DISK_GUARD_LIB=1 source "'"$GUARD"'"
    log() { echo "$*"; }
    reclaim_hermes_worktrees
  '
}

# Full --reclaim dispatch. Non-worktree classes are recorders so the fixture
# never sweeps caches, tmp, or node_modules. path_in_use is stubbed not-in-use
# so lsof/ps cannot veto a fixture dir. dead_task is the real function: a
# board kanban.db under the fixture HOME is what licenses workspace deletion.
dispatch() {
  base_env "$@" bash -c '
    set -uo pipefail
    export HOME="'"$HOME_FIX"'"
    export DISK_GUARD_HERMES_BIN="'"$BIN/hermes"'"
    export PATH="'"$BIN"':$PATH"
    DISK_GUARD_LIB=1 source "'"$GUARD"'"
    log() { echo "$*"; }
    path_in_use() { return 1; }
    reclaim_safe_caches() { echo "RECORDER reclaim_safe_caches"; }
    reclaim_tmp() { echo "RECORDER reclaim_tmp"; }
    reclaim_home_node_modules() { echo "RECORDER reclaim_home_node_modules"; }
    reclaim_pytest_roots() { echo "RECORDER reclaim_pytest_roots"; }
    reclaim_npm_cache() { echo "RECORDER reclaim_npm_cache"; }
    reclaim_stale_installs() { echo "RECORDER reclaim_stale_installs"; }
    reclaim_to_target() { echo "RECORDER reclaim_to_target"; }
    run_reclaim_classes
  '
}

SKIP_LINE="worktrees: SKIP fenced child context -- run disk-guard from an operator shell/launchd: delegate_task child contexts cannot mutate Kanban tasks via the CLI"
DEAD_SKIP="dead-card worktrees: SKIP fenced child context -- run disk-guard from an operator shell/launchd"

# 1. Fenced child, direct: exact SKIP quoting the real refusal. No upgrade, no FAIL.
out=$(drive HERMES_DELEGATED_CHILD_CONTEXT=1)
printf '%s\n' "$out" | grep -qx "$SKIP_LINE" \
  && ok "fenced child logs SKIP quoting the real refusal" \
  || { bad "fenced child did not log the exact SKIP line"; printf '    got: %s\n' "$out"; }
printf '%s\n' "$out" | grep -E '^worktrees: SKIP' | grep -F "cannot mutate Kanban tasks" >/dev/null \
  && ok "fenced child SKIP line contains the real refusal" \
  || { bad "fenced child SKIP line missing the real refusal"; printf '    got: %s\n' "$out"; }
printf '%s\n' "$out" | grep -E '^worktrees: SKIP' | grep -F "Command helper" >/dev/null \
  && bad "fenced child SKIP line quoted the Command helper preamble" \
  || ok "fenced child SKIP line does not quote Command helper"
printf '%s\n' "$out" | grep -q 'upgrade' \
  && bad "fenced child told the operator to upgrade hermes" \
  || ok "fenced child does not say upgrade"
printf '%s\n' "$out" | grep -q 'FAIL' \
  && bad "fenced child logged FAIL" \
  || ok "fenced child does not FAIL"

# 2. Clean operator shell, direct: success path unchanged.
out=$(drive)
printf '%s\n' "$out" | grep -F "worktrees:   -> 2 removed, 1 kept" >/dev/null \
  && ok "clean probe reclaims and logs the hermes summary" \
  || { bad "clean probe did not log the reclaim summary"; printf '    got: %s\n' "$out"; }

# 3. Broken probe, direct: FAIL carries rc and stderr, not a bare upgrade hint.
out=$(drive STUB_BROKEN=1)
want="worktrees: FAIL hermes reclaim probe rc=2 at $BIN/hermes: unrecognized arguments: --dry-run"
printf '%s\n' "$out" | grep -F "$want" >/dev/null \
  && ok "broken probe logs rc=2 and stderr" \
  || { bad "broken probe did not log rc=2 with stderr"; printf '    got: %s\n' "$out"; }
printf '%s\n' "$out" | grep -q 'upgrade' \
  && bad "broken probe said upgrade" \
  || ok "broken probe does not say upgrade"

# 4. Fenced but stderr is not the real refusal: FAIL, no SKIP, no deletion path.
out=$(drive HERMES_DELEGATED_CHILD_CONTEXT=1 STUB_STDERR='boom child context')
want="worktrees: FAIL hermes reclaim probe rc=1 at $BIN/hermes: boom child context"
printf '%s\n' "$out" | grep -F "$want" >/dev/null \
  && ok "fenced unrelated stderr logs FAIL with the line" \
  || { bad "fenced unrelated stderr did not FAIL with the line"; printf '    got: %s\n' "$out"; }
printf '%s\n' "$out" | grep -q 'SKIP' \
  && bad "fenced unrelated stderr was classified as SKIP" \
  || ok "fenced unrelated stderr is not a SKIP"

# 4b. Fenced + misleading substring, rc=2: FAIL quoting rc, not SKIP.
out=$(drive HERMES_DELEGATED_CHILD_CONTEXT=1 STUB_RC=2 STUB_STDERR="$MISLEAD")
want="worktrees: FAIL hermes reclaim probe rc=2 at $BIN/hermes: $MISLEAD"
printf '%s\n' "$out" | grep -F "$want" >/dev/null \
  && ok "fenced misleading substring rc=2 logs FAIL" \
  || { bad "fenced misleading substring rc=2 did not FAIL with the line"; printf '    got: %s\n' "$out"; }
printf '%s\n' "$out" | grep -E '^worktrees: SKIP' >/dev/null \
  && bad "fenced misleading substring rc=2 was classified as SKIP" \
  || ok "fenced misleading substring rc=2 is not a SKIP"

# 4c. Same misleading line at rc=1 is still not the exact refusal: FAIL, not SKIP.
out=$(drive HERMES_DELEGATED_CHILD_CONTEXT=1 STUB_RC=1 STUB_STDERR="$MISLEAD")
want="worktrees: FAIL hermes reclaim probe rc=1 at $BIN/hermes: $MISLEAD"
printf '%s\n' "$out" | grep -F "$want" >/dev/null \
  && ok "fenced misleading substring rc=1 logs FAIL" \
  || { bad "fenced misleading substring rc=1 did not FAIL with the line"; printf '    got: %s\n' "$out"; }
printf '%s\n' "$out" | grep -E '^worktrees: SKIP' >/dev/null \
  && bad "fenced misleading substring rc=1 was classified as SKIP" \
  || ok "fenced misleading substring rc=1 is not a SKIP"

# 4d. Real hermes refusal line (kanban: prefix, rc=1) is a SKIP. Preamble stays off the quote.
out=$(drive HERMES_DELEGATED_CHILD_CONTEXT=1 STUB_RC=1 STUB_STDERR='kanban: delegate_task child contexts cannot mutate Kanban tasks via the CLI')
want="worktrees: SKIP fenced child context -- run disk-guard from an operator shell/launchd: kanban: delegate_task child contexts cannot mutate Kanban tasks via the CLI"
printf '%s\n' "$out" | grep -qx "$want" \
  && ok "fenced exact kanban: refusal line is SKIP" \
  || { bad "fenced exact kanban: refusal line was not SKIP"; printf '    got: %s\n' "$out"; }

# --- full dispatch: one --reclaim sequence, real worktree on the fixture ---

# 5. Fenced + real refusal: nothing deleted, both SKIP lines, no upgrade, no FAIL.
reset_wt || { bad "could not create fixture worktree"; echo "=== RESULT: FAIL ==="; exit 1; }
reset_ws || { bad "could not create fixture workspace"; echo "=== RESULT: FAIL ==="; exit 1; }
out=$(dispatch HERMES_DELEGATED_CHILD_CONTEXT=1)
[ -d "$WT" ] && ok "fenced dispatch left the dead-card worktree in place" \
  || bad "fenced dispatch deleted the worktree"
[ -d "$WS_ROOT" ] && ok "fenced dispatch left the done-task workspace in place" \
  || bad "fenced dispatch deleted the done-task workspace"
printf '%s\n' "$out" | grep -F 'workspaces: SKIP' >/dev/null \
  && ok "fenced dispatch logs workspaces SKIP" \
  || { bad "fenced dispatch missing workspaces SKIP"; printf '    got: %s\n' "$out"; }
printf '%s\n' "$out" | grep -E '^worktrees: SKIP' | grep -F "cannot mutate Kanban tasks" >/dev/null \
  && ok "fenced dispatch logs worktrees SKIP with the real refusal" \
  || { bad "fenced dispatch missing worktrees SKIP refusal"; printf '    got: %s\n' "$out"; }
printf '%s\n' "$out" | grep -E '^worktrees: SKIP' | grep -F "Command helper" >/dev/null \
  && bad "fenced dispatch SKIP line quoted the Command helper preamble" \
  || ok "fenced dispatch SKIP line does not quote Command helper"
printf '%s\n' "$out" | grep -Fx "$DEAD_SKIP" >/dev/null \
  && ok "fenced dispatch logs dead-card SKIP" \
  || { bad "fenced dispatch missing dead-card SKIP"; printf '    got: %s\n' "$out"; }
printf '%s\n' "$out" | grep -q 'upgrade' \
  && bad "fenced dispatch said upgrade" \
  || ok "fenced dispatch does not say upgrade"
printf '%s\n' "$out" | grep -q 'FAIL' \
  && bad "fenced dispatch logged FAIL" \
  || ok "fenced dispatch does not FAIL"

# 6. Clean env: hermes summary logged, dead-card worktree and done workspace removed.
reset_wt || { bad "could not recreate fixture worktree"; echo "=== RESULT: FAIL ==="; exit 1; }
reset_ws || { bad "could not recreate fixture workspace"; echo "=== RESULT: FAIL ==="; exit 1; }
out=$(dispatch)
printf '%s\n' "$out" | grep -F -- "-> 2 removed, 1 kept" >/dev/null \
  && ok "clean dispatch logs the hermes summary" \
  || { bad "clean dispatch missing hermes summary"; printf '    got: %s\n' "$out"; }
[ ! -d "$WT" ] && ok "clean dispatch removed the dead-card worktree" \
  || bad "clean dispatch left the dead-card worktree in place"
[ ! -d "$WS_ROOT" ] && ok "clean dispatch removed the done-task workspace" \
  || bad "clean dispatch left the done-task workspace in place"

# 7. Fenced + unrelated stderr: FAIL, worktrees class does not SKIP, dir remains.
# workspaces/dead-card still emit their own SKIP lines — the env fence is
# independent of probe text, and those classes must not delete either.
reset_wt || { bad "could not recreate fixture worktree"; echo "=== RESULT: FAIL ==="; exit 1; }
out=$(dispatch HERMES_DELEGATED_CHILD_CONTEXT=1 STUB_STDERR='boom child context')
want="worktrees: FAIL hermes reclaim probe rc=1 at $BIN/hermes: boom child context"
printf '%s\n' "$out" | grep -F "$want" >/dev/null \
  && ok "fenced dispatch unrelated stderr logs FAIL" \
  || { bad "fenced dispatch unrelated stderr missing FAIL line"; printf '    got: %s\n' "$out"; }
# Anchor: "dead-card worktrees: SKIP" contains the substring but is a
# different class. The probe itself must not have been classified as SKIP.
printf '%s\n' "$out" | grep -E '^worktrees: SKIP' >/dev/null \
  && bad "fenced dispatch unrelated stderr classified worktrees as SKIP" \
  || ok "fenced dispatch unrelated stderr is not a worktrees SKIP"
[ -d "$WT" ] && ok "fenced dispatch unrelated stderr left the worktree in place" \
  || bad "fenced dispatch unrelated stderr deleted the worktree"

# 8. Not fenced, broken stub: FAIL with rc=2 and the stderr line.
reset_wt || { bad "could not recreate fixture worktree"; echo "=== RESULT: FAIL ==="; exit 1; }
out=$(dispatch STUB_BROKEN=1)
want="worktrees: FAIL hermes reclaim probe rc=2 at $BIN/hermes: unrecognized arguments: --dry-run"
printf '%s\n' "$out" | grep -F "$want" >/dev/null \
  && ok "broken dispatch logs rc=2 and stderr" \
  || { bad "broken dispatch missing rc=2 FAIL line"; printf '    got: %s\n' "$out"; }

# 9. Fenced dispatch + misleading substring, rc=2: FAIL, not worktrees SKIP, no deletion.
reset_wt || { bad "could not recreate fixture worktree"; echo "=== RESULT: FAIL ==="; exit 1; }
reset_ws || { bad "could not recreate fixture workspace"; echo "=== RESULT: FAIL ==="; exit 1; }
out=$(dispatch HERMES_DELEGATED_CHILD_CONTEXT=1 STUB_RC=2 STUB_STDERR="$MISLEAD")
want="worktrees: FAIL hermes reclaim probe rc=2 at $BIN/hermes: $MISLEAD"
printf '%s\n' "$out" | grep -F "$want" >/dev/null \
  && ok "fenced dispatch misleading substring rc=2 logs FAIL" \
  || { bad "fenced dispatch misleading substring rc=2 missing FAIL line"; printf '    got: %s\n' "$out"; }
printf '%s\n' "$out" | grep -E '^worktrees: SKIP' >/dev/null \
  && bad "fenced dispatch misleading substring rc=2 classified worktrees as SKIP" \
  || ok "fenced dispatch misleading substring rc=2 is not a worktrees SKIP"
[ -d "$WT" ] && ok "fenced dispatch misleading substring rc=2 left the worktree in place" \
  || bad "fenced dispatch misleading substring rc=2 deleted the worktree"
[ -d "$WS_ROOT" ] && ok "fenced dispatch misleading substring rc=2 left the workspace in place" \
  || bad "fenced dispatch misleading substring rc=2 deleted the workspace"

# 10. Fenced dispatch + same misleading line, rc=1: FAIL, not worktrees SKIP, no deletion.
reset_wt || { bad "could not recreate fixture worktree"; echo "=== RESULT: FAIL ==="; exit 1; }
out=$(dispatch HERMES_DELEGATED_CHILD_CONTEXT=1 STUB_RC=1 STUB_STDERR="$MISLEAD")
want="worktrees: FAIL hermes reclaim probe rc=1 at $BIN/hermes: $MISLEAD"
printf '%s\n' "$out" | grep -F "$want" >/dev/null \
  && ok "fenced dispatch misleading substring rc=1 logs FAIL" \
  || { bad "fenced dispatch misleading substring rc=1 missing FAIL line"; printf '    got: %s\n' "$out"; }
printf '%s\n' "$out" | grep -E '^worktrees: SKIP' >/dev/null \
  && bad "fenced dispatch misleading substring rc=1 classified worktrees as SKIP" \
  || ok "fenced dispatch misleading substring rc=1 is not a worktrees SKIP"
[ -d "$WT" ] && ok "fenced dispatch misleading substring rc=1 left the worktree in place" \
  || bad "fenced dispatch misleading substring rc=1 deleted the worktree"

rm -rf "$FIX"

[ "$FAIL" = 0 ] && { echo "=== RESULT: PASS ==="; exit 0; }
echo "=== RESULT: FAIL ==="
exit 1
