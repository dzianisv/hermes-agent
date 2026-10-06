#!/usr/bin/env bash
# disk-guard.sh — fail loudly below a free-space floor, and reclaim space from
# agent scratch that is provably dead (terminal kanban tasks, merged worktrees,
# stale /private/tmp clones).
#
# CLASS this fixes: an ENOSPC host makes every watchdog unreliable while it
# still reports green. Each concurrent agent worker materialises a du-apparent
# clone (kanban scratch workspace, /private/tmp clone, or git worktree) and
# nothing reaps them when the task ends.
#
# Everything checked here is DERIVED from the source of truth at runtime:
#   - live/dead task set  -> sqlite query against ~/.hermes/kanban.db
#   - merged branches     -> git merge-base --is-ancestor against origin/main
#   - in-use directories  -> running process table
# There is no hand-maintained list of paths to keep in sync.
#
# Usage:
#   disk-guard.sh            # check only; exit 1 if below floor
#   disk-guard.sh --reclaim  # check, reclaim dead scratch, re-check
#
# Exit codes: 0 = at or above floor, 1 = below floor after reclaim.

set -uo pipefail

# TWO NUMBERS, TWO JOBS. Conflating them is what produced both failure modes
# this guard has actually shown: a 10Gi floor paged 67 times in 13h with no
# failed write (alert fatigue), and a 0.5Gi floor let the host cross from
# healthy to ENOSPC inside one 900s tick (no warning at all).
#
#   RECLAIM_TARGET_GI  how much free space we try to HOLD. Enforced by evicting
#                      more sanctioned scratch, SILENTLY. Never pages.
#   FLOOR_GI           the only thing that pages. Owner-pinned at 0.5Gi
#                      (Den, 2026-09-22) — page only where writes actually fail.
#
# The target was 10Gi (the capacity derivation in disk-guard-cron.sh: 2.0GB
# worst case for 3 concurrent workers on a cold pnpm store + 8GB headroom for
# the non-worker writers on this volume — state.db, postgres, gateway logs).
# RAISED TO 25Gi on 2026-09-25 (card t_2a74e97a, review round 1): the card
# contracts >=25GB free, and a target of 10 never fires at 23Gi, so nothing
# defended the contracted number — free space hit 25.17Gi once and decayed
# straight back to 23.1Gi. The target is what holds a number; a one-shot sweep
# is not. 25Gi (26.8GB) also clears the 25GB decimal contract with margin.
# Raising it is safe by construction: the target only evicts sanctioned scratch
# and is SILENT — it can never page, so a high target cannot cause alert fatigue.
FLOOR_GI="${DISK_GUARD_FLOOR_GI:-0.5}"
RECLAIM_TARGET_GI="${DISK_GUARD_RECLAIM_TARGET_GI:-25}"
KANBAN_DB="${HERMES_KANBAN_DB:-$HOME/.hermes/kanban.db}"
WORKSPACES="$HOME/.hermes/kanban/workspaces"
TMPDIRS="/private/tmp"
TMP_AGE_SECONDS="${DISK_GUARD_TMP_AGE:-7200}"
RECLAIM=0
[ "${1:-}" = "--reclaim" ] && RECLAIM=1

log() { printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*"; }

# DISK_GUARD_FAKE_FREE_GI is a FIXTURE: it lets a harness drive the below-target
# and below-floor reporting without filling the volume. It is never allowed to
# drive a deletion (see the --reclaim fence at the bottom of this script).
free_mb() {
  if [ -n "${DISK_GUARD_FAKE_FREE_GI:-}" ]; then
    awk -v g="$DISK_GUARD_FAKE_FREE_GI" 'BEGIN{printf "%d\n", g*1024+0.5}'
    return 0
  fi
  df -m /System/Volumes/Data 2>/dev/null | awk 'NR==2{print $4}'
}
# FLOAT GiB. An integer `free_mb/1024` floored 1708MiB to "1Gi" and made the
# 0.5 comparison a lie at the only magnitude that matters.
free_gi() {
  if [ -n "${DISK_GUARD_FAKE_FREE_GI:-}" ]; then
    awk -v g="$DISK_GUARD_FAKE_FREE_GI" 'BEGIN{printf "%.2f", g}'
    return 0
  fi
  awk -v m="$(free_mb)" 'BEGIN{printf "%.2f", m/1024}'
}
# Float-exact comparison: `[ 0.60 -lt 0.5 ]` is a bash INTEGER error, which
# silently skips the branch and reports green on a dying host.
lt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0 < b+0)}'; }

# Bounded probe: dg_timeout <seconds> cmd... . A diagnostic probe that hangs
# would hang the guard itself, so every swap probe goes through this. launchd's
# PATH may not carry Homebrew's (g)timeout; perl's alarm survives exec and is
# always present on macOS, so the bound holds either way. With neither, the
# probe is SKIPPED (non-zero), never run unbounded: a missing measurement
# degrades to "unknown"/"unreadable", a hung one would wedge every tick.
dg_timeout() {
  local s=$1 t; shift
  t=$(command -v gtimeout || command -v timeout)
  if [ -n "$t" ]; then "$t" "$s" "$@"
  elif command -v perl >/dev/null 2>&1; then perl -e 'alarm shift; exec @ARGV' "$s" "$@"
  else return 125
  fi
}

# --- measured term: macOS swap. MEASURED 2026-10 (card t_515b8493): ~30GiB of
# /System/Volumes/VM/swapfile* (1GiB each) shares the APFS container with the
# data volume, so free space decayed while every scratch class reported
# nothing to reclaim. Swap is a MEASUREMENT here, never a reclaim class: the
# guard reports it so the shortfall is explained, and it can never page or
# change the exit code. Every probe is time-bounded and degrades silently.
swap_used_gi() { # prints numeric Gi (one decimal) or nothing
  if [ -n "${DISK_GUARD_FAKE_SWAP_GI:-}" ]; then
    printf '%s' "$DISK_GUARD_FAKE_SWAP_GI"; return 0
  fi
  # One measurement per tick: the main body memoizes it so the shortfall
  # clause does not pay a second (possibly hung) sysctl bound.
  if [ "${DG_SWAP_USED_MEMO+set}" = set ]; then
    printf '%s' "$DG_SWAP_USED_MEMO"; return 0
  fi
  local raw
  raw=$(dg_timeout 5 sysctl -n vm.swapusage 2>/dev/null) || return 0
  printf '%s\n' "$raw" | awk '{
    for (i = 1; i < NF; i++) if ($i == "used" && $(i+1) == "=") {
      v = $(i+2); u = substr(v, length(v)); n = v + 0
      if (v !~ /^[0-9.]+[KMG]$/) exit
      g = (u == "G") ? n : (u == "M") ? n/1024 : n/1048576
      printf "%.1f", g; exit
    }
  }'
  return 0
}

swap_files_term() { # "swapfiles=NxQ.QGi swapfiles_sum=S.SGi newest=<iso>" or "swapfiles=unreadable"
  # Q is the AVERAGE (sum/N), not the max: N x max overstated mixed-size swap.
  # The glob is expanded by the shell (no I/O per file); every size/mtime
  # comes from ONE stat call under ONE wall-clock bound.
  local dir="${DISK_GUARD_SWAP_DIR:-/System/Volumes/VM}" f out
  local files=()
  for f in "$dir"/swapfile*; do
    case "${f##*/}" in swapfile|swapfile\*|*.*) continue ;; esac
    files+=("$f")
  done
  if [ "${#files[@]}" -eq 0 ]; then printf 'swapfiles=unreadable'; return 0; fi
  # rc 1 = some file vanished mid-probe (rest still valid); >=124 = timed out,
  # killed or skipped (no bound available) -> nothing from it is trusted.
  out=$(dg_timeout 5 stat -f '%z %m' "${files[@]}" 2>/dev/null)
  [ $? -ge 124 ] && { printf 'swapfiles=unreadable'; return 0; }
  out=$(printf '%s\n' "$out" | awk '
    NF == 2 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ {
      n++; sum += $1; if ($2 > newest) newest = $2
    }
    END { if (n) printf "%d %.1f %.1f %d", n, sum/n/1073741824, sum/1073741824, newest }')
  if [ -z "$out" ]; then printf 'swapfiles=unreadable'; return 0; fi
  set -- $out
  printf 'swapfiles=%sx%sGi swapfiles_sum=%sGi' "$1" "$2" "$3"
  out=$(date -r "$4" '+%Y-%m-%dT%H:%M:%S%z' 2>/dev/null) && printf ' newest=%s' "$out"
  return 0
}

compressor_top_term() { # "compressor_top=a:X.XG,b:Y.YG" or nothing
  [ "${DISK_GUARD_SKIP_TOP:-0}" = 1 ] && return 0
  local out
  out=$(dg_timeout 5 top -l1 -stats cmprs,command -o cmprs -n 2 2>/dev/null) || return 0
  printf '%s\n' "$out" | awk '
    /^CMPRS[[:space:]]+COMMAND/ { h = 1; next }
    h && NF >= 2 && c < 2 {
      v = $1; sub(/[+-]$/, "", v); u = substr(v, length(v)); n = v + 0
      if (v ~ /^[0-9.]+G$/) g = n
      else if (v ~ /^[0-9.]+M$/) g = n / 1024
      else if (v ~ /^[0-9.]+K$/) g = n / 1048576
      else if (v ~ /^[0-9.]+B?$/) g = n / 1073741824
      else { bad = 1; exit }
      $1 = ""; sub(/^[ \t]+/, ""); sub(/[ \t]+$/, ""); gsub(/[ \t]+/, "_")
      s = s (c ? "," : "") $0 ":" sprintf("%.1f", g) "G"; c++
    }
    END { if (!bad && c > 0) printf "compressor_top=%s", s }'
  return 0
}

swap_term() { # one line of space-separated fields; always exit 0
  local u t line
  u=$(swap_used_gi)
  if [ -n "$u" ]; then line="swap_used=${u}Gi"; else line="swap_used=unknown"; fi
  line="$line $(swap_files_term)"
  t=$(compressor_top_term)
  [ -n "$t" ] && line="$line $t"
  printf '%s\n' "$line"
  return 0
}

# Shortfall clause shared by every target-shortfall line. $1 = free Gi now.
# Swap is reported twice, never clipped: the MEASURED figure (Z) as-is, and the
# part of the shortfall it accounts for (Y = min(Z, X)). Clipping Z to X hid
# how big swap actually was whenever it exceeded the gap.
swap_shortfall_clause() {
  local x z y
  x=$(awk -v t="$RECLAIM_TARGET_GI" -v n="$1" 'BEGIN{d=t-n; if(d<0)d=0; printf "%.1f", d}')
  z=$(swap_used_gi)
  if printf '%s' "$z" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
    y=$(awk -v z="$z" -v x="$x" 'BEGIN{y=z+0; if(y>x+0)y=x+0; printf "%.1f", y}')Gi
    z=$(awk -v z="$z" 'BEGIN{printf "%.1f", z}')Gi
  else
    y=unknown; z=unknown
  fi
  printf 'shortfall to target %sGi, of which swap accounts for %s (swap_used=%s measured; not reclaimable by this guard — needs memory pressure relief or reboot)' "$x" "$y" "$z"
}

# Check-only runs never escalate, so they never logged WHY free sat under the
# target. Report it (silent, informational) — never part of the paging branch.
report_check_only_shortfall() { # $1 = free Gi after
  [ "$RECLAIM" = 0 ] || return 0
  lt "$1" "$RECLAIM_TARGET_GI" || return 0
  log "below_target: ${1}Gi < ${RECLAIM_TARGET_GI}Gi (check-only); $(swap_shortfall_clause "$1")"
}

# --- derived fact 1: which task ids are dead (terminal or absent from the DB)
# Every kanban DB on the host, default board first. A card id is only a name;
# nothing ties it to the board whose directory it happens to sit under.
kanban_dbs() {
  [ -f "$KANBAN_DB" ] && printf '%s\n' "$KANBAN_DB"
  local d
  for d in "$HOME"/.hermes/kanban/boards/*/kanban.db; do
    [ -f "$d" ] || continue
    [ "$d" = "$KANBAN_DB" ] || printf '%s\n' "$d"
  done
}

dead_task() { # $1 = task id -> 0 if dead/unknown on EVERY board
  # SINGLE-BOARD BUG (measured 2026-09-25, card t_2a74e97a): this consulted
  # only $KANBAN_DB. A live run of reclaim_dead_task_worktrees then deleted
  # the worktrees of t_41e448bd and t_fb0730ea -- both `blocked`, i.e. very
  # much alive -- because they live on the `vibebrowser` and
  # `xsense-telegram-worker` boards, which the default DB has never heard of.
  # Absence from one DB is not evidence of death; it is evidence of looking in
  # one place. Ask every board and let ANY non-terminal status veto.
  local db st known=0 seen=0
  while IFS= read -r db; do
    seen=1
    st=$(sqlite3 -noheader "$db" \
          "select status from tasks where id='$1';" 2>/dev/null)
    [ -n "$st" ] && known=1
    case "$st" in
      ''|done|archived|cancelled) ;;
      *) return 1 ;;                 # alive somewhere: hands off
    esac
  done < <(kanban_dbs)
  [ "$known" = 1 ] && return 0
  # Unknown to every board: a card id whose row has been pruned. Treated as
  # dead on purpose -- that is the steady-state source of abandoned scratch --
  # but ONLY if at least one board was actually readable. If we could not
  # consult any DB, we have no evidence of anything and must not authorise a
  # deletion; losing sight of the boards has to fail safe, not fail open.
  [ "$seen" = 1 ] || return 1
  return 0
}

# --- derived fact 2: is any running process sitting in this path
# Deliberately derives from lsof's OUTPUT, not its exit code: on macOS
# `lsof +D <dir>` exits 1 even when it prints a matching holder (measured
# 2026-09-21: a `sleep` whose cwd was the dir was listed, rc=1). Trusting rc
# made the guard offer a live worker's directory for deletion. The argv check
# alone is also insufficient — a process that cd'd into the dir has a bare
# argv ("sleep 900") that never mentions the path.
path_in_use() { # $1 = abs path
  # SELF-MATCH BUG (found 2026-09-25): this used
  #   ps -Ao args= | grep -Fq -- "$1"
  # and `grep`'s OWN argv contains "$1", so ps listed it and the grep matched
  # itself. path_in_use therefore returned "in use" for EVERY path ever passed:
  # every reclaim class that consults it deleted nothing, and top_reclaimable
  # filtered out every candidate — which is exactly the "No single reclaimable
  # path over 100MB" line the operator kept reading on a host that was filling
  # up. Passing the needle through the ENVIRONMENT keeps it out of argv, so the
  # scan can no longer see itself.
  # (Exported, not a command-prefix assignment: a prefix binds only to the
  # first command of the pipeline -- `ps` -- and awk would see an empty needle,
  # matching every line and reinstating the same always-in-use bug.)
  # SYMLINKED-PREFIX BUG (found 2026-09-25): the caller's path and the live
  # process's argv routinely spell the SAME file differently. On macOS $TMPDIR
  # is /var/folders/... while /var is a symlink to /private/var, and `git
  # worktree list` reports the resolved form; a worker launched with the
  # unresolved form then goes undetected and its directory gets deleted out
  # from under it. Compare BOTH spellings, not just the one we were handed.
  local p1="$1" p2 p3 dg_n
  p2=$(cd "$(dirname -- "$1")" 2>/dev/null && printf '%s/%s' "$(pwd -P)" "$(basename -- "$1")") || p2=""
  [ -n "$p2" ] && [ "$p2" != "$p1" ] || p2=""
  # ...and the UNRESOLVED spelling, which is the direction that actually bit:
  # git reports /private/var/... while the worker's argv says /var/..., so
  # resolving alone never converges. Strip the macOS /private prefix too.
  p3=""
  case "$p1" in /private/*) p3="${p1#/private}" ;; esac
  case "$p2" in /private/*) [ -z "$p3" ] && p3="${p2#/private}" ;; esac
  # dg_n, not n: this function is called from loops that keep their own
  # counter, and an unlocalised `n` here silently clobbered the caller's.
  for dg_n in "$p1" $p2 $p3; do
    export DG_NEEDLE="$dg_n"
    ps -Ao args= 2>/dev/null \
      | awk 'index($0, ENVIRON["DG_NEEDLE"]) { found = 1 } END { exit !found }' \
      && { unset DG_NEEDLE; return 0; }
  done
  unset DG_NEEDLE
  [ -n "$(lsof +D "$1" 2>/dev/null | tail -n +2)" ] && return 0
  [ -n "$(lsof -- "$1" 2>/dev/null | tail -n +2)" ] && return 0
  return 1
}

# --- derived fact 3: the scratch roots worth ranking.
# DERIVED from the live filesystem, never a hand-kept list: every git repo's
# worktree parent, the kanban workspaces, the agent caches and the hermes log/
# session dirs. The 2026-09-21 alert ranked /private/tmp files of one du block
# each and would have sent an operator to delete 30 bytes — the roots were
# wrong and the sort was by du blocks.
RANK_MIN_MB="${DISK_GUARD_RANK_MIN_MB:-100}"

scratch_roots() {
  {
    echo "$WORKSPACES"
    echo "$HOME/workspace/AgentPod-worktrees"
    ls -d "$HOME"/workspace/*/.worktrees 2>/dev/null
    echo "$HOME/Library/pnpm/store"
    echo "$HOME/.cache"
    echo "$HOME/.copilot/session-state"
    echo "$HOME/.paperclip/scratch"
    echo "$HOME/.paperclip/cli/installs/git"
    echo "$HOME/.hermes/logs"
    echo "$HOME/.hermes/sessions"
  } | while read -r r; do [ -d "$r" ] && echo "$r"; done
}

# --- derived fact 3b: paths the reclaim classes are FORBIDDEN to delete.
#
# MEASURED 2026-10-05 (card t_4b6e5ab2): the RED page's "Top reclaimable,
# largest first" named exactly two paths —
#   2948MB  ~/.cache/huggingface        (guard_hf_cache: STT model cache, the
#                                        2026-09-29 voice outage; must stay local)
#   1854MB  ~/Library/pnpm/store/v10    (reclaim_stale_pnpm_stores: logged
#                                        "skip (referenced)" on every tick)
# Both are paths the guard's OWN reclaim classes refuse to touch. The operator
# acting on the page deleted them, taking down the local STT model and forcing
# a reinstall on three projects still linked against v10. The ranking and the
# reclaiming disagreed because top_reclaimable only ever consulted liveness
# (path_in_use), never the keep-predicates.
#
# CLASS FIX: an advisory list that recommends what the automation is forbidden
# to do is worse than no list. The keep-predicates are DERIVED here from the
# same runtime facts the reclaim classes use — the HF cache path that
# guard_hf_cache protects, and the pnpm generations that
# reclaim_stale_pnpm_stores keeps (current store + any generation an existing
# .modules.yaml references). No hand-kept path list: add a protected class to
# its reclaim function and it must be reflected here too.
protected_paths() {
  printf '%s\n' "$HOME/.cache/huggingface"
  local root current
  root="$HOME/Library/pnpm/store"
  if [ -d "$root" ] && command -v pnpm >/dev/null 2>&1; then
    current=$(pnpm store path 2>/dev/null)
    [ -n "$current" ] && printf '%s\n' "$current"
    for r in "$HOME/workspace" "$HOME/.hermes/kanban/workspaces" \
             "$HOME"/.hermes/kanban/boards/*/workspaces; do
      [ -d "$r" ] || continue
      find "$r" -maxdepth 5 -name .modules.yaml -not -path '*/node_modules/*/node_modules/*' \
           -exec grep -ho "$root/v[0-9]*" {} + 2>/dev/null
    done
  fi
}

is_protected() { # $1 = abs path; 0 when a reclaim class is forbidden to delete it
  local p
  while read -r p; do
    [ -n "$p" ] || continue
    case "$1" in "$p"|"$p"/*) return 0 ;; esac
  done <<EOF
$(protected_paths)
EOF
  return 1
}

# Rank first-level entries of every scratch root by APPARENT SIZE, descending,
# dropping anything below the floor of interest, anything a live process is
# sitting in, and anything the reclaim classes are forbidden to delete.
# Output: "<MB>\t<path>" lines, largest first.
top_reclaimable() {
  local n="${1:-8}"
  scratch_roots | while read -r root; do
    du -sxm "$root"/* 2>/dev/null
  done | sort -rn | awk -v min="$RANK_MIN_MB" '$1 >= min' | \
  while read -r mb path; do
    path_in_use "$path" && continue
    is_protected "$path" && continue
    printf '%s\t%s\n' "$mb" "$path"
  done | head -"$n"
}

if [ "${1:-}" = "--top-reclaimable" ]; then
  top_reclaimable "${2:-8}"
  exit 0
fi

# The threshold is ONE decision with ONE owner. The cron wrapper renders a "no
# path over NMB" message and used to carry its own `:-100` default literal; the
# two could drift and the operator would read a threshold the ranking never
# applied. Wrapper asks, guard answers.
if [ "${1:-}" = "--rank-min-mb" ]; then
  printf '%s\n' "$RANK_MIN_MB"
  exit 0
fi

# Delegate children are refused by the Kanban write fence. Skipping only the
# hermes probe left the other deletion classes in the same --reclaim run free
# to remove scratch the child is not allowed to touch. Do NOT unset
# HERMES_DELEGATED_CHILD_CONTEXT — clearing it would bypass the trust boundary.
fenced_child() { # 0 when this process is a delegate_task child
  [ -n "${HERMES_DELEGATED_CHILD_CONTEXT:-}" ]
}

reclaim_workspaces() {
  if fenced_child; then
    log "workspaces: SKIP fenced child context -- run disk-guard from an operator shell/launchd"
    return 0
  fi
  local freed=0 n=0 id sz root db
  # WORKSPACE ROOTS ARE DISCOVERED, NOT PINNED. Until 2026-09-25 this swept
  # only $HOME/.hermes/kanban/workspaces, i.e. the DEFAULT board. Every named
  # board keeps its own kanban/boards/<slug>/workspaces tree with its own
  # kanban.db, and none of them were ever reclaimed: boards/vibebrowser held
  # 1.8GB of dead-card scratch while the guard logged "removed 0". A pinned
  # list has the same rot as a pinned hash — a board created tomorrow would
  # leak silently — so enumerate the roots and pair each with ITS OWN db,
  # because a card id only has a status in the board that owns it (looking it
  # up in the wrong db returns '' = "unknown" = dead, which would delete a
  # RUNNING card's workspace).
  for root in "$WORKSPACES" "$HOME"/.hermes/kanban/boards/*/workspaces; do
    [ -d "$root" ] || continue
    db="$KANBAN_DB"
    case "$root" in
      "$HOME"/.hermes/kanban/boards/*)
        db="$(dirname "$root")/kanban.db"
        [ -f "$db" ] || continue ;;   # no db => cannot prove a card is dead
    esac
    for w in "$root"/*/; do
      [ -d "$w" ] || continue
      id=$(basename "$w")
      KANBAN_DB="$db" dead_task "$id" || continue
      path_in_use "${w%/}" && { log "  skip (in use) $w"; continue; }
      sz=$(du -sxm "$w" 2>/dev/null | cut -f1)
      rm -rf "$w" && { freed=$((freed + ${sz:-0})); n=$((n + 1)); }
    done
  done
  log "workspaces: removed $n dead scratch dirs, ${freed}MB"
}

reclaim_merged_worktrees() {
  # Superseded by `hermes kanban reclaim`. Kept as a fallback ONLY for when the
  # hermes venv is unavailable. Its eligibility test (ancestor-of-origin/main)
  # is wrong for a squash-merging repo: a squash-merged branch tip is never an
  # ancestor of main, so this loop removed 0 worktrees over its entire log while
  # reporting success. Do not extend it; extend hermes_cli/kanban_reclaim.py.
  local repo="$1" removed=0 freed=0 p b sha sz id
  [ -d "$repo/.git" ] || [ -f "$repo/.git" ] || return 0
  git -C "$repo" fetch origin main -q 2>/dev/null
  git -C "$repo" worktree prune 2>/dev/null
  git -C "$repo" worktree list --porcelain 2>/dev/null \
    | awk '/^worktree /{w=$2} /^branch /{print w" "$2}' \
    | while read -r p b; do
        [ "$p" = "$repo" ] && continue
        [ -d "$p" ] || continue
        sha=$(git -C "$repo" rev-parse "$b" 2>/dev/null) || continue
        # only reclaim branches fully contained in origin/main
        git -C "$repo" merge-base --is-ancestor "$sha" origin/main 2>/dev/null || continue
        # a worktree named after a live task is off limits
        id=$(basename "$p" | grep -oE 't_[0-9a-f]{8}' | head -1)
        if [ -n "$id" ] && ! dead_task "$id"; then
          log "  skip (live task $id) $p"; continue
        fi
        path_in_use "$p" && { log "  skip (in use) $p"; continue; }
        sz=$(du -sxm "$p" 2>/dev/null | cut -f1)
        git -C "$repo" worktree remove --force "$p" 2>/dev/null || rm -rf "$p"
        log "  removed merged worktree ${sz}MB $p"
      done
  git -C "$repo" worktree prune 2>/dev/null
}

reclaim_tmp() {
  local cutoff freed=0 n=0 m sz e
  cutoff=$(( $(date +%s) - TMP_AGE_SECONDS ))
  for e in "$TMPDIRS"/*; do
    [ -e "$e" ] || continue
    case "$(basename "$e")" in com.apple.*|*.sock) continue ;; esac
    m=$(stat -f %m "$e" 2>/dev/null) || continue
    [ "$m" -lt "$cutoff" ] || continue
    path_in_use "$e" && continue
    sz=$(du -sxm "$e" 2>/dev/null | cut -f1)
    rm -rf "$e" 2>/dev/null && { freed=$((freed + ${sz:-0})); n=$((n + 1)); }
  done
  log "tmp: removed $n stale entries, ${freed}MB"
}

# --- safe classes: caches and agent session-state that regenerate on demand.
# Each entry is DERIVED (a glob/age query against the live filesystem), and
# nothing here is a source of truth for any running job: caches refill, a
# copilot session older than the retention window is never resumed, and the
# pnpm store is rebuilt from the lockfile. Anything whose loss would cost a
# human decision (VM images, repos, databases) is deliberately NOT here.
SAFE_AGE_DAYS="${DISK_GUARD_SAFE_AGE_DAYS:-7}"

# --- STT model cache must stay on the boot volume. On 2026-09-29 ~/.cache/huggingface
# was relocated as a symlink to /Volumes/mac-offload; once that volume unmounted,
# every gateway voice transcription failed (EACCES). Never offload it; if it is a
# symlink whose target is missing, replace it with a real dir so the model refetches.
guard_hf_cache() {
  local hf="$HOME/.cache/huggingface"
  if [ -L "$hf" ] && [ ! -d "$hf/" ]; then
    rm -f "$hf" && mkdir -p "$hf/hub"
    log "  REPAIRED dangling symlink $hf (STT model cache must stay local)"
  fi
}

# --- superseded pnpm store GENERATIONS. MEASURED 2026-10-04: ~/Library/pnpm/store
# held v3 (826MB) + v10 (1753MB) + v11 (3486MB) = 5.3GB on a host that had 0.2Gi
# free, and these three paths were the top of the guard's own alert. No class
# reclaimed them: `pnpm store prune` (called by reclaim_safe_caches) only ever
# prunes the ONE generation the installed pnpm owns, and the store lives under
# ~/Library, outside every node_modules / cache / workspace root the guard sweeps.
# A store generation is pure content-addressable cache — refetchable from the
# registry — but deleting a generation a project is still linked against forces a
# reinstall, so liveness is DETECTED, never pinned:
#   - keep the generation `pnpm store path` resolves to (what installs use now)
#   - keep any generation referenced by an existing node_modules/.modules.yaml
#   - keep anything a live process holds
# Everything else is an abandoned generation left behind by a pnpm major upgrade.
reclaim_stale_pnpm_stores() {
  local root before after n=0 current d refs
  root="$HOME/Library/pnpm/store"
  [ -d "$root" ] || return 0
  command -v pnpm >/dev/null 2>&1 || return 0
  current=$(pnpm store path 2>/dev/null)
  [ -n "$current" ] || { log "pnpm stores: SKIP (cannot resolve current store path)"; return 0; }
  refs=$(for r in "$HOME/workspace" "$HOME/.hermes/kanban/workspaces" \
                 "$HOME"/.hermes/kanban/boards/*/workspaces; do
           [ -d "$r" ] || continue
           find "$r" -maxdepth 5 -name .modules.yaml -not -path '*/node_modules/*/node_modules/*' \
                -exec grep -ho "$root/v[0-9]*" {} + 2>/dev/null
         done | sort -u)
  before=$(free_mb)
  for d in "$root"/v*; do
    [ -d "$d" ] || continue
    case "$current" in "$d"|"$d"/*) continue ;; esac
    printf '%s\n' "$refs" | grep -qx "$d" && { log "  skip (referenced) $d"; continue; }
    path_in_use "$d" && { log "  skip (in use) $d"; continue; }
    rm -rf "$d" 2>/dev/null && n=$((n + 1))
  done
  after=$(free_mb)
  log "pnpm stores: removed $n superseded generations, $(( after - before ))MB"
}

reclaim_safe_caches() {
  local before after n=0 d
  guard_hf_cache
  before=$(free_mb)

  for d in "$HOME/.cache/uv" "$HOME/Library/Caches/pip" \
           "$HOME/Library/Caches/ms-playwright" "$HOME/Library/Caches/Homebrew"; do
    [ -d "$d" ] || continue
    path_in_use "$d" && { log "  skip (in use) $d"; continue; }
    rm -rf "$d"/* 2>/dev/null && n=$((n + 1))
  done

  # copilot session-state older than the retention window
  if [ -d "$HOME/.copilot/session-state" ]; then
    find "$HOME/.copilot/session-state" -mindepth 1 -maxdepth 1 \
         -mtime "+${SAFE_AGE_DAYS}" -exec rm -rf {} + 2>/dev/null
  fi

  # hermes worker logs beyond rotation (rotated copies only, never the live file)
  find "$HOME/.hermes/logs" "$HOME/.hermes/profiles" -type f \
       \( -name '*.log.[0-9]*' -o -name '*.log.gz' \) \
       -mtime "+${SAFE_AGE_DAYS}" -delete 2>/dev/null

  if command -v pnpm >/dev/null 2>&1; then
    pnpm store prune >/dev/null 2>&1
  fi

  after=$(free_mb)
  log "safe caches: pruned $n cache dirs + aged session-state/logs, $(( after - before ))MB"
}

# --- npm's content-addressable cache. MEASURED 2026-09-25: ~/.npm held 2.3GB
# (1.8GB _cacache + 495MB _npx) on a host contracted to hold 25GB free. It is
# pure regrowable cache — every entry is re-fetchable from the registry — but
# no class reclaimed it: reclaim_safe_caches enumerates a fixed list that never
# included it, and ~/.npm is a dotdir so the node_modules sweep skips it.
# `npm cache clean --force` is npm's own supported eviction, so we do not hand
# rm the cache layout out from under it; if npm is absent there is nothing to do.
reclaim_npm_cache() {
  local before after
  command -v npm >/dev/null 2>&1 || return 0
  [ -d "$HOME/.npm" ] || return 0
  path_in_use "$HOME/.npm/_cacache" && { log "  skip (in use) ~/.npm/_cacache"; return 0; }
  before=$(free_mb)
  npm cache clean --force >/dev/null 2>&1
  # _npx is a scratch tree of fully installed throwaway packages, not part of
  # the cache npm prunes. It is regenerated on the next `npx` invocation.
  find "$HOME/.npm/_npx" -mindepth 1 -maxdepth 1 -mtime +0 \
       -exec rm -rf {} + 2>/dev/null
  after=$(free_mb)
  log "npm cache: $(( after - before ))MB"
}

# --- superseded Hermes install generations. MEASURED 2026-09-25: 2.9GB across
# three generations under ~/.hermes/installs while exactly one is live.
#
# LIVENESS IS DETECTED, NEVER PINNED. An allowlist of hashes rots the moment
# hermes reinstalls, and would eventually name the live generation as garbage —
# deleting the runtime out from under every agent on the box. A generation is
# kept if ANY of:
#   - a live process has its path in argv (path_in_use), i.e. it is running now
#   - it is the newest generation (the one a fresh spawn will resolve to)
# Everything else is a superseded copy that reinstall can recreate.
reclaim_stale_installs() {
  local root before after n=0 newest d
  root="$HOME/.hermes/installs"
  [ -d "$root" ] || return 0
  newest=$(ls -dt "$root"/*/ 2>/dev/null | head -1)
  [ -n "$newest" ] || return 0
  before=$(free_mb); n=0
  for d in "$root"/*/; do
    [ -d "$d" ] || continue
    [ "$d" = "$newest" ] && continue
    path_in_use "${d%/}" && { log "  skip (live install) ${d%/}"; continue; }
    rm -rf "$d" 2>/dev/null && n=$((n + 1))
  done
  after=$(free_mb)
  log "installs: removed $n superseded generations, $(( after - before ))MB"
}

# --- worktrees named after a DEAD card, anywhere on the volume.
# MEASURED 2026-09-25: ~/.hermes/hermes-agent-wt-t_bf5b8389-b2 held 554MB for a
# card in status `done`. Two independent scoping bugs hid it:
#   1. repo discovery was pinned to $HOME/workspace/*/.git, so no repo outside
#      that one directory was ever scanned — including the agent's own checkout;
#   2. reclaim_merged_worktrees gates on `merge-base --is-ancestor <tip> main`,
#      which a squash-merging repo never satisfies, so it has removed 0
#      worktrees over its entire log while reporting success.
#
# This class keys on the property that actually licenses deletion: THE CARD IS
# TERMINAL. A worktree named t_<id> is scratch created for that card; once the
# card is done/archived/cancelled the scratch is garbage regardless of how (or
# whether) its branch landed. Cards that are not terminal, and paths a live
# process holds, are kept. Repos are discovered from the worktree parents the
# guard already derives, never from a pinned directory list.
reclaim_dead_task_worktrees() {
  if fenced_child; then
    log "dead-card worktrees: SKIP fenced child context -- run disk-guard from an operator shell/launchd"
    return 0
  fi
  local before after n=0 repo p b id sz
  before=$(free_mb)
  while read -r repo; do
    [ -n "$repo" ] || continue
    git -C "$repo" worktree prune 2>/dev/null
    while read -r p; do
      [ -d "$p" ] || continue
      [ "$p" = "$repo" ] && continue
      id=$(basename "$p" | grep -oE 't_[0-9a-f]{8}' | head -1)
      [ -n "$id" ] || continue          # not card scratch; not ours to judge
      dead_task "$id" || continue       # non-terminal card: hands off
      path_in_use "$p" && { log "  skip (in use) $p"; continue; }
      sz=$(du -sxm "$p" 2>/dev/null | cut -f1)
      git -C "$repo" worktree remove --force "$p" 2>/dev/null || rm -rf "$p"
      [ -d "$p" ] || { n=$((n + 1)); log "  removed dead-card worktree ${sz}MB $p"; }
    done < <(git -C "$repo" worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2}')
    git -C "$repo" worktree prune 2>/dev/null
  done < <(worktree_repos)
  after=$(free_mb)
  log "dead-card worktrees: removed $n, $(( after - before ))MB"
}

# Repos that actually have worktrees, DISCOVERED from the checkouts on this
# volume rather than a pinned $HOME/workspace list (see the bug above).
worktree_repos() {
  {
    ls -d "$HOME"/workspace/*/.git 2>/dev/null | xargs -n1 dirname 2>/dev/null
    ls -d "$HOME"/.hermes/*/.git 2>/dev/null | xargs -n1 dirname 2>/dev/null
    ls -d "$HOME"/workspace/*/*/.git 2>/dev/null | xargs -n1 dirname 2>/dev/null
  } | sort -u
}

# --- target enforcement: silent, and strictly more of the SAME sanctioned
# classes. It never introduces a new category of deletion (that would make the
# quiet path riskier than the loud one) and it never pages — being under the
# capacity target is not an incident, it is a reason to work harder quietly.
#
# The per-user temp root is DERIVED from the running shell's TMPDIR, not the
# hardcoded /var/folders/<hash> path of one machine. It held 4.0GB of >1d-old
# agent scratch on 2026-09-25 and nothing was reaping it: reclaim_tmp only ever
# looked at /private/tmp.
user_tmp_root() { printf '%s' "${TMPDIR:-/tmp}" | sed 's:/$::'; }

reclaim_user_tmp() { # $1 = age in days
  local root freed_before freed_after
  root=$(user_tmp_root)
  case "$root" in /tmp|/private/tmp|''|/) return 0 ;; esac
  [ -d "$root" ] || return 0
  freed_before=$(free_mb)
  find "$root" -mindepth 1 -maxdepth 1 -mtime "+$1" \
       ! -name 'com.apple.*' ! -name '*.sock' \
       -exec rm -rf {} + 2>/dev/null
  freed_after=$(free_mb)
  log "user tmp ($root, >${1}d): $(( freed_after - freed_before ))MB"
}

reclaim_to_target() {
  local now
  now=$(free_gi)
  lt "$now" "$RECLAIM_TARGET_GI" || return 0
  log "below_target: ${now}Gi < ${RECLAIM_TARGET_GI}Gi — escalating quietly"
  reclaim_user_tmp 1
  now=$(free_gi)
  lt "$now" "$RECLAIM_TARGET_GI" || return 0
  TMP_AGE_SECONDS=3600 reclaim_tmp
  reclaim_user_tmp 0
  now=$(free_gi)
  lt "$now" "$RECLAIM_TARGET_GI" && \
    log "below_target: still ${now}Gi after escalation; sanctioned scratch exhausted; $(swap_shortfall_clause "$now") (silent by design — only FLOOR_GI pages)"
  return 0
}

# --- agent review clones in $HOME. The 2026-09-24 ENOSPC incident: 8 agent
# clones directly under $HOME held 4.3Gi of node_modules while the guard
# reported "the remaining consumers are NOT agent scratch". The class is
# "a git checkout an agent made outside the sanctioned scratch roots".
#
# SELECTION (the same predicate prove_disk_guard_node_modules_scope.sh asserts):
#   include  $HOME/<dir>/ that is a git repo, or a parent of git repos
#   exclude  ~/workspace (human checkouts), ~/Library, and every dotdir —
#            which is what keeps ~/.local/lib/node_modules (the global npm
#            prefix holding the pi/opencode CLIs) and ~/.hermes/hermes-agent
#            (the running agent itself) out of it.
# A first cut without those exclusions would have deleted the agent's own
# runtime, so the narrowing is load-bearing, not tidiness.
reclaim_home_node_modules() {
  local before after n e d
  before=$(free_mb); n=0
  for e in "$HOME"/[!.]*/; do
    case "$e" in "$HOME/workspace/"|"$HOME/Library/") continue ;; esac
    [ -d "${e}.git" ] || [ -f "${e}.git" ] || \
      { ls -d "$e"*/.git >/dev/null 2>&1 || continue; }
    while IFS= read -r d; do
      [ -n "$d" ] || continue
      path_in_use "$d" && continue
      rm -rf "$d" 2>/dev/null && n=$((n + 1))
    done < <(find "$e" -maxdepth 3 -type d -name node_modules -prune -mtime +0 -print 2>/dev/null)
  done
  after=$(free_mb)
  log "node_modules: removed $n agent-clone dirs in \$HOME, $(( after - before ))MB"
}

# --- pytest scratch roots. MEASURED 2026-09-25: the host fell 25Gi -> 16Gi in
# 25 minutes while this card was open, and the consumer was not any sanctioned
# scratch root. Every agent pytest run that builds a throwaway hermes-home
# provisions its OWN 1.8GB Chromium under $TMPDIR/pytest-of-<user>/pytest-N;
# pytest's own retention keeps the last 3 numbered roots and never accounts for
# size, so a burst of test runs adds GB/minute and nothing reaps it. reclaim_tmp
# did not see it (it only ever looked at /private/tmp) and it is not a git
# checkout, so the node_modules sweep did not either.
#
# Liveness, not age, is the predicate: these roots are minutes old by
# construction, so an age rule either deletes a running test's fixture or never
# fires. Keep `pytest-current` (the symlink target of the run in progress) and
# anything a live process holds open; evict the rest.
reclaim_pytest_roots() {
  local root base before after n d
  root=$(user_tmp_root)
  base="$root/pytest-of-$(id -un)"
  [ -d "$base" ] || return 0
  before=$(free_mb); n=0
  for d in "$base"/pytest-*; do
    [ -d "$d" ] || continue
    case "$(basename "$d")" in pytest-current) continue ;; esac
    [ "$d" -ef "$base/pytest-current" ] && continue
    path_in_use "$d" && continue
    rm -rf "$d" 2>/dev/null && n=$((n + 1))
  done
  after=$(free_mb)
  log "pytest roots: removed $n idle roots under $base, $(( after - before ))MB"
}

# Shell copy of done-card worktree reclaim. Known no-op on squash-merging
# repos (see reclaim_merged_worktrees). Shared by the probe-failure path and
# the hermes-reclaim rc!=0 path so the loop cannot drift between them.
reclaim_shell_worktrees() {
  # A non-zero hermes probe falls through to here. The fence has to hold on
  # this path too, or the child deletes what the probe refusal forbade.
  if fenced_child; then
    log "worktrees: SKIP fenced child context -- run disk-guard from an operator shell/launchd"
    return 0
  fi
  local repo
  for repo in $(ls -d "$HOME"/workspace/*/.git 2>/dev/null | xargs -n1 dirname); do
    git -C "$repo" worktree list 2>/dev/null | grep -q . && reclaim_merged_worktrees "$repo"
  done
}

# Done-card worktrees: one implementation, in hermes, shared with `kanban gc`.
# The shell copy is a fallback only (see reclaim_merged_worktrees).
reclaim_hermes_worktrees() {
  local HERMES_BIN probe_err probe_rc probe_line refusal_line wt_out wt_rc
  # DISK_GUARD_HERMES_BIN wins so a proof harness can point at a stub without
  # hiding a real `hermes` on PATH. Otherwise prefer PATH, then the venv copy
  # launchd has always used.
  if [ -n "${DISK_GUARD_HERMES_BIN:-}" ]; then
    HERMES_BIN=$DISK_GUARD_HERMES_BIN
  else
    HERMES_BIN=$(command -v hermes 2>/dev/null || true)
    [ -n "$HERMES_BIN" ] || HERMES_BIN="$HOME/.hermes/hermes-agent/venv/bin/hermes"
  fi

  # --dry-run is a CAPABILITY PROBE, and under the Kanban write fence it is
  # also the refusal we must quote. Always run it — including when fenced.
  # Do NOT unset HERMES_DELEGATED_CHILD_CONTEXT (agent/delegation_context.py,
  # hermes_cli/kanban.py): clearing it would bypass the trust boundary.
  # stderr is kept: a real refusal and a missing flag are different failures.
  probe_err=$("$HERMES_BIN" kanban reclaim --dry-run 2>&1 >/dev/null)
  probe_rc=$?
  # Live hermes writes a preamble on stderr before the real message
  # ("  Command helper: applied 1 secret"). Those lines are not the refusal
  # and must not disqualify it. SKIP quotes the refusal line itself; FAIL
  # keeps the first non-empty line that is not that preamble.
  #
  # Exact line, not a substring. "cannot mutate Kanban tasks" also appears in
  # unrelated plugin errors. Real hermes (hermes_cli/kanban.py:154 via _err,
  # rc 1) prints the "kanban: " prefix; a stub may omit it. Anything else
  # while fenced is FAIL — still no deletion.
  refusal_line=$(printf '%s\n' "$probe_err" | grep -x -m1 \
    -e 'kanban: delegate_task child contexts cannot mutate Kanban tasks via the CLI' \
    -e 'delegate_task child contexts cannot mutate Kanban tasks via the CLI' || true)
  probe_line=$(printf '%s\n' "$probe_err" | grep -Ev '^ *Command helper:' | grep -m1 '[^[:space:]]' || true)
  [ -n "$probe_line" ] || probe_line="(probe rc=${probe_rc}, no stderr)"

  # A fenced child never deletes: no real reclaim, no shell fallback.
  # SKIP only for the real fence refusal (rc 1 AND an exact refusal line).
  if fenced_child; then
    if [ "$probe_rc" = 1 ] && [ -n "$refusal_line" ]; then
      log "worktrees: SKIP fenced child context -- run disk-guard from an operator shell/launchd: $refusal_line"
    else
      log "worktrees: FAIL hermes reclaim probe rc=$probe_rc at $HERMES_BIN: $probe_line"
    fi
    return 0
  fi

  if [ "$probe_rc" = 0 ]; then
    wt_out=$("$HERMES_BIN" kanban reclaim --logs 2>&1); wt_rc=$?
    if [ "$wt_rc" = 0 ]; then
      log "worktrees: $(printf '%s' "$wt_out" | grep -E '^ *-> ' | tr '\n' ' ')"
    else
      log "worktrees: FAIL hermes reclaim rc=$wt_rc, falling back to the shell copy"
      printf '%s\n' "$wt_out" | tail -3 | while read -r l; do log "  $l"; done
      reclaim_shell_worktrees
    fi
    return 0
  fi

  # NOT routine. The shell fallback is a known no-op (squash-merge gate).
  # Every non-zero probe is this branch — say FAIL with rc and stderr so it
  # is greppable. Do not special-case stderr text and do not call it an upgrade.
  log "worktrees: FAIL hermes reclaim probe rc=$probe_rc at $HERMES_BIN: $probe_line"
  reclaim_shell_worktrees
}

# Ordinary --reclaim sequence. Defined above the library seam so a proof
# harness can drive the whole dispatch against a fixture HOME.
run_reclaim_classes() {
  reclaim_workspaces
  reclaim_safe_caches
  reclaim_hermes_worktrees
  reclaim_tmp
  reclaim_home_node_modules
  reclaim_pytest_roots
  reclaim_npm_cache
  reclaim_stale_pnpm_stores
  reclaim_stale_installs
  reclaim_dead_task_worktrees
  # Target enforcement runs LAST: only after every ordinary class has been
  # reclaimed do we decide whether to escalate.
  reclaim_to_target
}

before=$(free_gi)

# Library seam: `DISK_GUARD_LIB=1 source disk-guard.sh` defines the reclaim
# functions and stops, so a proof harness can drive ONE class against its own
# fixture. Without it a harness can only re-implement the predicate in its own
# words, which is how prove_disk_guard_node_modules_scope.sh ended up asserting
# a copy of the selection while the real sweep was missing from the script
# entirely for a full day.
[ "${DISK_GUARD_LIB:-0}" = 1 ] && return 0 2>/dev/null

DG_SWAP_USED_MEMO=$(swap_used_gi)
swap_before=$(swap_term)
log "free=${before}Gi floor=${FLOOR_GI}Gi target=${RECLAIM_TARGET_GI}Gi ${swap_before}"

# A fixture figure must never drive a real deletion.
if [ "$RECLAIM" = 1 ] && [ -n "${DISK_GUARD_FAKE_FREE_GI:-}" ]; then
  log "fixture: DISK_GUARD_FAKE_FREE_GI set — reclaim disabled (no deletion from a fixture figure)"
  RECLAIM=0
fi

if [ "$RECLAIM" = 1 ]; then
  run_reclaim_classes
fi

after=$(free_gi)
# Re-measure only when a reclaim ran in between; a check-only run is one
# instant, and a second top probe would just double a hung probe's bound.
swap_after=$swap_before
if [ "$RECLAIM" = 1 ]; then
  unset DG_SWAP_USED_MEMO
  DG_SWAP_USED_MEMO=$(swap_used_gi)
  swap_after=$(swap_term)
fi
log "free=${after}Gi (reclaimed $(awk -v a="$after" -v b="$before" 'BEGIN{printf "%.2f", a-b}')Gi) ${swap_after}"
report_check_only_shortfall "$after"

# The ONLY paging decision. RECLAIM_TARGET_GI deliberately does not appear
# below this line: a capacity shortfall is handled silently above, and letting
# the target reach this branch is exactly how the guard paged 67 times in 13h.
if lt "$after" "$FLOOR_GI"; then
  below_floor=1
  log "FAIL: ${after}Gi free is below the ${FLOOR_GI}Gi floor. ${swap_after}"
  log "An ENOSPC host silently breaks every agent tool call and watchdog."
  log "Top reclaimable, largest first (live-pid paths excluded):"
  top_reclaimable 8 | awk -F'\t' '{printf "  %sMB  %s\n", $1, $2}'
  log "NOTE: figures are du-apparent. On APFS du overstates a worker's MARGINAL"
  log "cost ~25x (pnpm clonefile CoW clones counted in full). Deleting 5 such"
  log "dirs on 2026-09-10 really freed 4.9Gi. Size capacity with scratch-cost.sh,"
  log "never from this list."
  exit 1
fi

log "OK: ${after}Gi free. ${swap_after}"
exit 0
