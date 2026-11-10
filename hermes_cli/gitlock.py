"""Stale git lock-file and aborted-fetch pack-debris recovery for update/check paths.

A killed ``git fetch`` can leave ``.git/shallow.lock`` behind (every later fetch fails with "Unable to
create '.../shallow.lock': File exists") and ``tmp_pack_*`` files git itself never cleans up.

The same class of debris is worse without a sweep: a crashed ``git merge``/``git add`` leaves
``.git/index.lock`` and git then reports ``error: Unable to write index.`` followed by
``Automatic merge failed; fix conflicts and then commit the result.`` — wording that reads like a
content conflict and sends the operator off to resolve files that do not conflict (2026-10-21: the
live runtime checkout was write-blocked for ~6.5 hours by a 0-byte lock). ``describe_repo_lock``
names the dead lock, its age and the liveness check that cleared it; every removal path is gated on
a live-holder probe of the exact file.
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Files younger than this are presumed live (a fetch may be in flight) and are never removed. Lock
# files live for seconds and a healthy fetch completes in minutes; 10 minutes is abandoned.
STALE_LOCK_MIN_AGE_SECONDS = 10 * 60
STALE_TMP_PACK_MIN_AGE_SECONDS = STALE_LOCK_MIN_AGE_SECONDS
# ``shallow.lock`` is the one observed in the wild; the others are the same class of failure
# (interrupted git operation). Locks held by a live git process are protected by the process guard.
LOCK_NAMES = ("shallow.lock", "index.lock", "HEAD.lock", "MERGE_HEAD.lock")
# Temp-file prefixes git writes into .git/objects/pack during a transfer and renames away on
# success; anything left with these names after a fetch died is garbage by definition.
# ``.tmp-<pid>-pack*`` is the same thing from ``pack-objects`` (repack/gc) killed mid-write.
_TMP_PACK_PREFIXES = ("tmp_pack_", "tmp_idx_", "tmp_rev_", "tmp_mtimes_", ".tmp-")


def _proc_cwd(pid: int) -> Optional[str]:
    """``/proc/<pid>/cwd`` target; None where the kernel has no /proc or it is unreadable.

    ``os.readlink`` and not ``os.path.realpath``: realpath NORMALISES a path it cannot resolve
    instead of raising, so on a host without /proc it returned ``"/proc/<pid>/cwd"`` for every
    pid and the conservative ``cwd is None`` branch in :func:`_git_pids_working_in` was
    unreachable. That made every pid — including a live git working in the repo — classify as
    "not our writer", and the lock sweep then unlinked a lock a running git held open.
    """
    try:
        target = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return None
    # A deleted cwd cannot be matched against a repo root; report it as unknown so callers stay
    # conservative rather than treating the pid as "working somewhere else".
    return None if target.endswith(" (deleted)") else target


def _proc_fd_scan_available() -> bool:
    """True when this host exposes the ``/proc/<pid>/fd`` view used for exact-holder probes."""
    return (Path("/proc") / "self" / "fd").is_dir()


def _git_pids_working_in(repo_root: Optional[Path]) -> tuple[int, ...]:
    """Live ``git`` PIDs working in ``repo_root`` (all of them when the scope is unknown).

    Machine-wide ``pgrep -x git`` is too blunt to gate anything: a git credential-cache or
    fsmonitor daemon has the comm ``git`` and runs for days, so the coarse probe made the lock
    sweep skip forever on such hosts. Does that make us miss a real writer? Only one operating on
    another repo — which cannot be holding THIS repo's lock; a ``git -C <repo>`` run whose cwd is
    elsewhere is still caught by the exact open-fd probe on the file itself.
    """
    pids = _git_pids()
    if not pids or repo_root is None:
        return pids
    root = os.path.realpath(repo_root)
    working = []
    for pid in pids:
        cwd = _proc_cwd(pid)
        # Unknown cwd (non-Linux, restricted /proc) stays conservative: assume it may be the writer.
        if cwd is None or cwd == root or cwd.startswith(root + os.sep):
            working.append(pid)
    return tuple(working)


def _git_proc_running(repo_root: Optional[Path] = None) -> bool:
    """True when a ``git`` process is running — the check that stops us yanking a lock a live fetch holds.

    Scoped to ``repo_root`` when given (a git in another checkout is not our writer); machine-wide
    when the scope is unknown. A failed probe logs and returns False; the age floor still applies.
    """
    try:
        if os.name == "nt":
            proc = subprocess.run(["tasklist", "/FI", "IMAGENAME eq git.exe", "/FO", "CSV"],
                                  capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
            return "git.exe" in proc.stdout.lower()
        if repo_root is None:
            proc = subprocess.run(["pgrep", "-x", "git"], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=10)
            return proc.returncode == 0
        return bool(_git_pids_working_in(repo_root))
    except Exception:
        logger.debug("git process probe failed; assuming no git running", exc_info=True)
        return False


def _open_holders(paths: List[Path]) -> dict[Path, tuple[int, ...]]:
    """PIDs holding any of ``paths`` open (one /proc pass); {} when no holder probe can run.

    Removals are gated on this exact-file probe, not only on the process-name guess: a git that has
    the inode open is live even when ``pgrep -x git`` raced and found nothing.

    Without a ``/proc`` fd view (macOS, Windows) the probe falls back to ``lsof -t``. Before that
    fallback existed the sweep had NO holder check on those platforms, and once the cwd probe was
    fixed to stay conservative it could still unlink a lock a live process held open.
    """
    targets = {os.path.realpath(p): p for p in paths}
    if not targets:
        return {}
    found: dict[Path, set[int]] = {}
    if _proc_fd_scan_available():
        for fd_dir in Path("/proc").glob("[0-9]*/fd"):
            try:
                pid = int(fd_dir.parent.name)
                for entry in os.scandir(fd_dir):
                    try:
                        target = targets.get(os.readlink(entry.path))
                    except OSError:
                        continue
                    if target is not None:
                        found.setdefault(target, set()).add(pid)
            except OSError:
                continue
    else:
        for original in targets.values():
            pids = _lsof_holders(original)
            if pids:
                found.setdefault(original, set()).update(pids)
    return {path: tuple(sorted(pids)) for path, pids in found.items()}


def _sweep_stale(directory: Path, candidates: Callable[[], Iterable[Path]], *, min_age_seconds: Optional[int],
                 default_age: int, skip_msg: str, log_removed: Callable[[Path, int], None],
                 repo_root: Optional[Path] = None) -> List[str]:
    """Shared guard + age-floor sweep. Never raises; skips anything it cannot stat/unlink."""
    if not directory.is_dir():
        return []
    if _git_proc_running(repo_root):
        logger.debug(skip_msg)
        return []
    cutoff = time.time() - (min_age_seconds if min_age_seconds is not None else default_age)
    eligible: List[tuple[Path, os.stat_result]] = []
    for entry in candidates():
        try:
            if entry.is_file() and (st := entry.stat()).st_mtime < cutoff:
                eligible.append((entry, st))
        except OSError:
            continue
    holders = _open_holders([entry for entry, _st in eligible])
    removed: List[str] = []
    for entry, st in eligible:
        if holders.get(entry):
            logger.warning("Skipping %s: still held open by PID(s) %s",
                           entry, ", ".join(map(str, holders[entry])))
            continue
        try:
            if os.name == "nt":
                # git renames its transfer temps into place read-only; Windows refuses to
                # unlink a read-only file (EACCES/13), so without clearing the write bit
                # this sweep silently removes nothing on real debris (#116384).
                try:
                    os.chmod(entry, 0o666)
                except OSError:
                    pass
            entry.unlink()
            removed.append(str(entry))
            log_removed(entry, st.st_size)
        except OSError as exc:
            # A cleaner that fails silently is worse than none: debug-level skips hid the
            # Windows read-only unlink failure for months while debris grew to gigabytes.
            logger.warning("Could not clear %s (skipping): %s", entry, exc)
    return removed


def clear_stale_git_locks(repo_root: Path, *, min_age_seconds: Optional[int] = None) -> List[str]:
    """Remove abandoned ``.git`` lock files under ``repo_root``; returns the removed paths.

    Removes only when older than the age floor, no git process is working in this repo, and no live
    process has the file open. Never raises: a lock we cannot stat/unlink is skipped (it may have
    been re-created between the age check and the unlink; skipping is safe).
    """
    git_dir = Path(repo_root) / ".git"
    return _sweep_stale(
        git_dir, lambda: [git_dir / name for name in LOCK_NAMES],
        min_age_seconds=min_age_seconds, default_age=STALE_LOCK_MIN_AGE_SECONDS,
        skip_msg="git process running; skipping stale-lock sweep",
        log_removed=lambda p, _size: logger.info("Removed stale git lock %s", p),
        repo_root=Path(repo_root),
    )


def clear_stale_tmp_packs(repo_root: Path, *, min_age_seconds: Optional[int] = None) -> List[str]:
    """Remove aborted-transfer temp pack files; same contract as clear_stale_git_locks.

    Resolves ``.git/objects/pack`` for a checkout and ``objects/pack`` for a bare repo such as
    the checkpoint store — a ``git gc`` killed by a timeout strands the same debris there."""
    git_dir = Path(repo_root) / ".git"
    pack_dir = (git_dir if git_dir.is_dir() else Path(repo_root)) / "objects" / "pack"

    def _candidates():
        try:
            return [e for e in pack_dir.iterdir() if e.name.startswith(_TMP_PACK_PREFIXES)]
        except OSError:
            return []

    return _sweep_stale(
        pack_dir, _candidates,
        min_age_seconds=min_age_seconds, default_age=STALE_TMP_PACK_MIN_AGE_SECONDS,
        skip_msg="git process running; skipping tmp-pack sweep",
        log_removed=lambda p, size: logger.info("Removed aborted-fetch pack debris %s (%d bytes)", p, size),
        repo_root=Path(repo_root),
    )


# ---- stale-lock diagnosis (name the dead lock; never call it a conflict) ----

# git's wording when a leftover lock blocks a write is a WRITE failure, but it is followed by
# "Automatic merge failed; fix conflicts and then commit the result." — which reads like a content
# conflict. The lock names worth diagnosing, in the order the incident produced them.
DIAGNOSABLE_LOCK_NAMES = ("index.lock", "MERGE_HEAD.lock", "HEAD.lock", "shallow.lock", "gc.log.lock")


@dataclass(frozen=True)
class LockReport:
    """What the liveness probes found for one existing lock file."""

    path: Path
    age_seconds: float
    size: int
    holder_pids: tuple[int, ...] = ()   # processes that have the lock file open
    git_pids: tuple[int, ...] = ()      # live git processes (scoped to the repo when known)
    checks: tuple[str, ...] = ()        # the probes that actually ran, with their verdict
    probe_available: bool = False       # some probe could see a holder on this platform

    @property
    def held(self) -> bool:
        return bool(self.holder_pids)


def _format_age(seconds: float) -> str:
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m {total % 60}s"
    if total < 86400:
        return f"{total // 3600}h {(total % 3600) // 60}m"
    return f"{total // 86400}d {(total % 86400) // 3600}h"


def _pids_from_probe(args: List[str], *, accept_codes: tuple[int, ...]) -> Optional[tuple[int, ...]]:
    """PIDs printed by a probe (one per line); None when the probe could not run."""
    try:
        proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=10)
    except Exception:
        return None
    if proc.returncode not in accept_codes:
        return None
    return tuple(sorted({int(tok) for tok in proc.stdout.split() if tok.isdigit()}))


def _proc_fd_holders(target: Path) -> Optional[tuple[int, ...]]:
    """PIDs holding ``target`` open, or None when this platform has no /proc fd view (macOS, Windows).

    This is the check that matters for the safety rule: a ``git`` between closing its lock and
    renaming it into place holds the fd even though it is not visible as a separate process name.
    """
    if not _proc_fd_scan_available():
        return None
    real = os.path.realpath(target)
    holders: set[int] = set()
    for fd_dir in Path("/proc").glob("[0-9]*/fd"):
        try:
            for entry in os.scandir(fd_dir):
                try:
                    if os.readlink(entry.path) == real:
                        holders.add(int(fd_dir.parent.name))
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return tuple(sorted(holders))


def _lsof_holders(target: Path) -> Optional[tuple[int, ...]]:
    """``lsof -t`` PIDs holding ``target``; None when lsof is not installed."""
    exe = shutil.which("lsof")
    if not exe:
        return None
    return _pids_from_probe([exe, "-t", str(target)], accept_codes=(0, 1))


def _pids_from_tasklist_csv(output: str) -> tuple[int, ...]:
    """PIDs from ``tasklist /FO CSV`` output, parsed as CSV.

    ``tasklist`` quotes every field, so the whitespace tokeniser in :func:`_pids_from_probe`
    never sees a bare PID (it would report the Session# or nothing). Rows are
    ``"Image Name","PID","Session Name","Session#","Mem Usage"``; the header and the
    "INFO: No tasks are running..." line are skipped by the ``git.exe`` name check.
    """
    pids: set[int] = set()
    for row in csv.reader(output.splitlines()):
        if len(row) < 2 or row[0].strip().lower() != "git.exe":
            continue
        pid = row[1].strip()
        if pid.isdigit():
            pids.add(int(pid))
    return tuple(sorted(pids))


def _git_pids() -> tuple[int, ...]:
    """Live ``git`` PIDs. Machine-wide by nature: pgrep cannot say which repo a git works on."""
    if os.name == "nt":
        try:
            proc = subprocess.run(["tasklist", "/FI", "IMAGENAME eq git.exe", "/FO", "CSV"],
                                  capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=10)
        except Exception:
            return ()
        if proc.returncode != 0:
            return ()
        return _pids_from_tasklist_csv(proc.stdout)
    return _pids_from_probe(["pgrep", "-x", "git"], accept_codes=(0, 1)) or ()


def inspect_lock_file(lock_path: Path, *, repo_root: Optional[Path] = None) -> Optional[LockReport]:
    """Probe one existing lock file; None when it does not exist. Never raises.

    ``repo_root`` scopes the git-process probe to writers of that repo (a credential-cache daemon
    elsewhere must not be reported as a possible owner); without it the probe stays machine-wide.
    """
    try:
        st = Path(lock_path).stat()
    except OSError:
        return None
    checks: List[str] = []
    holders: set[int] = set()
    probe_available = False
    fd_holders = _proc_fd_holders(lock_path)
    if fd_holders is None:
        checks.append("/proc/*/fd open-fd scan: unavailable on this platform")
    else:
        probe_available = True
        holders.update(fd_holders)
        checks.append("/proc/*/fd open-fd scan: " + (
            "held by PID(s) " + ", ".join(map(str, fd_holders)) if fd_holders else "no holder"))
    lsof_holders = _lsof_holders(lock_path)
    if lsof_holders is None:
        checks.append("lsof: unavailable")
    else:
        probe_available = True
        holders.update(lsof_holders)
        checks.append("lsof: " + (
            "held by PID(s) " + ", ".join(map(str, lsof_holders)) if lsof_holders else "no holder"))
    if repo_root is not None:
        git_pids = _git_pids_working_in(Path(repo_root))
        probe = "tasklist git.exe + process cwd" if os.name == "nt" else "pgrep -x git + /proc/<pid>/cwd"
        # A pid whose cwd could not be read is kept conservatively, so only the resolved ones are
        # certainly working here; naming that keeps the report honest on hosts without /proc.
        verdict = ("working in (or not excludable from) this repo as PID(s) "
                   + ", ".join(map(str, git_pids))) if git_pids else "no git process working in this repo"
    else:
        git_pids = _git_pids()
        probe = "tasklist git.exe, machine-wide" if os.name == "nt" else "pgrep -x git, machine-wide"
        verdict = ("live PID(s) " + ", ".join(map(str, git_pids))) if git_pids else "no git process"
    checks.append(f"git process probe ({probe}): {verdict}")
    return LockReport(
        path=Path(lock_path), age_seconds=max(0.0, time.time() - st.st_mtime), size=st.st_size,
        holder_pids=tuple(sorted(holders)), git_pids=git_pids, checks=tuple(checks),
        probe_available=probe_available,
    )


def format_lock_report(report: LockReport) -> str:
    """One operator-facing paragraph: what the lock is, whether it is safe to remove, and why."""
    where = str(report.path)
    label = report.path.name
    age = _format_age(report.age_seconds)
    checks = "; ".join(report.checks)
    if report.held:
        return (f"{label} (age {age}, {report.size} bytes) at {where} is HELD by live PID(s) "
                f"{', '.join(map(str, report.holder_pids))} — {checks}. Do NOT remove it: wait for that "
                f"process to finish. This is not a conflict.")
    if report.git_pids:
        return (f"{label} (age {age}, {report.size} bytes) at {where} is not held open by any process, "
                f"but live git process(es) {', '.join(map(str, report.git_pids))} are running — {checks}. "
                f"If those are not working on this repo the lock is stale: confirm with `lsof {where}` "
                f"showing no holder, then `rm {where}`. There is no conflict to resolve.")
    if not report.probe_available:
        return (f"{label} (age {age}, {report.size} bytes) at {where} exists — {checks}. No liveness "
                f"probe is available here, so confirm by hand (`lsof {where}` or the platform "
                f"equivalent) before removing it: git reported a write failure, not a conflict.")
    return (f"stale {label} (age {age}, {report.size} bytes) at {where} — owner PID is gone: no live "
            f"process holds it and no git process is running ({checks}). git's \"fix conflicts\" line is "
            f"misleading; this lock, not a conflict, blocked the write. Safe to remove: rm {where}, then "
            f"retry. If git also refuses to abort because .git/MERGE_HEAD is left over, remove that file too.")


def describe_lock_file(lock_path: Path, *, repo_root: Optional[Path] = None) -> Optional[str]:
    """Message for one lock path; None when it does not exist."""
    report = inspect_lock_file(lock_path, repo_root=repo_root)
    return format_lock_report(report) if report is not None else None


def _resolve_git_dir(repo_root: Path) -> Optional[Path]:
    """The repo's git dir via git (handles worktrees and ``.git`` files), else ``<repo>/.git``."""
    try:
        proc = subprocess.run(["git", "rev-parse", "--absolute-git-dir"], cwd=str(repo_root),
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
        if proc.returncode == 0 and proc.stdout.strip():
            return Path(proc.stdout.strip())
    except Exception:
        logger.debug("git dir probe failed for %s", repo_root, exc_info=True)
    fallback = Path(repo_root) / ".git"
    return fallback if fallback.is_dir() else None


def describe_repo_lock(repo_root: Path, *, names: Iterable[str] = DIAGNOSABLE_LOCK_NAMES) -> Optional[str]:
    """Message naming the first leftover lock in ``repo_root``; None when there is none.

    Used by the terminal failure hint so a blocked ``git merge`` reports the dead lock instead of
    sending the reader to resolve conflicts that do not exist.
    """
    git_dir = _resolve_git_dir(Path(repo_root))
    if git_dir is None:
        return None
    for name in names:
        described = describe_lock_file(git_dir / name, repo_root=Path(repo_root))
        if described:
            return described
    return None


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def is_ancestor_of_head(repo_root: Path, rev: str) -> bool:
    """True when ``rev`` is an ancestor of (or equal to) HEAD.

    Wraps ``git merge-base --is-ancestor <rev> HEAD``. This is the correct
    question for update checks: a local cherry-pick on top of the remote tip
    makes HEAD *different* from ``origin/main`` but still *contains* it, so
    the answer to "is there an update?" is no.

    Returns False on any probe failure (missing rev, shallow boundary, git
    error) — callers treat that as "can't prove contained", which is the
    conservative direction for an update check.
    """
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", rev, "HEAD"],
            cwd=str(repo_root),
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        )
        return result.returncode == 0
    except Exception:
        logger.debug("merge-base --is-ancestor probe failed for %s", rev, exc_info=True)
        return False
# ---- END PLUGIN-COMPAT ----


def _git_stdout_lines(repo_root: Path, args: List[str]) -> List[str]:
    """Run a read-only git query in ``repo_root``; [] on any failure."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=str(repo_root),
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        logger.debug("git query failed: %s", args, exc_info=True)
        return []


def _batch_missing_parents(repo_root: Path, candidates: List[str]) -> set[str]:
    """Return local commit objects whose parent objects are missing.

    Parents are read from the commit *header* only (lines before the first blank
    line) — a ``parent <sha>`` line inside a commit message body is prose, not
    an edge.
    """
    if not candidates:
        return set()
    try:
        parents_by_commit = {}
        parents = set()
        request = "\n".join(candidates) + "\n"
        result = subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=str(repo_root),
            input=request.encode(),
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            return set()
        data = result.stdout
        cursor = 0
        for candidate in candidates:
            header_end = data.find(b"\n", cursor)
            if header_end < 0:
                return set()
            header = data[cursor:header_end].split()
            cursor = header_end + 1
            if len(header) >= 3 and header[1] == b"commit":
                size = int(header[2])
                body = data[cursor:cursor + size]
                cursor += size
                if data[cursor:cursor + 1] != b"\n":
                    return set()
                cursor += 1
                commit_parents = set()
                for line in body.split(b"\n"):
                    if not line:  # blank line ends the commit header block
                        break
                    fields = line.split()
                    if len(fields) >= 2 and fields[0] == b"parent":
                        commit_parents.add(fields[1].decode())
                parents_by_commit[candidate] = commit_parents
                parents.update(commit_parents)
            elif len(header) < 2 or header[1] != b"missing":
                return set()
        if not parents:
            return set()
        check = subprocess.run(
            ["git", "cat-file", "--batch-check"],
            cwd=str(repo_root),
            input=("\n".join(sorted(parents)) + "\n").encode(),
            capture_output=True,
            timeout=10,
        )
        if check.returncode != 0:
            return set()
        missing = {
            line.split()[0]
            for line in check.stdout.decode(errors="replace").splitlines()
            if line.endswith(" missing")
        }
        return {commit for commit, commit_parents in parents_by_commit.items() if commit_parents & missing}
    except Exception:
        logger.debug("parent-object probe failed for %s", repo_root, exc_info=True)
        return set()


def _shallow_file_path(repo_root: Path) -> Optional[Path]:
    """Resolve ``.git/shallow`` via git, or None when the repo has none."""
    shallow_rel = _git_stdout_lines(repo_root, ["rev-parse", "--git-path", "shallow"])
    if not shallow_rel:
        return None
    shallow_path = Path(shallow_rel[0])
    if not shallow_path.is_absolute():
        shallow_path = Path(repo_root) / shallow_rel[0]
    return shallow_path if shallow_path.is_file() else None


class _ShallowLock:
    """Git's own ``shallow.lock`` protocol, so a concurrent ``git fetch`` that
    writes ``.git/shallow`` between our read and our write is never clobbered:
    the fetch fails fast on the lock and we re-read before writing."""

    def __init__(self, shallow_path: Path):
        self._path = shallow_path
        self._lock_path = shallow_path.with_name(shallow_path.name + ".lock")

    def __enter__(self) -> "_ShallowLock":
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f"shallow lock held: {self._lock_path}")
        except OSError as exc:
            raise RuntimeError(f"cannot create shallow lock: {exc}") from exc
        try:
            os.write(fd, b"hermes shallow maintenance\n")
        finally:
            os.close(fd)
        return self

    def __exit__(self, *exc_info) -> None:
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass


def _write_shallow(shallow_path: Path, content: str, *, suffix: str) -> None:
    """Atomically replace ``.git/shallow`` (temp file + os.replace)."""
    tmp_path = shallow_path.with_name(shallow_path.name + suffix)
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, shallow_path)


def repair_broken_shallow_boundaries(repo_root: Path) -> int:
    """Re-append shallow boundaries for reflog-reachable commits whose parents
    were never fetched (#108286).

    A reflog-only commit whose graft was pruned leaves the repo unwalkable
    (``gc``/``fsck``/``fetch`` fail) and unable to self-heal: reflog expiry
    happens during ``git gc``, which is exactly what the corruption breaks.
    Only reflog-reachable commits are considered, so unrelated object loss is
    never re-labelled as shallow history. Returns the number of boundaries
    appended; never raises.
    """
    try:
        shallow_path = _shallow_file_path(repo_root)
        if shallow_path is None:
            return 0
        # Cheap gate: repair only when the walk the corruption breaks already fails.
        probe = subprocess.run(
            ["git", "rev-list", "--count", "--all", "--reflog"],
            cwd=str(repo_root), capture_output=True, timeout=10,
        )
        if probe.returncode == 0:
            return 0
        with _ShallowLock(shallow_path):
            # Keep the rollback image byte-exact, including BOM and line endings.
            original = shallow_path.read_bytes()
            existing = {line for line in original.decode("utf-8-sig").splitlines() if line}
            if not existing:
                return 0
            # Boundary candidates: commits recorded as *fetch tips* in remote-tracking
            # refs' reflogs (NOT --batch-all-objects, and not HEAD's reflog): the bug
            # class is fetched tips whose graft the prune dropped, and restricting to
            # fetch-recorded tips is what keeps unrelated object loss (a deleted parent
            # of a locally-created commit) from being re-labelled as shallow history.
            # On an already-corrupted repo a rev-list --reflog walk is exactly what
            # fails, so read the reflog hash list directly. Scope: fetch-by-SHA
            # installs (scripts/install.sh) record their tip only in HEAD's reflog
            # and are NOT candidates — new corruption of that shape is prevented by
            # the prune's reflog fail-safe instead.
            reflog = _git_stdout_lines(
                repo_root, ["reflog", "show", "--all", "--format=%H%x00%gD"])
            candidates = sorted({
                sha
                for entry in reflog
                for sha, selector in [entry.split("\x00", 1)]
                if selector.startswith("refs/remotes/")
            })
            broken = _batch_missing_parents(repo_root, candidates)
            repaired = broken - existing
            if not repaired:
                return 0
            _write_shallow(shallow_path, "\n".join(sorted(existing | repaired)) + "\n",
                           suffix=".hermes-repair")
            # Self-check under the same lock hold (rev-list never takes
            # shallow.lock): the rollback cannot be defeated by lock contention.
            if not _git_stdout_lines(repo_root, ["rev-list", "--count", "--all", "--reflog"]):
                shallow_path.write_bytes(original)
                logger.debug("shallow boundary repair self-check failed; file restored")
                return 0
        logger.info("Restored %d broken shallow boundary(ies) in %s", len(repaired), repo_root)
        return len(repaired)
    except Exception:
        logger.debug("shallow boundary repair failed for %s", repo_root, exc_info=True)
        return 0


def prune_stale_shallow_grafts(repo_root: Path) -> int:
    """Drop ``.git/shallow`` graft lines no live ref or reflog still points at (#105951).

    Every ``git fetch --depth 1`` appends the fetched tip to ``.git/shallow`` as a new
    graft and never removes the previous one, so a long-lived shallow installer checkout
    accumulates one graft per update check (57 observed in the wild). The stale grafts
    break ``merge-base`` and push ``hermes update`` into the orphan-divergence reset path
    on every run. Keep only the boundaries that still protect referenced tips (HEAD,
    FETCH_HEAD, and every ref tip): the dropped commits are already unreachable and their
    objects are left for ``git gc``. Returns the number of graft lines removed; never
    raises, and restores the original file if the trimmed set breaks history walking
    (including the ``--reflog`` walk, so a graft a reflog-only commit still needs is
    never dropped, #108286).
    """
    try:
        shallow_path = _shallow_file_path(repo_root)
        if shallow_path is None:
            return 0
        with _ShallowLock(shallow_path):
            # Decode for pruning, but retain the same capture for a lossless rollback.
            original = shallow_path.read_bytes()
            lines = [line for line in original.decode("utf-8-sig").splitlines() if line]
            if not lines:
                return 0
            keep = set(lines) & {
                *(_git_stdout_lines(repo_root, ["rev-parse", "HEAD"]) or []),
                *(_git_stdout_lines(repo_root, ["rev-parse", "--verify", "--quiet", "FETCH_HEAD"]) or []),
                *_git_stdout_lines(repo_root, ["for-each-ref", "--format=%(objectname)"]),
            }
            if len(keep) == len(lines):
                return 0
            _write_shallow(shallow_path, "\n".join(sorted(keep)) + "\n", suffix=".hermes-prune")
            # Fail-safe: if any reachable walk now crosses a boundary we wrongly
            # removed, put the grafts back — a growing file beats a broken repo.
            # Runs under the same lock hold (rev-list never takes shallow.lock) so
            # the rollback cannot be defeated by lock contention.
            still_walks = _git_stdout_lines(repo_root, ["rev-list", "--count", "HEAD"]) and \
                _git_stdout_lines(repo_root, ["rev-list", "--count", "--all"]) and \
                _git_stdout_lines(repo_root, ["rev-list", "--count", "--all", "--reflog"])
            if not still_walks:
                shallow_path.write_bytes(original)
                logger.debug("shallow prune self-check failed; grafts restored")
                return 0
        logger.info("Pruned %d stale shallow graft(s) in %s", len(lines) - len(keep), repo_root)
        return len(lines) - len(keep)
    except Exception:
        logger.debug("shallow graft prune failed for %s", repo_root, exc_info=True)
        return 0


def fetch_full_commit_graph(repo_root: Path, **run_kwargs) -> bool:
    """Refresh release tags and fill shallow history before publishing identity.

    A full commit graph does not imply current tags, especially after a --no-tags
    clone. Fetch version tags explicitly without fetching every remote branch or
    replacing existing tags. Trees and blobs stay on demand. Returns whether the
    checkout was unshallowed; fetch failures raise subprocess errors.
    """
    shallow = _shallow_file_path(repo_root) is not None
    subprocess.run(
        ["git", "fetch", "--quiet", *(["--unshallow"] if shallow else []),
         "--filter=tree:0", "--no-tags", "origin", "refs/tags/v*:refs/tags/v*"],
        cwd=str(repo_root), check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=900, **run_kwargs,
    )
    return shallow
