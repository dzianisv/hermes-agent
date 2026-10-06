#!/usr/bin/env python3
"""Forbid hard-coded org-only ``*-core`` runner labels in GitHub workflows.

NousResearch has GitHub larger runners (``ubuntu-latest-96-core`` etc.); forks
do not, so a bare ``*-core`` label queues forever there. Every such label must
sit inside ``${{ github.repository_owner == 'NousResearch' && '<label>' || '<hosted>' }}``
so upstream renders the original label and forks fall back to hosted runners.

Line-based on purpose (stdlib only, no PyYAML): it scans ``runs-on:`` and
``runner:`` values, including ``runs-on:`` block lists and inline ``[a, b]``
lists. Workflows listed in the baseline file are skipped; a baseline entry
that no longer has offenders fails so the list can only shrink.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_BASELINE = SCRIPT_DIR / "runner_labels_baseline.txt"

CORE_LABEL = re.compile(r"[A-Za-z0-9_.-]+-core\b")
KEY_LINE = re.compile(r"^(\s*)(?:-\s+)?(runs-on|runner)\s*:(.*)$")
LIST_ITEM = re.compile(r"^(\s*)-\s+(.*)$")
GUARD = "github.repository_owner == 'NousResearch' && '{label}'"


def strip_comment(line: str) -> str:
    """Drop a ``#`` comment that is outside single/double quotes."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def offending_labels(value: str) -> list[str]:
    """Return ``-core`` labels in *value* not guarded by the NousResearch check."""
    return [
        label
        for label in CORE_LABEL.findall(value)
        if GUARD.format(label=label) not in value
    ]


def iter_values(text: str):
    """Yield ``(lineno, value)`` for every runs-on/runner value in *text*."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = strip_comment(lines[i])
        m = KEY_LINE.match(line)
        i += 1
        if not m:
            continue
        indent, key, value = len(m.group(1)), m.group(2), m.group(3).strip()
        if value:
            yield i, value
            continue
        if key != "runs-on":
            continue
        # Block list on following lines: ``- label``.
        while i < len(lines):
            nxt = strip_comment(lines[i])
            if not nxt.strip():
                i += 1
                continue
            item = LIST_ITEM.match(nxt)
            if not item or len(item.group(1)) < indent:
                break
            yield i + 1, item.group(2).strip()
            i += 1


def scan_file(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8-sig")
    return [
        (lineno, label)
        for lineno, value in iter_values(text)
        for label in offending_labels(value)
    ]


def load_baseline(path: Path) -> set[str]:
    names = set()
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        name = raw.split("#", 1)[0].strip()
        if name:
            names.add(name)
    return names


def workflow_files(root: Path) -> list[Path]:
    wf = root / ".github" / "workflows"
    return sorted([*wf.glob("*.yml"), *wf.glob("*.yaml")])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", type=Path, help="explicit workflow files")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args(argv)

    files = args.files or workflow_files(args.root)
    baseline = set() if args.no_baseline else load_baseline(args.baseline)

    failed = False
    seen_baseline_offenders: set[str] = set()
    for path in files:
        offenders = scan_file(path)
        if path.name in baseline:
            if offenders:
                seen_baseline_offenders.add(path.name)
            continue
        for lineno, label in offenders:
            failed = True
            print(
                f"{path}:{lineno}: hard-coded org runner label '{label}' "
                "outside github.repository_owner == 'NousResearch' guard"
            )

    scanned_names = {p.name for p in files}
    for name in sorted(baseline - seen_baseline_offenders):
        if args.files and name not in scanned_names:
            continue
        failed = True
        print(
            f"{args.baseline}: stale baseline entry '{name}' has no hard-coded "
            "org runner labels; remove it"
        )

    if failed:
        return 1
    print(f"check_runner_labels: OK ({len(files)} workflow files scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
