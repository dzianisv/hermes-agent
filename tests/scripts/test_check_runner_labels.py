"""scripts/ci/check_runner_labels.py: bare org-only ``*-core`` runner labels fail; guarded ones pass."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_runner_labels.py"

RED = """\
name: red
on: push
jobs:
  a:
    runs-on: ubuntu-latest-32-core
    steps: []
  b:
    strategy:
      matrix:
        include:
          - os: windows
            runner: windows-latest-32-core
    runs-on: ${{ matrix.runner }}
    steps: []
"""

GREEN = """\
name: green
on: push
jobs:
  a:
    # was ubuntu-latest-32-core before the fork fallback
    runs-on: ${{ github.repository_owner == 'NousResearch' && 'ubuntu-latest-32-core' || 'ubuntu-latest' }}
    steps: []
  b:
    strategy:
      matrix:
        include:
          - runner: ${{ github.repository_owner == 'NousResearch' && 'windows-latest-32-arm-core' || 'windows-11-arm' }}
    runs-on: ${{ matrix.runner }}
    steps: []
"""


def _write(root: Path, name: str, text: str) -> Path:
    wf = root / ".github" / "workflows"
    wf.mkdir(parents=True, exist_ok=True)
    path = wf / name
    path.write_text(text, encoding="utf-8")
    return path


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_bare_core_labels_fail_with_file_and_line(tmp_path):
    path = _write(tmp_path, "red.yml", RED)
    res = _run("--root", str(tmp_path), "--no-baseline")
    assert res.returncode == 1
    assert f"{path}:5: hard-coded org runner label 'ubuntu-latest-32-core'" in res.stdout
    assert f"{path}:12: hard-coded org runner label 'windows-latest-32-core'" in res.stdout


def test_block_and_inline_lists_are_scanned(tmp_path):
    text = "jobs:\n  a:\n    runs-on:\n      - self-hosted\n      - ubuntu-latest-96-core\n  b:\n    runs-on: [x, ubuntu-latest-32-arm-core]\n"
    path = _write(tmp_path, "lists.yaml", text)
    res = _run("--root", str(tmp_path), "--no-baseline")
    assert res.returncode == 1
    assert f"{path}:5:" in res.stdout
    assert f"{path}:7:" in res.stdout


def test_guarded_labels_and_comments_pass(tmp_path):
    _write(tmp_path, "green.yml", GREEN)
    _write(tmp_path, "comment.yml", "jobs:\n  a:\n    runs-on: ubuntu-latest  # not ubuntu-latest-32-core\n    #   runner: windows-latest-32-core\n")
    res = _run("--root", str(tmp_path), "--no-baseline")
    assert res.returncode == 0, res.stdout
    assert "check_runner_labels: OK (2 workflow files scanned)" in res.stdout


def test_baseline_listed_offender_is_skipped(tmp_path):
    _write(tmp_path, "red.yml", RED)
    _write(tmp_path, "green.yml", GREEN)
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# release lanes\nred.yml\n", encoding="utf-8")
    res = _run("--root", str(tmp_path), "--baseline", str(baseline))
    assert res.returncode == 0, res.stdout


def test_stale_baseline_entry_fails(tmp_path):
    _write(tmp_path, "green.yml", GREEN)
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("green.yml\n", encoding="utf-8")
    res = _run("--root", str(tmp_path), "--baseline", str(baseline))
    assert res.returncode == 1
    assert "stale baseline entry 'green.yml'" in res.stdout
