"""Larger-runner labels must never be requested unconditionally on a fork.

GitHub's *larger runners* (``ubuntu-latest-96-core``, ``ubuntu-latest-32-core``,
``ubuntu-latest-32-arm-core``, ``windows-latest-32-core``, ...) are billed to,
and only exist in, the upstream organization's runner pool. A fork that asks
for one gets a job that stays queued forever with ``runner_id=0``; it never
fails, it never starts, and the PR never goes green.

So every larger-runner request in ``.github/workflows`` must either

* be repository-conditional — the ``runs-on`` value is a ``${{ ... }}``
  expression containing ``github.repository == 'NousResearch/hermes-agent'``
  so the fork falls back to a standard hosted runner — or
* live in a job that is skipped outright on a fork via an ``if:`` containing
  the same repository check (this is how ``docker.yml`` is written).

This file parses the workflow YAML itself rather than asserting on a
hand-maintained list of files, so a NEWLY added larger-runner job anywhere in
the tree turns this test red automatically. It reads workflow *configuration*,
never Python source.

Classification is fail-closed: anything that is not one of the known standard
GitHub-hosted labels counts as a larger runner.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"

UPSTREAM_GUARD = "github.repository == 'NousResearch/hermes-agent'"

# Standard GitHub-hosted runner labels. This set exists ONLY to classify a
# label as "standard"; everything else — including a label nobody here has
# seen before — is treated as a larger runner (fail-closed).
STANDARD_HOSTED_LABELS = frozenset(
    {
        "ubuntu-latest",
        "ubuntu-24.04",
        "ubuntu-22.04",
        "ubuntu-24.04-arm",
        "windows-latest",
        "windows-2022",
        "windows-2025",
        "macos-latest",
        "macos-14",
        "macos-15",
    }
)

_EXPRESSION_RE = re.compile(r"\$\{\{(.+?)\}\}", re.DOTALL)
_MATRIX_REF_RE = re.compile(r"matrix\.([A-Za-z_][A-Za-z0-9_-]*)")
_STRING_LITERAL_RE = re.compile(r"'([^']*)'")
# A quoted operand shaped like a hosted runner label: "<os>-<something>".
_RUNNER_LABEL_SHAPE_RE = re.compile(r"^(?:ubuntu|windows|macos)-[A-Za-z0-9._-]+$")


def _workflow_files() -> list[Path]:
    files = sorted(
        p for p in WORKFLOWS_DIR.iterdir() if p.suffix in (".yml", ".yaml")
    )
    assert files, f"no workflow files found under {WORKFLOWS_DIR}"
    return files


def _is_expression(value: str) -> bool:
    return bool(_EXPRESSION_RE.search(value))


def _is_larger_runner(label: str) -> bool:
    """Fail-closed classification of a concrete (non-expression) label."""
    return label.strip() not in STANDARD_HOSTED_LABELS


def _embedded_larger_runner_literals(value: str) -> set[str]:
    """Larger-runner labels hardcoded as string literals inside an expression.

    ``runs-on: ${{ <something-else> && 'ubuntu-latest-96-core' || ... }}`` is
    still an unconditional larger-runner request as far as a fork is
    concerned, so the expression branch must not be a blanket pass.

    Inside an expression a quoted string can be any operand (``'push'``,
    ``'refs/heads/main'``), so only operands SHAPED like a hosted runner
    label — ``<os>-...`` for the three hosted operating systems — are
    considered. Bare (non-expression) ``runs-on`` values stay fully
    fail-closed: anything not in ``STANDARD_HOSTED_LABELS`` is a larger
    runner there, self-hosted pool names included.
    """
    found: set[str] = set()
    for expr in _EXPRESSION_RE.findall(value):
        for literal in _STRING_LITERAL_RE.findall(expr):
            if not _RUNNER_LABEL_SHAPE_RE.match(literal):
                continue
            if _is_larger_runner(literal):
                found.add(literal)
    return found


def _guarded_expression(value: str) -> bool:
    """True when the expression itself routes forks away from the big pool."""
    return UPSTREAM_GUARD in value


def _runs_on_labels(runs_on) -> list[str]:
    """Normalize a ``runs-on`` value into a list of label strings.

    ``runs-on`` may be a string, a list of labels, or a mapping with
    ``group``/``labels`` keys.
    """
    if runs_on is None:
        return []
    if isinstance(runs_on, str):
        return [runs_on]
    if isinstance(runs_on, list):
        return [str(item) for item in runs_on]
    if isinstance(runs_on, dict):
        labels: list[str] = []
        group = runs_on.get("group")
        if group is not None:
            labels.append(str(group))
        raw = runs_on.get("labels")
        if isinstance(raw, str):
            labels.append(raw)
        elif isinstance(raw, list):
            labels.extend(str(item) for item in raw)
        return labels
    return [str(runs_on)]


def _matrix_includes(job: dict) -> list[dict]:
    strategy = job.get("strategy")
    if not isinstance(strategy, dict):
        return []
    matrix = strategy.get("matrix")
    if not isinstance(matrix, dict):
        return []
    include = matrix.get("include")
    if not isinstance(include, list):
        return []
    return [entry for entry in include if isinstance(entry, dict)]


def _candidate_labels(job: dict) -> list[str]:
    """Every runner string this job can actually resolve ``runs-on`` to.

    Direct labels come from ``runs-on``. When ``runs-on`` interpolates a
    matrix key (``${{ matrix.runner }}``), the values of that key in every
    ``include`` entry are runner strings too.
    """
    labels: list[str] = []
    matrix_keys: set[str] = set()

    for value in _runs_on_labels(job.get("runs-on")):
        for expr in _EXPRESSION_RE.findall(value):
            matrix_keys.update(_MATRIX_REF_RE.findall(expr))
        labels.append(value)

    if matrix_keys:
        for entry in _matrix_includes(job):
            for key in matrix_keys:
                if key in entry:
                    labels.append(str(entry[key]))

    return labels


def _job_skipped_on_fork(job: dict) -> bool:
    condition = job.get("if")
    return isinstance(condition, str) and UPSTREAM_GUARD in condition


def _violations() -> list[str]:
    problems: list[str] = []

    for path in _workflow_files():
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        jobs = data.get("jobs")
        if not isinstance(jobs, dict):
            continue

        for job_name, job in jobs.items():
            if not isinstance(job, dict):
                continue
            # Reusable-workflow calls have no runner of their own.
            if "uses" in job and "runs-on" not in job:
                continue

            skipped_on_fork = _job_skipped_on_fork(job)

            for label in _candidate_labels(job):
                if _is_expression(label):
                    # An expression is not a bare label request, but it can
                    # still HARDCODE a larger-runner label behind some other
                    # condition. Accept it only when it either carries the
                    # upstream repo check, or embeds no larger-runner literal
                    # at all (matrix refs are resolved from `include` above,
                    # workflow inputs are the caller's choice).
                    if _guarded_expression(label) or skipped_on_fork:
                        continue
                    embedded = _embedded_larger_runner_literals(label)
                    if not embedded:
                        continue
                    problems.append(
                        f"{path.name}: job '{job_name}' hardcodes larger "
                        f"runner(s) {sorted(embedded)} inside a runs-on "
                        f"expression that does not check "
                        f"`{UPSTREAM_GUARD}`"
                    )
                    continue
                if not _is_larger_runner(label):
                    continue
                if skipped_on_fork:
                    continue
                problems.append(
                    f"{path.name}: job '{job_name}' requests larger runner "
                    f"'{label}' without a "
                    f"`{UPSTREAM_GUARD}` guard on runs-on or on the job's `if:`"
                )

    return problems


def test_no_unconditional_larger_runner_on_forks():
    problems = _violations()
    assert not problems, (
        "Larger-runner jobs queue forever on forks (no larger-runner pool).\n"
        "Make runs-on repository-conditional, e.g.\n"
        "  runs-on: ${{ " + UPSTREAM_GUARD + " && 'ubuntu-latest-32-core' "
        "|| 'ubuntu-latest' }}\n"
        "or gate the whole job with `if: " + UPSTREAM_GUARD + "`.\n\n"
        + "\n".join(problems)
    )


@pytest.mark.parametrize(
    "label",
    [
        "ubuntu-latest-96-core",
        "ubuntu-latest-32-core",
        "ubuntu-latest-32-arm-core",
        "windows-latest-32-core",
        # An unknown / future label must classify as a larger runner, never
        # silently as standard.
        "ubuntu-latest-256-core",
        "some-self-hosted-pool",
    ],
)
def test_unknown_labels_classify_as_larger_runner(label):
    assert _is_larger_runner(label)


@pytest.mark.parametrize("label", sorted(STANDARD_HOSTED_LABELS))
def test_standard_labels_classify_as_standard(label):
    assert not _is_larger_runner(label)


def test_guard_detects_a_bare_larger_runner_label():
    """The classifier is what makes a regression visible — prove it bites."""
    job = {"runs-on": "${{ matrix.runner }}", "strategy": {
        "matrix": {"include": [{"runner": "ubuntu-latest-32-core"}]}}}
    labels = _candidate_labels(job)
    assert "ubuntu-latest-32-core" in labels
    assert any(_is_larger_runner(lbl) for lbl in labels)
    assert not _job_skipped_on_fork(job)


def test_repository_conditional_expression_is_accepted():
    expr = (
        "${{ github.repository == 'NousResearch/hermes-agent' "
        "&& 'ubuntu-latest-32-core' || 'ubuntu-latest' }}"
    )
    assert _is_expression(expr)
    assert _guarded_expression(expr)
    assert _embedded_larger_runner_literals(expr) == {"ubuntu-latest-32-core"}


def test_expression_hardcoding_a_larger_runner_without_the_repo_check_is_caught():
    """An expression is not an escape hatch: a hardcoded big label still counts."""
    expr = "${{ github.event_name == 'push' && 'ubuntu-latest-96-core' || 'ubuntu-latest' }}"
    assert _is_expression(expr)
    assert not _guarded_expression(expr)
    assert _embedded_larger_runner_literals(expr) == {"ubuntu-latest-96-core"}
    # A pure matrix/input reference embeds no literal and stays allowed.
    assert _embedded_larger_runner_literals("${{ matrix.runner }}") == set()
    assert _embedded_larger_runner_literals("${{ inputs.runner }}") == set()
