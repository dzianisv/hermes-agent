"""Runner selection in the workflows that ask for org-only larger runners.

``tests.yml``, ``tests-os.yml``, ``js-tests.yml``, ``rust-tests.yml`` and
``nix.yml`` pick runners that only the upstream organisation has (96-core and
32-core Linux, 32-core Windows). A fork has none of them, so those jobs would
queue forever and hold the PR open. Each workflow therefore chooses its runner,
timeout and worker budget from ``github.repository_owner``. The contracts:

* the upstream owner renders exactly the original values;
* any other owner renders hosted runners and a budget sized for 4 cores;
* every lane stays scheduled and the review gate is untouched;
* no other org-only runner label exists unless it is unreachable on a fork.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
UPSTREAM = "NousResearch"
FORK = "dzianisv"
OWNERS = [UPSTREAM, FORK]

_EXPR = re.compile(
    r"\$\{\{\s*github\.repository_owner\s*(?P<op>==|!=)\s*'(?P<owner>[^']+)'"
    r"\s*&&\s*(?P<a>'[^']*'|\d+)\s*\|\|\s*(?P<b>'[^']*'|\d+)\s*\}\}"
)
_ORG_ONLY_RUNNER = re.compile(r"-\d+(-arm)?-core$")

# (workflow, job) -> (upstream runner, upstream timeout, fork timeout)
LANES = {
    ("tests.yml", "test"): ("ubuntu-latest-96-core", 30, 90),
    ("js-tests.yml", "check"): ("ubuntu-latest-32-core", 30, 60),
    ("rust-tests.yml", "bootstrap-installer"): ("ubuntu-latest-32-core", 30, 60),
    ("nix.yml", "flake-check"): ("ubuntu-latest-32-core", 60, 120),
}

# Org-only runners that a fork never reaches, and why (asserted below).
UNREACHABLE_ON_FORKS = {
    ("docker.yml", "build"),
    ("docker.yml", "publish"),
    ("e2e-desktop.yml", "e2e"),
}


def _literal(token: str):
    return int(token) if token.isdigit() else token[1:-1]


def _select(match: re.Match, owner: str):
    chosen, other = _literal(match["a"]), _literal(match["b"])
    # GitHub treats '' and 0 as false, so `cond && '' || other` always yields
    # `other`. The value after `&&` must therefore be truthy.
    assert chosen not in ("", 0), f"falsy value after &&: {match[0]!r}"
    same = owner == match["owner"]
    return chosen if (same if match["op"] == "==" else not same) else other


def render(value, owner: str):
    """Evaluate ``owner ==|!= X && A || B``; reject any other expression."""
    if not isinstance(value, str):
        return value
    whole = _EXPR.fullmatch(value.strip())
    if whole:
        return _select(whole, owner)
    out = _EXPR.sub(lambda m: str(_select(m, owner)), value)
    assert "${{" not in out, f"unsupported expression: {value!r}"
    return out


_CACHE: dict[str, dict] = {}


def _load(name: str) -> dict:
    if name not in _CACHE:
        _CACHE[name] = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    return _CACHE[name]


def _job(workflow: str, job: str) -> dict:
    return _load(workflow)["jobs"][job]


def _runner_values(job: dict, owner: str) -> list[str]:
    runs_on = job.get("runs-on")
    if not isinstance(runs_on, str):
        return []
    if runs_on.strip() == "${{ matrix.runner }}":
        matrix = job["strategy"]["matrix"]
        rows = [row["runner"] for row in matrix.get("include", []) if "runner" in row]
        return [render(value, owner) for value in rows + list(matrix.get("runner", []))]
    if "${{" in runs_on and not _EXPR.fullmatch(runs_on.strip()):
        return []  # supplied by a caller input, not a literal label
    return [render(runs_on, owner)]


def _step(job: dict, name: str) -> dict:
    (step,) = [s for s in job["steps"] if s.get("name") == name]
    return step


@pytest.mark.parametrize("lane", LANES)
def test_upstream_owner_keeps_original_values(lane):
    runner, timeout, _ = LANES[lane]
    job = _job(*lane)
    assert render(job["runs-on"], UPSTREAM) == runner
    assert render(job["timeout-minutes"], UPSTREAM) == timeout


@pytest.mark.parametrize("lane", LANES)
def test_fork_owner_uses_hosted_runner_with_longer_timeout(lane):
    _, _, timeout = LANES[lane]
    job = _job(*lane)
    assert render(job["runs-on"], FORK) == "ubuntu-latest"
    assert render(job["timeout-minutes"], FORK) == timeout


def test_python_worker_budget():
    env = _step(_job("tests.yml", "test"), "Run tests")["env"]
    assert str(render(env["HERMES_TEST_WORKERS"], UPSTREAM)) == "96"
    assert str(render(env["HERMES_TEST_WORKERS"], FORK)) == "4"


def test_js_concurrency_budget():
    run = _step(_job("js-tests.yml", "check"), "Run all workspace checks")["run"]
    original = "node .github/scripts/run-workspace-checks.mjs"
    assert render(run, UPSTREAM) == original
    assert render(run, FORK) == original + " --concurrency 2"


def test_os_lane_runners():
    job = _job("tests-os.yml", "os-tests")
    assert _runner_values(job, UPSTREAM) == ["macos-latest", "windows-latest-32-core"]
    assert _runner_values(job, FORK) == ["macos-latest", "windows-latest"]
    for owner in OWNERS:
        assert render(job["timeout-minutes"], owner) == 30


@pytest.mark.parametrize("lane", [*LANES, ("tests-os.yml", "os-tests")])
def test_fork_gets_no_org_only_runner(lane):
    runners = _runner_values(_job(*lane), FORK)
    assert runners and not [r for r in runners if _ORG_ONLY_RUNNER.search(r)]


@pytest.mark.parametrize("owner", OWNERS)
def test_every_lane_is_still_scheduled(owner):
    for lane in [("tests.yml", "test"), ("js-tests.yml", "check"), ("rust-tests.yml", "bootstrap-installer"), ("tests-os.yml", "os-tests")]:
        assert "if" not in _job(*lane) and "needs" not in _job(*lane), lane
    flake = _job("nix.yml", "flake-check")
    assert flake["needs"] == ["detect"]
    assert flake["if"] == "needs.detect.outputs.nix == 'true'"
    os_job = _job("tests-os.yml", "os-tests")
    assert [(r["marker"], r["name"]) for r in os_job["strategy"]["matrix"]["include"]] == [
        ("macos_only", "macOS-only tests"),
        ("windows_only", "Windows-only tests"),
    ]
    assert os_job["strategy"]["fail-fast"] is False
    e2e = _job("tests.yml", "e2e")
    assert render(e2e["runs-on"], owner) == "ubuntu-latest"
    assert render(e2e["timeout-minutes"], owner) == 15


def test_review_gate_and_lane_gating_are_untouched():
    jobs = _load("ci.yaml")["jobs"]
    assert jobs["js-tests"]["if"] == "needs.detect.outputs.frontend == 'true'"
    assert jobs["rust-tests"]["if"] == "needs.detect.outputs.rust == 'true'"
    assert "needs.detect.outputs.ci_review == 'true'" in jobs["review-labels"]["if"]
    assert {"tests", "tests-os", "js-tests", "rust-tests", "review-labels"} <= set(
        jobs["all-checks-pass"]["needs"]
    )
    gate = _job("review-labels.yml", "check")
    fail = _step(gate, "Fail on missing label")
    assert fail["if"] == "steps.label-check.outputs.ci_reviewed != 'true'"
    assert "exit 1" in fail["run"]


def test_every_org_only_runner_is_adapted_or_unreachable_on_forks():
    """A new `-N-core` label must be adapted here or documented as unreachable."""
    found = set()
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        for name, job in (_load(path.name).get("jobs") or {}).items():
            if any(_ORG_ONLY_RUNNER.search(r) for r in _runner_values(job, FORK)):
                found.add((path.name, name))
    assert found == UNREACHABLE_ON_FORKS

    for name in ("build", "publish"):
        assert "github.repository == 'NousResearch/hermes-agent'" in _job("docker.yml", name)["if"]
    assert _load("ci.yaml")["jobs"]["e2e-desktop"]["if"] is False


def test_dependency_push_preserves_main_and_pull_request_entry_points():
    definition = _load("ci.yaml")
    events = definition.get("on", definition.get(True))
    assert isinstance(events, dict)
    assert set(events) == {"pull_request", "push"}
    assert events["pull_request"] is None
    assert events["push"] == {"branches": ["main", "agentpod/dependency-ci/**"]}
    assert definition["permissions"] == {
        "contents": "read",
        "pull-requests": "write",
    }
