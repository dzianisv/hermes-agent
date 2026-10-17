"""Attribution baseline of the contributor check.

``contributor-check.yml`` flags commit author emails that have no mapping. It
compared against ``origin/main`` only, so a pinned dependency snapshot cut from
an older base was charged with the authors already inside that base. The
reusable workflow now takes an optional ``comparison_base`` input (default
``origin/main``) and ``ci.yaml`` passes the exact pin only for pushes to this
fork's reserved ``agentpod/dependency-ci/`` branches. The contracts:

* the default baseline is still ``origin/main`` and still rejects an unmapped
  author that sits between the fork point and the tip;
* with the pin as the baseline the pin's own history is not scanned, but every
  commit added after it is, with the same validation as before;
* an unresolvable, malformed or unrelated baseline fails the check, never skips
  it, never degrades to a successful empty or full-history scan, and its text is
  never executed or taken for a git option;
* only a push to the reserved namespace of this fork selects the pin.

The behaviour tests run the workflow's real ``run:`` step against tiny
throwaway Git repositories built from synthetic emails.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

FORK_REPO = "dzianisv/hermes-agent"
SNAPSHOT_REF_PREFIX = "refs/heads/agentpod/dependency-ci/"
PIN = "47fc6a99ddfc2272e31308d312e0fd01f264b961"
DEFAULT_BASE = "origin/main"
INPUT_EXPR = "${{ inputs.comparison_base }}"
COPILOT_TRAILER = "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"

MAIN_AUTHOR = "main-author@example.invalid"
PIN_AUTHOR = "pin-author@unmapped.example.invalid"
NEW_NOREPLY = "4242+snapshot-dev@users.noreply.github.com"
NEW_UNMAPPED = "late-author@unmapped.example.invalid"
FILE_MAPPED = "file-mapped@example.invalid"
RELEASE_MAPPED = "release-mapped@example.invalid"


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(doc: dict) -> dict:
    # PyYAML reads the bare key ``on`` as the boolean True.
    return doc["on"] if "on" in doc else doc[True]


def _check_step() -> dict:
    steps = _workflow("contributor-check.yml")["jobs"]["check-attribution"]["steps"]
    [step] = [s for s in steps if s.get("id") == "check-emails"]
    return step


def _caller_job() -> dict:
    return _workflow("ci.yaml")["jobs"]["contributor-check"]


# --------------------------------------------------------------------------
# ci.yaml: which baseline each event gets
# --------------------------------------------------------------------------


def _literal(token: str) -> str:
    assert re.fullmatch(r"'[^']*'", token), f"not a string literal: {token!r}"
    return token[1:-1]


def render_base(expr: str, *, event: str, repository: str, ref: str) -> str:
    """Evaluate ``c1 && c2 && ... && 'A' || 'B'``; reject any other shape."""
    whole = re.fullmatch(r"\$\{\{\s*(.+?)\s*\}\}", expr.strip())
    assert whole, f"not a single expression: {expr!r}"
    chain, sep, fallback = whole[1].partition(" || ")
    assert sep and " || " not in fallback, f"expected one '||': {expr!r}"
    *conditions, chosen_token = chain.split(" && ")
    chosen, other = _literal(chosen_token), _literal(fallback)
    # GitHub treats '' as false, so the value after the last && must be truthy.
    assert chosen, f"falsy value after &&: {expr!r}"
    context = {
        "github.event_name": event,
        "github.repository": repository,
        "github.ref": ref,
    }

    def holds(condition: str) -> bool:
        # GitHub compares strings case-insensitively in == and startsWith.
        equal = re.fullmatch(r"(github\.\w+) == '([^']*)'", condition)
        if equal:
            return context[equal[1]].lower() == equal[2].lower()
        starts = re.fullmatch(r"startsWith\((github\.\w+), '([^']*)'\)", condition)
        if starts:
            return context[starts[1]].lower().startswith(starts[2].lower())
        raise AssertionError(f"unsupported condition: {condition!r}")

    return chosen if all(holds(c) for c in conditions) else other


SNAPSHOT_REF = SNAPSHOT_REF_PREFIX + "browser-link-refresh-20261017"

BASE_SELECTION = [
    # (event, repository, ref, expected)
    ("push", FORK_REPO, SNAPSHOT_REF, PIN),
    ("push", FORK_REPO, SNAPSHOT_REF_PREFIX + "nested/branch", PIN),
    ("push", FORK_REPO, "refs/heads/main", DEFAULT_BASE),
    ("push", FORK_REPO, "refs/heads/agentpod/pin-a68be41d", DEFAULT_BASE),
    ("push", FORK_REPO, "refs/heads/agentpod/dependency-ci", DEFAULT_BASE),
    ("push", FORK_REPO, "refs/heads/agentpod/dependency-cifoo/x", DEFAULT_BASE),
    ("push", FORK_REPO, "refs/heads/feature/agentpod/dependency-ci/x", DEFAULT_BASE),
    ("push", FORK_REPO, "refs/tags/agentpod/dependency-ci/v1", DEFAULT_BASE),
    ("push", "NousResearch/hermes-agent", SNAPSHOT_REF, DEFAULT_BASE),
    ("push", "someone-else/hermes-agent", SNAPSHOT_REF, DEFAULT_BASE),
    ("pull_request", FORK_REPO, "refs/pull/7/merge", DEFAULT_BASE),
    ("pull_request", FORK_REPO, SNAPSHOT_REF, DEFAULT_BASE),
    ("pull_request", "NousResearch/hermes-agent", "refs/pull/7/merge", DEFAULT_BASE),
    ("workflow_dispatch", FORK_REPO, SNAPSHOT_REF, DEFAULT_BASE),
    ("schedule", FORK_REPO, "refs/heads/main", DEFAULT_BASE),
]


class TestCallerSelectsBaseline:
    def test_pin_is_a_full_commit_sha(self):
        assert re.fullmatch(r"[0-9a-f]{40}", PIN)

    @pytest.mark.parametrize("event,repository,ref,expected", BASE_SELECTION)
    def test_baseline_by_event(self, event, repository, ref, expected):
        expr = _caller_job()["with"]["comparison_base"]
        assert render_base(expr, event=event, repository=repository, ref=ref) == expected

    def test_only_the_reserved_fork_push_selects_the_pin(self):
        expr = _caller_job()["with"]["comparison_base"]
        selected = {
            (e, r, f)
            for e, r, f, _ in BASE_SELECTION
            if render_base(expr, event=e, repository=r, ref=f) != DEFAULT_BASE
        }
        assert selected == {
            ("push", FORK_REPO, SNAPSHOT_REF),
            ("push", FORK_REPO, SNAPSHOT_REF_PREFIX + "nested/branch"),
        }

    def test_the_check_stays_scheduled_and_required(self):
        job = _caller_job()
        assert job["uses"] == "./.github/workflows/contributor-check.yml"
        assert job["needs"] == "detect"
        assert job["if"] == "needs.detect.outputs.python == 'true'"
        assert set(job["with"]) == {"comparison_base"}
        assert "continue-on-error" not in job
        gate = _workflow("ci.yaml")["jobs"]["all-checks-pass"]
        assert "contributor-check" in gate["needs"]

    def test_ci_yaml_is_the_only_caller(self):
        callers = [
            path.name
            for path in sorted(WORKFLOWS.glob("*.y*ml"))
            if "contributor-check.yml" in path.read_text(encoding="utf-8")
            and path.name != "contributor-check.yml"
        ]
        assert callers == ["ci.yaml"]


class TestReusableWorkflowContract:
    def test_input_is_optional_and_defaults_to_origin_main(self):
        declared = _triggers(_workflow("contributor-check.yml"))["workflow_call"]
        spec = declared["inputs"]["comparison_base"]
        assert spec["type"] == "string"
        assert spec["required"] is False
        assert spec["default"] == DEFAULT_BASE
        assert "review_status" in declared["outputs"]

    def test_step_reads_the_input_through_the_environment_only(self):
        step = _check_step()
        assert step["env"] == {"ATTRIBUTION_BASE": INPUT_EXPR}
        # No expression is spliced into shell text.
        assert "${{" not in step["run"]
        assert 'git merge-base -- "$ATTRIBUTION_BASE" HEAD' in step["run"]
        code = "\n".join(
            line for line in step["run"].splitlines() if not line.lstrip().startswith("#")
        )
        assert "origin/main" not in code


# --------------------------------------------------------------------------
# Real step against tiny Git repositories
# --------------------------------------------------------------------------


class Repo:
    def __init__(self, root: Path):
        self.root = root
        self.scratch = root.parent
        self.home = self.scratch / "home"
        self.home.mkdir(parents=True)
        root.mkdir()
        self._n = itertools.count(1)
        self.env = {
            "PATH": os.environ["PATH"],
            "HOME": str(self.home),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str, **extra_env: str) -> str:
        done = subprocess.run(
            ["git", *args],
            cwd=self.root,
            env={**self.env, **extra_env},
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return done.stdout.strip()

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def commit(self, email: str, subject: str) -> str:
        self.write(f"changes/{next(self._n)}.txt", subject + "\n")
        self.git("add", "-A")
        self.git(
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            f"{subject}\n\n{COPILOT_TRAILER}",
            GIT_AUTHOR_NAME="Synthetic Author",
            GIT_AUTHOR_EMAIL=email,
            GIT_COMMITTER_NAME="Synthetic Author",
            GIT_COMMITTER_EMAIL=email,
        )
        return self.git("rev-parse", "HEAD")


class Snapshot(NamedTuple):
    repo: Repo
    pin: str
    head: str


def build_snapshot(
    tmp_path: Path,
    new_authors: tuple[str, ...] = (NEW_NOREPLY,),
    *,
    file_mapped: tuple[str, ...] = (),
    all_mapped: bool = False,
) -> Snapshot:
    """A pinned dependency snapshot cut from an older base than origin/main.

    main:      M0 ---------------- M1   (= origin/main)
    snapshot:   \\-- P1 -- P2(pin) -- new commits (= HEAD)

    ``all_mapped`` gives every commit a mapped author, so a scan of the whole
    history would succeed: it shows a bad baseline is rejected by the baseline
    check itself, not by an unmapped author that happens to be in range.
    """
    main_author, pin_author = (NEW_NOREPLY, NEW_NOREPLY) if all_mapped else (MAIN_AUTHOR, PIN_AUTHOR)
    repo = Repo(tmp_path / "repo")
    repo.write("scripts/release.py", f'AUTHOR_MAP = {{\n    "{RELEASE_MAPPED}": "release-mapped",\n}}\n')
    for email in file_mapped:
        repo.write(f"contributors/emails/{email}", "file-mapped\n")
    repo.commit(main_author, "main base")
    repo.git("checkout", "-q", "-b", "snapshot")
    repo.commit(pin_author, "pin history 1")
    pin = repo.commit(pin_author, "pin history 2")
    head = pin
    for number, email in enumerate(new_authors, 1):
        head = repo.commit(email, f"snapshot change {number}")
    repo.git("checkout", "-q", "main")
    repo.commit(main_author, "main moved on")
    repo.git("update-ref", "refs/remotes/origin/main", "HEAD")
    repo.git("checkout", "-q", "snapshot")
    return Snapshot(repo, pin, head)


class Outcome(NamedTuple):
    returncode: int
    output: str
    review_status: list | None


def run_check(repo: Repo, comparison_base: str | None) -> Outcome:
    """Run the workflow's own ``check-emails`` step the way Actions would.

    ``comparison_base=None`` models a caller that omits the input, which takes
    the default declared by the reusable workflow. Like Actions, the script text
    has its ``${{ inputs.comparison_base }}`` expressions replaced by the input
    value before bash sees it, so a step that splices the input into shell text
    would run it here too.
    """
    step = _check_step()
    assert step["env"]["ATTRIBUTION_BASE"] == INPUT_EXPR
    if comparison_base is None:
        declared = _triggers(_workflow("contributor-check.yml"))["workflow_call"]
        comparison_base = declared["inputs"]["comparison_base"]["default"]
    script = repo.scratch / "step.sh"
    script.write_text(step["run"].replace(INPUT_EXPR, comparison_base), encoding="utf-8")
    github_output = repo.scratch / "github_output"
    github_output.write_text("", encoding="utf-8")
    done = subprocess.run(
        ["bash", "-e", str(script)],
        cwd=repo.root,
        env={
            **repo.env,
            "ATTRIBUTION_BASE": comparison_base,
            "GITHUB_OUTPUT": str(github_output),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    status = None
    for line in github_output.read_text(encoding="utf-8").splitlines():
        if line.startswith("review_status="):
            status = json.loads(line.split("=", 1)[1])
    return Outcome(done.returncode, done.stdout + done.stderr, status)


def _grep_supports_pcre() -> bool:
    probe = subprocess.run(
        ["grep", "-qP", "x", os.devnull], capture_output=True, timeout=30
    )
    return probe.returncode in (0, 1)


@pytest.fixture
def step_tools():
    for tool in ("git", "jq", "bash"):
        assert shutil.which(tool), f"the contributor-check step needs {tool}"
    if not _grep_supports_pcre():
        if sys.platform.startswith("linux"):
            pytest.fail("the contributor-check step needs GNU grep (-P)")
        pytest.skip("the workflow step uses grep -P, which this platform lacks")


def _flagged(outcome: Outcome) -> str:
    assert outcome.review_status, "expected an action_required review status"
    [source] = outcome.review_status
    [result] = source["results"]
    assert result["kind"] == "action_required"
    return result["detail"]


CANARY = "CANARY_PATH"

# Every value is a comparison_base the step must reject. CANARY_PATH is replaced
# by a path the test owns; the shell must never create it.
MALFORMED_BASES = {
    # absent from the repository
    "missing-branch": "refs/heads/does-not-exist",
    "missing-remote-branch": "origin/no-such-branch",
    "missing-full-sha": "deadbeef" * 5,
    "missing-abbreviated-sha": "deadbeef",
    "real-pin-sha-absent-from-this-repo": PIN,
    "beyond-the-history": "HEAD~99",
    "empty-reflog-entry": "HEAD@{99}",
    # not a ref
    "empty": "",
    "whitespace": "   ",
    "two-refs-in-one-argument": "origin/main HEAD",
    "newline-separated": "origin/main\nHEAD",
    "newline-then-workflow-command": "origin/main\n::error::injected",
    "dots": "..",
    "tilde": "~",
    "colon": ":",
    "backslash": "origin/main\\",
    "glob": "origin/*",
    "unbalanced-peel": "origin/main^{commit",
    "range": "HEAD..origin/main",
    "symmetric-range": "HEAD...origin/main",
    "negated": "^HEAD",
    "parent-shorthand-range": "origin/main^!",
    # an object that is not a commit
    "tree": "snapshot^{tree}",
    "blob": "HEAD:scripts/release.py",
    # option text
    "option-all": "--all",
    "option-help": "--help",
    "option-short-help": "-h",
    "option-octopus": "--octopus",
    "option-is-ancestor": "--is-ancestor",
    "option-fork-point": "--fork-point",
    # shell and expression text
    "command-substitution": "$(touch CANARY_PATH)",
    "backticks": "`touch CANARY_PATH`",
    "command-after-semicolon": "origin/main; touch CANARY_PATH",
    "command-after-and": "origin/main && touch CANARY_PATH",
    "command-after-pipe": "origin/main | touch CANARY_PATH",
    "command-before-ref": "$(touch CANARY_PATH) origin/main",
    "single-quote-breakout": "'; touch CANARY_PATH; '",
    "double-quote-breakout": '"; touch CANARY_PATH; "',
    "command-on-next-line": "origin/main\ntouch CANARY_PATH",
    "shell-variable-text": "${ATTRIBUTION_BASE}",
    "actions-secret-expression": "${{ secrets.ANY }}",
}

# A cross-section that must fail on a history where a scan would otherwise pass.
CLEAN_SCAN_PROBES = [
    "missing-branch",
    "empty",
    "option-all",
    "tree",
    "two-refs-in-one-argument",
    "command-substitution",
]


@pytest.mark.usefixtures("step_tools")
class TestStepAgainstGit:
    def test_fixture_commits_carry_the_copilot_trailer(self, tmp_path):
        repo = build_snapshot(tmp_path, (NEW_NOREPLY, NEW_UNMAPPED)).repo
        commits = int(repo.git("rev-list", "--count", "--all"))
        messages = repo.git("log", "--all", "--format=%B")
        assert commits == 6
        assert messages.count(COPILOT_TRAILER) == commits

    def test_default_baseline_is_origin_main_and_rejects_the_unmapped_pin_history(
        self, tmp_path
    ):
        snap = build_snapshot(tmp_path)
        outcome = run_check(snap.repo, None)
        assert outcome.returncode == 1
        assert PIN_AUTHOR in _flagged(outcome)
        assert NEW_NOREPLY not in _flagged(outcome)
        assert "without a mapping" in outcome.output

    def test_explicit_origin_main_behaves_like_the_default(self, tmp_path):
        snap = build_snapshot(tmp_path)
        assert run_check(snap.repo, DEFAULT_BASE).returncode == 1

    def test_snapshot_base_excludes_the_pin_and_passes_a_new_noreply_commit(
        self, tmp_path
    ):
        snap = build_snapshot(tmp_path)
        outcome = run_check(snap.repo, snap.pin)
        assert outcome.returncode == 0, outcome.output
        assert "All contributor emails are mapped" in outcome.output
        assert outcome.review_status == []

    def test_snapshot_base_with_no_new_commit_has_nothing_to_check(self, tmp_path):
        snap = build_snapshot(tmp_path, new_authors=())
        outcome = run_check(snap.repo, snap.pin)
        assert outcome.returncode == 0, outcome.output
        assert "No new commits to check" in outcome.output

    def test_a_new_unmapped_commit_after_the_pin_still_fails(self, tmp_path):
        snap = build_snapshot(tmp_path, (NEW_NOREPLY, NEW_UNMAPPED))
        outcome = run_check(snap.repo, snap.pin)
        assert outcome.returncode == 1
        detail = _flagged(outcome)
        assert NEW_UNMAPPED in detail
        assert PIN_AUTHOR not in detail
        assert NEW_NOREPLY not in detail

    @pytest.mark.parametrize(
        "author,file_mapped",
        [
            (NEW_NOREPLY, ()),
            (FILE_MAPPED, (FILE_MAPPED,)),
            (RELEASE_MAPPED, ()),
        ],
        ids=["github-noreply", "contributors-file", "release-author-map"],
    )
    def test_mapped_new_authors_pass(self, tmp_path, author, file_mapped):
        snap = build_snapshot(tmp_path, (author,), file_mapped=file_mapped)
        outcome = run_check(snap.repo, snap.pin)
        assert outcome.returncode == 0, outcome.output
        assert outcome.review_status == []

    def test_a_new_author_is_scanned_not_assumed_mapped(self, tmp_path):
        # Same commit as the contributors-file case, without the mapping file.
        snap = build_snapshot(tmp_path, (FILE_MAPPED,))
        outcome = run_check(snap.repo, snap.pin)
        assert outcome.returncode == 1
        assert FILE_MAPPED in _flagged(outcome)

    @pytest.mark.parametrize("base", ["", "no-such-ref", "0" * 40])
    def test_an_unresolvable_base_fails_instead_of_skipping(self, tmp_path, base):
        snap = build_snapshot(tmp_path)
        outcome = run_check(snap.repo, base)
        assert outcome.returncode != 0
        assert outcome.review_status is None
        assert "refusing to skip the attribution check" in outcome.output
        assert "All contributor emails are mapped" not in outcome.output
        assert "No new commits" not in outcome.output

    def test_an_unrelated_base_fails_instead_of_skipping(self, tmp_path):
        snap = build_snapshot(tmp_path)
        repo = snap.repo
        repo.git("checkout", "-q", "--orphan", "unrelated")
        unrelated = repo.commit(MAIN_AUTHOR, "unrelated root")
        repo.git("checkout", "-q", "snapshot")
        outcome = run_check(repo, unrelated)
        assert outcome.returncode != 0
        assert outcome.review_status is None
        assert "refusing to skip the attribution check" in outcome.output

    @pytest.mark.parametrize("name", sorted(MALFORMED_BASES))
    def test_a_malformed_base_fails_closed_and_runs_nothing(self, tmp_path, name):
        canary = tmp_path / "executed"
        base = MALFORMED_BASES[name].replace(CANARY, shlex.quote(str(canary)))
        snap = build_snapshot(tmp_path)
        outcome = run_check(snap.repo, base)
        assert not canary.exists(), "the base text was run as shell"
        assert outcome.returncode != 0
        assert outcome.review_status is None
        assert "refusing to skip the attribution check" in outcome.output
        assert "All contributor emails are mapped" not in outcome.output
        assert "No new commits" not in outcome.output

    @pytest.mark.parametrize("name", CLEAN_SCAN_PROBES)
    def test_a_malformed_base_is_not_rescued_by_a_clean_history(self, tmp_path, name):
        # Every author is mapped, so scanning the whole history would pass.
        # A bad baseline must still be rejected, not turned into that scan.
        canary = tmp_path / "executed"
        base = MALFORMED_BASES[name].replace(CANARY, shlex.quote(str(canary)))
        snap = build_snapshot(tmp_path, all_mapped=True)
        for good in (DEFAULT_BASE, snap.pin):
            passing = run_check(snap.repo, good)
            assert passing.returncode == 0, passing.output
            assert passing.review_status == []
        outcome = run_check(snap.repo, base)
        assert not canary.exists(), "the base text was run as shell"
        assert outcome.returncode != 0
        assert outcome.review_status is None
        assert "refusing to skip the attribution check" in outcome.output
        assert "All contributor emails are mapped" not in outcome.output

    def test_a_new_unmapped_commit_fails_whichever_baseline_is_given(self, tmp_path):
        # The same history, judged against each accepted baseline, with one
        # unmapped commit added after the pin.
        snap = build_snapshot(tmp_path, (NEW_NOREPLY, NEW_UNMAPPED, NEW_NOREPLY))
        for base in (snap.pin, DEFAULT_BASE):
            outcome = run_check(snap.repo, base)
            assert outcome.returncode == 1, outcome.output
            assert NEW_UNMAPPED in _flagged(outcome)
