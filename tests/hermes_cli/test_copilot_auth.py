"""Tests for hermes_cli.copilot_auth — Copilot token validation and resolution."""

import pytest
from unittest.mock import patch


class TestTokenValidation:
    """Token type validation."""

    def test_classic_pat_rejected(self):
        from hermes_cli.copilot_auth import validate_copilot_token
        valid, msg = validate_copilot_token("ghp_abcdefghijklmnop1234")
        assert valid is False

    @pytest.mark.parametrize("token", ["gho_abcdefghijklmnop1234", "github_pat_abcdefghijklmnop1234", "ghu_abcdefghijklmnop1234"])
    def test_supported_token_families_accepted(self, token):
        from hermes_cli.copilot_auth import validate_copilot_token
        assert validate_copilot_token(token) == (True, "OK")

    def test_arbitrary_string_rejected(self):
        """A non-GitHub value in GITHUB_TOKEN must fail validation instead of reaching the API (#12650)."""
        from hermes_cli.copilot_auth import validate_copilot_token
        valid, msg = validate_copilot_token("not_a_github_token")
        assert valid is False


class TestResolveToken:
    """Token resolution with env var priority."""


    def test_gh_token_second_priority(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GH_TOKEN", "gho_gh_second")
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")
        token, source = resolve_copilot_token()
        assert token == "gho_gh_second"
        assert source == "GH_TOKEN"


    def test_gh_cli_classic_pat_raises(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value="ghp_classic"):
            with pytest.raises(ValueError):
                resolve_copilot_token()

    def test_invalid_env_var_skips_gh_cli_fallback(self, monkeypatch):
        """When an env var is set but holds an unsupported classic PAT,
        resolve_copilot_token must NOT fall back to ``gh auth token``.

        The user explicitly exported a token; silently substituting one
        from the gh CLI credential store is surprising and the subprocess
        call adds up to 5s of latency on Windows cold starts (#60800).
        Only fall back to the CLI when NO Copilot env var is set at all.
        """
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_classic_pat_nope")
        with patch("hermes_cli.copilot_auth._try_gh_cli_token") as mock_cli:
            token, source = resolve_copilot_token()
        assert token == ""
        assert source == ""
        mock_cli.assert_not_called()

    def test_all_env_vars_invalid_skips_gh_cli_fallback(self, monkeypatch):
        """All three env vars set to classic PATs → no gh CLI call."""
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghp_one")
        monkeypatch.setenv("GH_TOKEN", "ghp_two")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_three")
        with patch("hermes_cli.copilot_auth._try_gh_cli_token") as mock_cli:
            token, source = resolve_copilot_token()
        assert token == ""
        assert source == ""
        mock_cli.assert_not_called()


class TestGhCliTokenCache:
    """The gh-CLI probe result is cached — a miss must not re-spawn gh.

    Regression: /api/model/options ran `gh auth token` four times per build;
    with no gh credential store each probe blocked its full 5s timeout, so the
    Desktop Models/Providers settings pages took 20s per open and exceeded the
    renderer's 15s IPC budget (Aug 2026 desktop audit).
    """

    def _reset(self):
        from hermes_cli.copilot_auth import _invalidate_gh_cli_token_cache
        _invalidate_gh_cli_token_cache()

    def test_miss_is_cached_and_probe_runs_once(self):
        from hermes_cli import copilot_auth
        self._reset()
        with patch.object(copilot_auth, "_probe_gh_cli_token", return_value=None) as probe:
            assert copilot_auth._try_gh_cli_token() is None
            assert copilot_auth._try_gh_cli_token() is None
            assert copilot_auth._try_gh_cli_token() is None
        assert probe.call_count == 1
        self._reset()


    def test_cold_exec_timeout_is_retried_warm(self, monkeypatch):
        """A cold `gh` exec times out, the immediate warm retry succeeds → token, not a miss.

        Measured on macOS 2026-09-27: `gh auth token` after 5 min idle took 19.6s (over the 15s
        probe timeout) while the very next exec took 0.49s. Reading that lone timeout as "no
        credential" is what crashed kanban workers with "No usable credentials found for provider
        'copilot'" while the gh login was perfectly healthy.
        """
        import subprocess

        from hermes_cli import copilot_auth
        self._reset()
        for env_var in copilot_auth.COPILOT_ENV_VARS:
            monkeypatch.delenv(env_var, raising=False)
        monkeypatch.setattr(copilot_auth, "_gh_cli_candidates", lambda: ["/fake/gh"])
        outcomes = iter([
            subprocess.TimeoutExpired(cmd="gh", timeout=15),          # cold exec
            subprocess.CompletedProcess([], 0, stdout="gho_warm\n", stderr=""),  # warm retry
        ])

        def _fake_run(*_args, **_kwargs):
            outcome = next(outcomes)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        monkeypatch.setattr(copilot_auth.subprocess, "run", _fake_run)
        assert copilot_auth.resolve_copilot_token() == ("gho_warm", "gh auth token")
        self._reset()

    def test_ttl_expiry_reprobes(self, monkeypatch):
        from hermes_cli import copilot_auth
        self._reset()
        clock = {"now": 1000.0}
        monkeypatch.setattr(copilot_auth.time, "monotonic", lambda: clock["now"])
        with patch.object(copilot_auth, "_probe_gh_cli_token", return_value=None) as probe:
            copilot_auth._try_gh_cli_token()
            clock["now"] += copilot_auth._GH_CLI_TOKEN_CACHE_TTL_SECONDS + 1
            copilot_auth._try_gh_cli_token()
        assert probe.call_count == 2
        self._reset()

    def test_failed_reprobe_keeps_the_token_already_obtained(self, monkeypatch):
        """Under host load `gh auth token` timed out after a mass resume, and the cached miss made
        every gateway raise "No usable credentials found for provider 'copilot'" in 5-minute
        bursts while the token stayed valid (2026-09-26)."""
        import subprocess

        from hermes_cli import copilot_auth
        self._reset()
        for env_var in copilot_auth.COPILOT_ENV_VARS:
            monkeypatch.delenv(env_var, raising=False)
        clock = {"now": 1000.0}
        monkeypatch.setattr(copilot_auth.time, "monotonic", lambda: clock["now"])
        monkeypatch.setattr(copilot_auth, "_gh_cli_candidates", lambda: ["/fake/gh"])
        outcomes = iter([
            subprocess.CompletedProcess([], 0, stdout="gho_good\n", stderr=""),
            # Each re-probe now makes up to _GH_CLI_PROBE_ATTEMPTS attempts (a cold-exec timeout is
            # retried warm), so a failing re-probe consumes two outcomes.
            subprocess.TimeoutExpired(cmd="gh", timeout=1),
            subprocess.TimeoutExpired(cmd="gh", timeout=1),
            subprocess.CompletedProcess([], 1, stdout="", stderr="error connecting to keyring"),
        ])

        def _fake_run(*_args, **_kwargs):
            outcome = next(outcomes)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        monkeypatch.setattr(copilot_auth.subprocess, "run", _fake_run)
        assert copilot_auth.resolve_copilot_token() == ("gho_good", "gh auth token")
        for _ in range(2):  # a timeout, then a failing exit — each after the cache expired
            clock["now"] += copilot_auth._GH_CLI_TOKEN_CACHE_TTL_SECONDS + 1
            assert copilot_auth.resolve_copilot_token() == ("gho_good", "gh auth token")
        self._reset()

    def test_cold_miss_is_retried_within_a_minute(self, monkeypatch):
        """A process that never saw a token must not stay blind for the full success TTL."""
        from hermes_cli import copilot_auth
        self._reset()
        clock = {"now": 1000.0}
        monkeypatch.setattr(copilot_auth.time, "monotonic", lambda: clock["now"])
        with patch.object(copilot_auth, "_probe_gh_cli_token", side_effect=[None, "gho_late"]):
            assert copilot_auth._try_gh_cli_token() is None
            clock["now"] += 60
            assert copilot_auth._try_gh_cli_token() == "gho_late"
        self._reset()


class TestRequestHeaders:
    """Copilot API header generation."""

    def test_default_headers_include_openai_intent(self):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = copilot_request_headers()
        assert headers["Openai-Intent"] == "conversation-edits"
        assert headers["User-Agent"] == "HermesAgent/1.0"
        assert "Editor-Version" in headers


    def test_no_vision_header_by_default(self):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = copilot_request_headers()
        assert "Copilot-Vision-Request" not in headers


class TestCopilotDefaultHeaders:
    """The models.py copilot_default_headers uses copilot_auth."""


    def test_param_passthrough_both_values(self):
        """is_agent_turn param correctly maps to x-initiator for both True and False."""
        from hermes_cli.models import copilot_default_headers
        for is_agent, expected in [(True, "agent"), (False, "user")]:
            headers = copilot_default_headers(is_agent_turn=is_agent)
            assert headers["x-initiator"] == expected, (
                f"is_agent_turn={is_agent} should produce x-initiator={expected!r}, "
                f"got {headers['x-initiator']!r}"
            )


class TestEnvVarOrder:
    """PROVIDER_REGISTRY has correct env var order."""

    def test_copilot_env_vars_include_copilot_github_token(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        copilot = PROVIDER_REGISTRY["copilot"]
        assert "COPILOT_GITHUB_TOKEN" in copilot.api_key_env_vars
        # COPILOT_GITHUB_TOKEN should be first
        assert copilot.api_key_env_vars[0] == "COPILOT_GITHUB_TOKEN"

