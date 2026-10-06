"""The ACP agent (pi-acp, copilot) must start in the card's project folder so it loads that repo's AGENTS.md."""
import subprocess
from types import SimpleNamespace

import pytest

import agent.copilot_acp_client as acp
from agent.copilot_acp_client import CopilotACPClient


@pytest.fixture
def no_terminals(monkeypatch):
    import tools.terminal_tool as tt
    monkeypatch.setattr(tt, "_active_environments", {})
    return tt


def _capture_popen(monkeypatch):
    seen = {}
    def fake_popen(*a, **kw):
        seen["cwd"] = kw.get("cwd")
        raise OSError("stop after spawn args are captured")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(acp, "_acp_supported", lambda *a, **k: True)
    return seen


def test_spawn_uses_kanban_workspace_not_launch_dir(monkeypatch, tmp_path, no_terminals):
    repo = tmp_path / "repo"; repo.mkdir()
    launch = tmp_path / "launch"; launch.mkdir()
    monkeypatch.chdir(launch)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(repo))
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    client = CopilotACPClient(acp_command="pi-acp", acp_args=[])
    seen = _capture_popen(monkeypatch)
    with pytest.raises(Exception):
        client._spawn()
    assert seen["cwd"] == str(repo.resolve())


def test_spawn_follows_terminal_cd_into_checkout(monkeypatch, tmp_path, no_terminals):
    ws = tmp_path / "ws"; ws.mkdir()
    checkout = ws / "AgentPod"; checkout.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(ws))
    client = CopilotACPClient(acp_command="pi-acp", acp_args=[])
    no_terminals._active_environments["t1"] = SimpleNamespace(cwd=str(checkout))
    seen = _capture_popen(monkeypatch)
    with pytest.raises(Exception):
        client._spawn()
    assert seen["cwd"] == str(checkout.resolve())


def test_explicit_acp_cwd_still_wins(monkeypatch, tmp_path, no_terminals):
    explicit = tmp_path / "x"; explicit.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(tmp_path))
    client = CopilotACPClient(acp_command="pi-acp", acp_args=[], acp_cwd=str(explicit))
    seen = _capture_popen(monkeypatch)
    with pytest.raises(Exception):
        client._spawn()
    assert seen["cwd"] == str(explicit.resolve())
