"""Durable launchers must bind the install's own store, never a transient runtime dir.

Regression: a probe ran with ``HERMES_RUNTIME_DIR=/tmp/opencode/hermes-fbtest``.
Launcher publication baked that temp interpreter into
``~/.hermes/hermes-agent/.hermes/bin/hermes``; the temp dir was then deleted, so
every CLI call died in ``exec`` with "not found".
"""
from pathlib import Path

import hermes_cli._launchers as launchers


def _checkout(tmp_path: Path) -> Path:
    repo = tmp_path / "home" / "hermes-agent"
    repo.mkdir(parents=True)
    (repo / ".git").mkdir()
    return repo


def test_store_outside_install_home_is_rejected(tmp_path):
    repo = _checkout(tmp_path)
    outside = tmp_path / "tmp" / "opencode" / "hermes-fbtest" / "tools"
    assert launchers.store_within_install_home(repo, outside) is False


def test_store_under_install_home_is_accepted(tmp_path):
    repo = _checkout(tmp_path)
    assert launchers.store_within_install_home(repo, tmp_path / "home" / "tools") is True


def test_non_git_tree_keeps_its_own_launchers(tmp_path):
    repo = tmp_path / "payload"
    repo.mkdir()
    assert launchers.store_within_install_home(repo, tmp_path / "elsewhere" / "tools") is True


def test_stage_launcher_refuses_outside_store(tmp_path, monkeypatch):
    repo = _checkout(tmp_path)
    out = repo / ".hermes" / "bin"
    temp_store = tmp_path / "tmp" / "opencode" / "hermes-fbtest" / "tools"
    temp_python = temp_store / "bin" / "python3"
    temp_python.parent.mkdir(parents=True)
    temp_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(launchers, "store_root", lambda root: temp_store)
    monkeypatch.setattr(launchers, "resolve_store_python", lambda root: temp_python)

    assert launchers.stage_launcher("hermes", repo, out) is None
    assert not (out / "hermes").exists()
