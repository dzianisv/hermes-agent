"""Dashboard assignee patches must reject non-runnable owners (t_e36115cf).

Companion to ``tests/hermes_cli/test_kanban_runnable_assignee.py`` — the
dashboard is the second surface that can create placeholder ownership.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# These suites ARE the guard's tests: they install a real profile on a temp
# HERMES root and must see the real profile_exists predicate, not the autouse
# stub that makes every assignee resolve.
pytestmark = pytest.mark.real_profile_gate


from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def installed_profile(kanban_home):
    name = "software-engineer"
    (kanban_home / "profiles" / name).mkdir(parents=True)
    return name


@pytest.fixture
def conn(kanban_home):
    c = kb.connect()
    yield c
    c.close()


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_runnable_assignee_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def test_patch_rejects_unknown_assignee(client, conn, installed_profile):
    tid = kb.create_task(conn, title="t", assignee=installed_profile)
    r = client.patch(
        f"/api/plugins/kanban/tasks/{tid}", json={"assignee": "copilot-external"},
    )
    assert r.status_code == 400
    assert "copilot-external" in r.json()["detail"]
    task = kb.get_task(conn, tid)
    assert task is not None and task.assignee == installed_profile


def test_patch_accepts_installed_profile(client, conn, installed_profile):
    tid = kb.create_task(conn, title="t")
    r = client.patch(
        f"/api/plugins/kanban/tasks/{tid}", json={"assignee": installed_profile},
    )
    assert r.status_code == 200, r.text
    task = kb.get_task(conn, tid)
    assert task is not None and task.assignee == installed_profile


def test_patch_still_allows_unassignment(client, conn, installed_profile):
    tid = kb.create_task(conn, title="t", assignee=installed_profile)
    r = client.patch(f"/api/plugins/kanban/tasks/{tid}", json={"assignee": ""})
    assert r.status_code == 200, r.text
    task = kb.get_task(conn, tid)
    assert task is not None and task.assignee is None


def test_bulk_patch_rejects_unknown_assignee(client, conn, installed_profile):
    tids = [
        kb.create_task(conn, title=f"t{i}", assignee=installed_profile)
        for i in range(2)
    ]
    r = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={"ids": tids, "assignee": "builder"},
    )
    # Whole batch is refused up front — no partial placeholder ownership.
    assert r.status_code == 400
    assert "builder" in r.json()["detail"]
    for tid in tids:
        task = kb.get_task(conn, tid)
        assert task is not None and task.assignee == installed_profile


def test_reassign_endpoint_rejects_unknown_assignee(
    client, conn, installed_profile
):
    tid = kb.create_task(conn, title="t", assignee=installed_profile)
    r = client.post(
        f"/api/plugins/kanban/tasks/{tid}/reassign",
        json={"profile": "copilot-external"},
    )
    assert r.status_code == 400
    assert "copilot-external" in r.json()["detail"]
    task = kb.get_task(conn, tid)
    assert task is not None and task.assignee == installed_profile


def test_create_endpoint_rejects_unknown_assignee(
    client, conn, installed_profile
):
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "t", "assignee": "copilot-external"},
    )
    assert r.status_code == 400
    assert "copilot-external" in r.json()["detail"]
    assert kb.list_tasks(conn) == []
