"""A root-level module that shipped code imports must be packaged.

``[tool.setuptools.packages.find]`` only collects *packages*; a single-file
module that sits at the repo root ships only if it is named in
``[tool.setuptools] py-modules``. Miss one and the source checkout keeps
working — the repo root is on ``sys.path`` — while a sealed wheel / uv2nix
install raises ``ModuleNotFoundError`` on the first import that needs it.

The coverage here is derived, not enumerated: the declared root modules are
imported for real, the resulting import graph is read out of ``sys.modules``,
and any module that lives directly in the repo root but is absent from
``py-modules`` fails. A new root module wired into shipped code is caught the
moment it is imported by one, with nothing to keep in sync by hand.
"""

from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def _declared_py_modules() -> list[str]:
    return list(_pyproject()["tool"]["setuptools"].get("py-modules", []))


def _root_modules_in_import_graph() -> set[str]:
    """Top-level modules whose file sits directly in the repo root."""
    found: set[str] = set()
    for name, module in list(sys.modules.items()):
        if "." in name:
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and path.parent == REPO_ROOT:
            found.add(name)
    return found


def test_declared_root_modules_are_importable() -> None:
    """Everything py-modules promises to ship must actually exist and import."""
    declared = _declared_py_modules()
    assert declared, "pyproject declares no py-modules at all"

    broken: dict[str, str] = {}
    for name in declared:
        if not (REPO_ROOT / f"{name}.py").is_file():
            broken[name] = "declared in py-modules but no such file at the repo root"
            continue
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - a real packaging break
            broken[name] = f"{type(exc).__name__}: {exc}"
    assert not broken, f"py-modules entries that cannot be shipped: {broken}"


def test_root_modules_pulled_in_by_shipped_code_are_declared() -> None:
    """Import the shipped root modules; every root module they reach is declared.

    ``sqlite_schema_text`` is the case this was written for: it is imported at
    import time by ``hermes_state_schema``/``hermes_cli.kanban_db``/
    ``hermes_cli.projects_db``, so a sealed install without it fails on the
    first ``connect()``.
    """
    declared = set(_declared_py_modules())
    for name in sorted(declared):
        try:
            importlib.import_module(name)
        except Exception:
            # Import health is asserted by the test above; this one only needs
            # the graph that did load.
            continue

    undeclared = sorted(_root_modules_in_import_graph() - declared)
    # Test modules and conftest also live outside packages but never in the
    # repo root, so nothing here needs filtering beyond the root check itself.
    assert not undeclared, (
        "root-level modules reached by shipped code but missing from "
        f"[tool.setuptools] py-modules in pyproject.toml: {undeclared}. "
        "A sealed wheel / uv2nix venv would raise ModuleNotFoundError on them."
    )
