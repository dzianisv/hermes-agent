"""Blast-radius contract for the synthetic-assignee test gate (card t_e36115cf).

The autouse gate in the ROOT ``tests/conftest.py`` exists so that kanban tests
can own tasks with synthetic assignees ("alice", "worker") that have no profile
directory on disk. Its first implementation stubbed the SHARED predicate
``hermes_cli.profiles.profile_exists`` to always-True for every test in the
tree. That is the same predicate behind web_server profile scoping, cron
profile validation, session_search and gateway profile resolution, so the gate
silently disabled every "unknown profile is rejected" assertion in the repo —
two of them turned red, and the rest would have gone vacuously green.

These tests pin the correction: the gate must reach ONLY the kanban seam.
No ``real_profile_gate`` marker here on purpose — the autouse fixture is
exactly what is under test.
"""

from __future__ import annotations

from hermes_cli import kanban_db as kb
from hermes_cli import profiles as profiles_mod


def test_gate_does_not_stub_the_shared_profile_exists_predicate():
    """The shared predicate keeps telling the truth under the autouse gate.

    ``profile_exists`` is consumed by web_server, cron, session_search and the
    gateway. If the kanban gate stubs it, every negative-path test in those
    suites either fails or passes for the wrong reason.
    """
    assert profiles_mod.profile_exists("no-such-profile-t-e36115cf") is False


def test_gate_makes_synthetic_kanban_assignees_resolve():
    """...while kanban's own seam still accepts synthetic assignees.

    This is the whole point of the gate: kanban tests across the tree create
    tasks owned by names that were never installed.
    """
    assert kb._assignee_profile_exists("alice") is True
    assert kb.ensure_runnable_assignee("alice") == "alice"
