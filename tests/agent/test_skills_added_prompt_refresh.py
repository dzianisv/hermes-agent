"""A skill installed or replaced after a session stored its system prompt must reach the model in that same session.

A continuing session restores its stored prompt byte-for-byte on every turn (a restart does not refresh it),
so the skills index inside it freezes at session start. ``_restore_or_build_system_prompt`` rebuilds ONCE
when the current index lists a skill the stored one lacks or lists differently; the session id, history and
title are untouched. Only the renderer-owned index counts: the caller's system message, context files and
memory may quote the same ``<available_skills>`` tag.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import prompt_builder as pb
from agent.conversation_loop import _restore_or_build_system_prompt, _skills_changed_since_stored
from agent.system_prompt import _skills_prompt, build_system_prompt
from hermes_state import SessionDB

SKILL_TOOLS = ["skills_list", "skill_view", "skill_manage"]


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    yield tmp_path
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)


def _write_skill(home: Path, name: str, description: str = "Does a thing.", category: str = "tools") -> None:
    skill_dir = home / "skills" / category / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n")


def _render(by_category, *, compact=None, oneshot=False) -> str:
    with patch("agent.oneshot_footprint.is_single_query_session", return_value=oneshot):
        return pb._render_skills_index(by_category, {"tools": "Tool skills"}, compact, None)


def _shaped(index: str, *, context: str = "", memory: str = "") -> str:
    """A stored prompt in the real tier order: identity, caller/context text, then the volatile tier that opens with the index."""
    return "\n\n".join(part for part in ("identity", context, index, memory) if part)


class TestSkillsIndexEntries:
    @pytest.mark.parametrize("oneshot", [False, True])
    def test_round_trips_the_real_renderer(self, oneshot):
        index = _render(
            {"tools": [("dashboard-access", "How to reach the dashboard."), ("bare", "")],
             "org:acme": [("shared", "[org-shared] by someone")]},
            oneshot=oneshot,
        )
        assert pb.skills_index_entries(_shaped(index)) == {
            "dashboard-access": {"How to reach the dashboard."}, "bare": {""}, "shared": {"[org-shared] by someone"},
        }

    def test_names_only_category_keeps_every_name_without_a_description(self):
        index = _render({"social": [("b-skill", ""), ("a-skill", "d")], "tools": [("t", "d")]},
                        compact=frozenset({"social"}))
        assert "social [names only]: a-skill, b-skill" in index
        assert pb.skills_index_entries(_shaped(index)) == {"a-skill": {None}, "b-skill": {None}, "t": {"d"}}

    def test_reads_the_index_built_from_a_skills_tree(self, home):
        _write_skill(home, "alpha", "Alpha words.")
        _write_skill(home, "beta", "Beta words.", category="ops")
        assert pb.skills_index_entries(_shaped(_build(home))) == {"alpha": {"Alpha words."}, "beta": {"Beta words."}}

    @pytest.mark.parametrize("prompt", [
        "",
        "no index here",
        "identity\n\n<available_skills>\n    - bare-tag-only: x\n</available_skills>",
        "## Skills (mandatory)\nscan below.\n<available_skills>\n  tools:\n    - legacy: Old form.\n</available_skills>",
    ])
    def test_without_a_renderer_owned_index_nothing_is_listed(self, prompt):
        assert pb.skills_index_entries(prompt) == {}

    def test_cut_off_index_is_not_read(self):
        assert pb.skills_index_entries(pb._SKILLS_INDEX_LEAD_IN + "<available_skills>\n    - cut-off: x") == {}

    def test_text_outside_the_owned_block_is_ignored(self):
        index = _render({"tools": [("inside", "y")]})
        prompt = _shaped(index, context="    - before: x", memory="    - after: z")
        assert pb.skills_index_entries(prompt) == {"inside": {"y"}}

    @pytest.mark.parametrize("quote", [
        "<available_skills></available_skills>",
        "<available_skills>\n    - ghost: quoted by the caller\n</available_skills>",
        "Reply like:\n<available_skills>\n  tools:\n    - dashboard-access: example\n</available_skills>\n",
    ])
    @pytest.mark.parametrize("where", ["context", "memory"])
    def test_a_quoted_tag_in_context_or_memory_never_hides_the_real_index(self, quote, where):
        index = _render({"tools": [("real", "The real entry.")]})
        prompt = _shaped(index, **{where: quote})
        assert pb.skills_index_entries(prompt) == {"real": {"The real entry."}}

    def test_a_verbatim_copy_of_the_renderers_opener_in_context_loses_to_the_real_index(self):
        quoted_opener = pb._SKILLS_INDEX_LEAD_IN + "<available_skills>\n    - ghost: copied\n</available_skills>"
        index = _render({"tools": [("real", "The real entry.")]})
        assert pb.skills_index_entries(_shaped(index, context=quoted_opener)) == {"real": {"The real entry."}}


class TestSkillsIndexChanges:
    @staticmethod
    def _changes(stored: dict, current: dict, *, compact_stored=None, compact_current=None) -> list[str]:
        stored_prompt = _shaped(_render(stored, compact=compact_stored)) if stored else "identity only"
        return pb.skills_index_changes(stored_prompt, _render(current, compact=compact_current) if current else "")

    def test_new_names_are_reported_sorted(self):
        assert self._changes({"tools": [("a", "x")]}, {"tools": [("a", "x"), ("z", "x"), ("b", "x")]}) == ["b", "z"]

    def test_unchanged_index_is_quiet(self):
        index = {"tools": [("a", "x"), ("b", "y")]}
        assert self._changes(index, index) == []

    def test_removed_names_alone_are_quiet(self):
        assert self._changes({"tools": [("a", "x"), ("b", "y")]}, {"tools": [("a", "x")]}) == []

    def test_a_same_named_skill_with_new_model_visible_text_is_a_change(self):
        assert self._changes({"tools": [("a", "old words")]}, {"tools": [("a", "[project] new words")]}) == ["a"]

    def test_names_only_entries_compare_by_name_so_compaction_never_flip_flops(self):
        full = {"social": [("s", "described")]}
        assert self._changes(full, full, compact_current=frozenset({"social"})) == []
        assert self._changes(full, full, compact_stored=frozenset({"social"})) == []
        assert self._changes(full, {"social": [("s", "described"), ("t", "new")]},
                             compact_stored=frozenset({"social"}), compact_current=frozenset({"social"})) == ["t"]

    def test_session_without_an_index_gets_every_current_name(self):
        assert self._changes({}, {"tools": [("a", "x"), ("b", "y")]}) == ["a", "b"]

    def test_current_without_an_index_never_refreshes(self):
        assert self._changes({"tools": [("a", "x")]}, {}) == []

    def test_legacy_stored_form_refreshes_once_then_is_stable(self):
        legacy = "## Skills (mandatory)\n<available_skills>\n  tools:\n    - a: x\n</available_skills>"
        current = _render({"tools": [("a", "x")]})
        assert pb.skills_index_changes(legacy, current) == ["a"]
        assert pb.skills_index_changes(_shaped(current), current) == []


class TestSkillsPromptCacheSeesNewSkills:
    def test_new_skill_is_indexed_without_clearing_the_cache(self, home):
        _write_skill(home, "first")
        before = pb.build_skills_system_prompt(skills_dir_override=home / "skills")
        _write_skill(home, "dashboard-access")
        after = pb.build_skills_system_prompt(skills_dir_override=home / "skills")
        assert "dashboard-access" not in before
        assert "dashboard-access" in after

    def test_unchanged_tree_is_served_from_the_cache(self, home):
        _write_skill(home, "first")
        one = pb.build_skills_system_prompt(skills_dir_override=home / "skills")
        two = pb.build_skills_system_prompt(skills_dir_override=home / "skills")
        assert one is two


def _skill_in(root: Path, name: str, description: str = "Does a thing.") -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n")


def _build(home: Path) -> str:
    return pb.build_skills_system_prompt(skills_dir_override=home / "skills")


@pytest.fixture
def roots(home, monkeypatch) -> SimpleNamespace:
    """Two external roots plus a trusted project root, all wired through the real config.yaml."""
    from agent import skill_utils as su

    external = [home / "ext-a", home / "ext-b"]
    for root in external:
        root.mkdir()
    repo = home / "proj"
    (repo / ".git").mkdir(parents=True)
    (home / "skills").mkdir()
    (home / "config.yaml").write_text(
        "skills:\n  external_dirs:\n" + "".join(f"    - {r}\n" for r in external)
        + f"  trusted_project_dirs: ['{repo}']\n")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.chdir(repo)
    su._external_dirs_cache_clear()
    su._PROJECT_QUARANTINE_CACHE.clear()
    yield SimpleNamespace(external=external, project=repo / ".hermes" / "skills")
    su._external_dirs_cache_clear()
    su._PROJECT_QUARANTINE_CACHE.clear()


class TestEverySkillRootFeedsTheCacheKey:
    @pytest.mark.parametrize("which", [0, 1])
    def test_skill_added_to_an_external_root_is_indexed_and_unchanged_roots_hit_the_cache(self, home, roots, which):
        _skill_in(home / "skills" / "tools", "local-skill")
        _skill_in(roots.external[0], "ext-seed")
        before = _build(home)
        assert _build(home) is before
        _skill_in(roots.external[which], "ext-added")
        after = _build(home)
        assert "ext-added" not in before and "ext-added" in after
        assert _build(home) is after

    def test_skill_added_to_a_project_root_is_indexed_and_unchanged_roots_hit_the_cache(self, home, roots):
        _skill_in(roots.project, "proj-seed")
        before = _build(home)
        assert "proj-seed" in before and _build(home) is before
        _skill_in(roots.project, "proj-added")
        after = _build(home)
        assert "proj-added" not in before and "proj-added" in after
        assert _build(home) is after

    def test_edited_external_description_reaches_the_index(self, home, roots):
        _skill_in(roots.external[0], "ext-seed", "old wording")
        before = _build(home)
        _skill_in(roots.external[0], "ext-seed", "brand new wording here")
        after = _build(home)
        assert "old wording" in before and "brand new wording here" in after

    def test_precedence_is_kept_when_a_root_changes(self, home, roots):
        _skill_in(home / "skills" / "tools", "shared", "from local")
        _skill_in(roots.external[0], "shared", "from external")
        assert "from local" in _build(home)
        _skill_in(roots.external[1], "ext-added")
        after = _build(home)
        assert "ext-added" in after and "from local" in after and "from external" not in after

    def test_added_external_and_project_skills_are_reported_for_a_stored_prompt(self, home, roots):
        # Every root already exists when the prompt is stored: only new files change, not the root list.
        _skill_in(home / "skills" / "tools", "local-skill")
        _skill_in(roots.external[0], "ext-seed")
        _skill_in(roots.project, "proj-seed")
        stored = "identity\n\n" + _skills_prompt(_agent())
        _skill_in(roots.external[1], "ext-added")
        _skill_in(roots.project, "proj-added")
        assert _skills_changed_since_stored(_agent(), stored) == ["ext-added", "proj-added"]

    def test_project_skill_replacing_a_same_named_local_skill_is_reported(self, home, roots):
        _skill_in(home / "skills" / "tools", "shared", "old local words")
        _skill_in(roots.project, "proj-seed")
        stored = "identity\n\n" + _skills_prompt(_agent())
        _skill_in(roots.project, "shared", "new project words")
        current = _skills_prompt(_agent())
        assert "[project] new project words" in current and "old local words" not in current
        assert _skills_changed_since_stored(_agent(), stored) == ["shared"]


def _agent(tools=SKILL_TOOLS, **extra):
    return SimpleNamespace(session_id="sess-1", valid_tool_names=list(tools), platform="cli",
                           _session_db=None, **extra)


class TestSkillsChangedSinceStored:
    def test_reports_only_added_skills(self, home):
        _write_skill(home, "alpha")
        stored = "identity\n\n" + _skills_prompt(_agent())
        _write_skill(home, "dashboard-access")
        assert _skills_changed_since_stored(_agent(), stored) == ["dashboard-access"]

    def test_session_started_before_its_first_skill_has_no_index_yet(self, home):
        _write_skill(home, "dashboard-access")
        assert _skills_changed_since_stored(_agent(), "identity only, no skills block") == ["dashboard-access"]

    def test_skill_tools_off_never_refreshes(self, home):
        _write_skill(home, "dashboard-access")
        assert _skills_changed_since_stored(_agent(tools=["terminal"]), "identity only") == []

    def test_unchanged_and_removed_skills_do_not_refresh(self, home):
        _write_skill(home, "alpha")
        _write_skill(home, "beta")
        stored = "identity\n\n" + _skills_prompt(_agent())
        assert _skills_changed_since_stored(_agent(), stored) == []
        (home / "skills" / "tools" / "beta" / "SKILL.md").unlink()
        assert _skills_changed_since_stored(_agent(), stored) == []

    def test_edited_description_of_an_installed_skill_is_reported(self, home):
        _write_skill(home, "alpha", "old words")
        stored = "identity\n\n" + _skills_prompt(_agent())
        _write_skill(home, "alpha", "new and longer words")
        assert _skills_changed_since_stored(_agent(), stored) == ["alpha"]

    def test_probe_failure_warns_once_keeps_the_prompt_and_retries(self, home, caplog):
        _write_skill(home, "dashboard-access")
        with patch("agent.system_prompt._skills_prompt", side_effect=RuntimeError("boom " * 200)):
            with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
                assert _skills_changed_since_stored(_agent(), "identity only") == []
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "sess-1" in message and "RuntimeError" in message and "re-runs next turn" in message
        assert len(message) < 400
        # Nothing is remembered about the failure: the next turn probes again and succeeds.
        assert _skills_changed_since_stored(_agent(), "identity only") == ["dashboard-access"]


class TestRestoreWithARealSessionDb:
    """Real SQLite SessionDB, closed and reopened between turns the way a gateway restart does."""

    @staticmethod
    def _turn_agent(db, session_id, tools=SKILL_TOOLS):
        agent = SimpleNamespace(
            load_soul_identity=True, skip_context_files=True, valid_tool_names=list(tools),
            _task_completion_guidance=False, _tool_use_enforcement=False, _environment_probe=False,
            _kanban_worker_guidance="", _memory_store=None, _memory_manager=None, model="test-model",
            provider="test-provider", platform="cli", pass_session_id=False, session_id=session_id,
            _session_db=db, _use_prompt_caching=False, _cached_system_prompt=None,
            _platform_hint_overrides=None, _surface_switch_note="", _gateway_turn_context_notes="",
            _emit_status=lambda *_a, **_k: None,
        )
        builds = []

        def build(system_message=None):
            builds.append(1)
            return build_system_prompt(agent, system_message)

        agent._build_system_prompt = build
        agent.builds = builds
        return agent

    @staticmethod
    def _history(db, session_id):
        return [{"role": m["role"], "content": m["content"]} for m in db.get_messages(session_id)]

    def _start_session(self, home, *, tools=SKILL_TOOLS, system_message=None):
        (home / "SOUL.md").write_text("PERSONA-ONE")
        db = SessionDB(home / "state.db")
        db.create_session("sess-1", "cli", model="test-model")
        agent = self._turn_agent(db, "sess-1", tools)
        _restore_or_build_system_prompt(agent, system_message, [])  # first turn: builds and persists
        db.append_message("sess-1", "user", "hello")
        db.append_message("sess-1", "assistant", "hi there")
        db.set_session_title("sess-1", "Existing title")
        return db, agent._cached_system_prompt

    def _turn(self, home, *, system_message=None, tools=SKILL_TOOLS):
        """One gateway turn on a reopened DB: restore or rebuild the prompt, then close.

        Returns (agent, DB prompt writes during the turn, stored prompt, history, title)."""
        db = SessionDB(home / "state.db")
        history = self._history(db, "sess-1")
        agent = self._turn_agent(db, "sess-1", tools)
        with patch.object(db, "update_system_prompt", wraps=db.update_system_prompt) as write:
            _restore_or_build_system_prompt(agent, system_message, history)
        result = (agent, write.call_count, db.get_session("sess-1")["system_prompt"], history,
                  db.get_session_title("sess-1"))
        db.close()
        return result

    def test_added_skill_reaches_the_same_session_once(self, home):
        _write_skill(home, "old-skill")
        db, first_prompt = self._start_session(home)
        history = self._history(db, "sess-1")
        assert "old-skill" in first_prompt and "dashboard-access" not in first_prompt
        db.close()

        _write_skill(home, "dashboard-access")
        (home / "SOUL.md").write_text("PERSONA-TWO")
        db = SessionDB(home / "state.db")
        agent = self._turn_agent(db, "sess-1")
        _restore_or_build_system_prompt(agent, None, history)

        refreshed = agent._cached_system_prompt
        assert len(agent.builds) == 1
        assert "dashboard-access" in refreshed and "old-skill" in refreshed
        # A rebuild adopts the CURRENT persona (and memory/context/timestamp) along with the new skill.
        assert "PERSONA-TWO" in refreshed and "PERSONA-ONE" not in refreshed
        assert db.get_session("sess-1")["system_prompt"] == refreshed
        assert self._history(db, "sess-1") == history
        assert db.get_session_title("sess-1") == "Existing title"
        assert not hasattr(agent, "_session_title_hint")
        db.close()

        db = SessionDB(home / "state.db")
        agent = self._turn_agent(db, "sess-1")
        _restore_or_build_system_prompt(agent, None, history)
        assert agent.builds == []
        assert agent._cached_system_prompt == refreshed
        assert self._history(db, "sess-1") == history
        db.close()

    def test_session_created_before_its_first_skill_is_refreshed(self, home):
        db, first_prompt = self._start_session(home)
        history = self._history(db, "sess-1")
        assert "<available_skills>" not in first_prompt
        db.close()

        _write_skill(home, "dashboard-access")
        db = SessionDB(home / "state.db")
        agent = self._turn_agent(db, "sess-1")
        _restore_or_build_system_prompt(agent, None, history)
        assert len(agent.builds) == 1
        assert "dashboard-access" in db.get_session("sess-1")["system_prompt"]
        db.close()

    def test_session_with_skill_tools_off_keeps_its_prompt(self, home):
        db, first_prompt = self._start_session(home, tools=["terminal"])
        history = self._history(db, "sess-1")
        db.close()

        _write_skill(home, "dashboard-access")
        db = SessionDB(home / "state.db")
        agent = self._turn_agent(db, "sess-1", tools=["terminal"])
        _restore_or_build_system_prompt(agent, None, history)
        assert agent.builds == []
        assert agent._cached_system_prompt == first_prompt
        db.close()

    def test_system_message_quoting_an_empty_tag_rebuilds_once_not_every_turn(self, home):
        quote = "Answer format, for example: <available_skills></available_skills>"
        _write_skill(home, "old-skill")
        db, first = self._start_session(home, system_message=quote)
        db.close()

        agent, writes, stored, _history, _title = self._turn(home, system_message=quote)
        assert (agent.builds, writes, stored) == ([], 0, first)  # nothing changed: nothing rebuilt

        _write_skill(home, "dashboard-access")
        agent, writes, refreshed, history, title = self._turn(home, system_message=quote)
        assert (len(agent.builds), writes) == (1, 1)
        assert set(pb.skills_index_entries(refreshed)) == {"old-skill", "dashboard-access"}
        assert quote in refreshed and title == "Existing title"

        agent, writes, stored, again, _title = self._turn(home, system_message=quote)
        assert (agent.builds, writes, stored, again) == ([], 0, refreshed, history)

    def test_system_message_listing_the_new_skill_does_not_hide_its_addition(self, home):
        quote = "Answer format:\n<available_skills>\n    - dashboard-access: quoted example\n</available_skills>"
        _write_skill(home, "old-skill")
        db, _first = self._start_session(home, system_message=quote)
        db.close()

        _write_skill(home, "dashboard-access", "The real entry.")
        agent, writes, refreshed, _history, _title = self._turn(home, system_message=quote)
        assert (len(agent.builds), writes) == (1, 1)
        assert pb.skills_index_entries(refreshed)["dashboard-access"] == {"The real entry."}

    def test_project_skill_replacing_a_local_one_refreshes_once_with_history_and_title_kept(self, home, roots):
        _write_skill(home, "dashboard-access", "Old local words.")
        _write_skill(home, "unrelated")
        _skill_in(roots.project, "proj-seed")  # the project root exists before the replacement arrives
        db, first = self._start_session(home)
        history = self._history(db, "sess-1")
        db.close()
        assert pb.skills_index_entries(first)["dashboard-access"] == {"Old local words."}

        _skill_in(roots.project, "dashboard-access", "New project words.")
        agent, writes, refreshed, seen, title = self._turn(home)
        assert (len(agent.builds), writes) == (1, 1)
        assert pb.skills_index_entries(refreshed)["dashboard-access"] == {"[project] New project words."}
        assert seen == history and title == "Existing title"

        # Stable afterwards, and removing an unrelated skill is not a reason to rebuild.
        (home / "skills" / "tools" / "unrelated" / "SKILL.md").unlink()
        agent, writes, stored, _history, _title = self._turn(home)
        assert (agent.builds, writes, stored) == ([], 0, refreshed)

        # Removing the project copy changes what the local one shows, which is a visible change: once.
        (roots.project / "dashboard-access" / "SKILL.md").unlink()
        agent, writes, restored, _history, _title = self._turn(home)
        assert (len(agent.builds), writes) == (1, 1)
        assert pb.skills_index_entries(restored)["dashboard-access"] == {"Old local words."}
