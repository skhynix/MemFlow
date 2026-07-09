# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from memflow.claude_catalog import (
    apply_claude_catalog_settings,
    build_claude_catalog_settings_plan,
    discover_claude_skills,
    normalize_native_catalog_mode,
)


def _write_skill(root, *, name: str | None = None) -> None:
    root.mkdir(parents=True)
    frontmatter = "---\ndescription: Test skill.\n---\n"
    if name is not None:
        frontmatter = f"---\nname: {name}\ndescription: Test skill.\n---\n"
    (root / "SKILL.md").write_text(
        f"{frontmatter}# Test Skill\n\nUse this skill.\n",
        encoding="utf-8",
    )


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_missing_config_uses_default_effective_mode():
    mode = normalize_native_catalog_mode({})

    assert mode.raw_mode is None
    assert mode.effective_mode == "hidden_or_minimized"
    assert mode.warnings == ()


def test_visible_generates_no_catalog_suppression_patch(tmp_path):
    _write_skill(tmp_path / ".claude" / "skills" / "project-skill")

    plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "visible"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert plan.settings_after == {}
    assert plan.managed_skill_overrides == ()
    assert plan.settings_changed is False


def test_hidden_or_minimized_maps_discovered_skills(tmp_path):
    user_skills_root = tmp_path / "home" / ".claude" / "skills"
    _write_skill(
        tmp_path / ".claude" / "skills" / "project-dir",
        name="project-skill",
    )
    _write_skill(user_skills_root / "user-skill")

    plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "hidden_or_minimized"}},
        project_root=tmp_path,
        user_skills_root=user_skills_root,
    )

    assert plan.settings_after["disableBundledSkills"] is True
    assert plan.settings_after["skillOverrides"] == {
        "project-skill": "user-invocable-only",
        "user-skill": "user-invocable-only",
    }


def test_disabled_maps_discovered_skills_to_off(tmp_path):
    _write_skill(
        tmp_path / ".claude" / "skills" / "project-dir",
        name="project-skill",
    )

    plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "disabled"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert plan.settings_after["disableBundledSkills"] is True
    assert plan.settings_after["skillOverrides"] == {"project-skill": "off"}


@pytest.mark.parametrize(
    ("mode", "existing_override"),
    (
        ("visible", "off"),
        ("hidden_or_minimized", "off"),
        ("disabled", "user-invocable-only"),
    ),
)
def test_existing_catalog_restrictions_warn_when_state_is_unavailable(
    tmp_path,
    mode,
    existing_override,
):
    warning = "catalog_state_unavailable_existing_catalog_settings_preserved"
    _write_skill(tmp_path / ".claude" / "skills" / "existing-skill")
    settings_before = {
        "disableBundledSkills": True,
        "skillOverrides": {"existing-skill": existing_override},
    }

    plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": mode}},
        project_root=tmp_path,
        settings_before=settings_before,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert plan.warnings.count(warning) == 1
    assert plan.settings_after == settings_before


def test_initial_catalog_configure_does_not_warn_about_unavailable_state(tmp_path):
    warning = "catalog_state_unavailable_existing_catalog_settings_preserved"

    plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "hidden_or_minimized"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert warning not in plan.warnings


def test_trusted_catalog_state_does_not_warn_about_unavailable_state(tmp_path):
    warning = "catalog_state_unavailable_existing_catalog_settings_preserved"
    _write_skill(tmp_path / ".claude" / "skills" / "project-skill")
    hidden_plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "hidden_or_minimized"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )
    apply_claude_catalog_settings(hidden_plan)

    disabled_plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "disabled"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert warning not in disabled_plan.warnings
    assert disabled_plan.settings_after["skillOverrides"] == {"project-skill": "off"}


@pytest.mark.parametrize("managed_mode", ("hidden_or_minimized", "disabled"))
def test_disable_bundled_original_absent_rolls_back_to_absent(
    tmp_path,
    managed_mode,
):
    managed_plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": managed_mode}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )
    apply_claude_catalog_settings(managed_plan)

    assert managed_plan.settings_after["disableBundledSkills"] is True
    assert managed_plan.state_after["disable_bundled_skills_original"] == {
        "present": False
    }

    visible_plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "visible"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert "disableBundledSkills" not in visible_plan.settings_after


def test_disable_bundled_original_true_rolls_back_to_true(tmp_path):
    settings_path = tmp_path / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps({"disableBundledSkills": True}),
        encoding="utf-8",
    )

    hidden_plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "hidden_or_minimized"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )
    apply_claude_catalog_settings(hidden_plan)

    assert hidden_plan.state_after["disable_bundled_skills_original"] == {
        "present": True,
        "value": True,
    }

    visible_plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "visible"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert visible_plan.settings_after["disableBundledSkills"] is True


def test_disable_bundled_original_false_rolls_back_to_false(tmp_path):
    settings_path = tmp_path / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps({"disableBundledSkills": False}),
        encoding="utf-8",
    )

    hidden_plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "hidden_or_minimized"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )
    apply_claude_catalog_settings(hidden_plan)

    visible_plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "visible"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert visible_plan.settings_after["disableBundledSkills"] is False


def test_disable_bundled_original_not_overwritten_between_managed_modes(tmp_path):
    settings_path = tmp_path / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps({"disableBundledSkills": False}),
        encoding="utf-8",
    )

    hidden_plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "hidden_or_minimized"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )
    apply_claude_catalog_settings(hidden_plan)

    disabled_plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "disabled"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert disabled_plan.state_after["disable_bundled_skills_original"] == {
        "present": True,
        "value": False,
    }


def test_legacy_disable_bundled_state_preserves_value_on_visible_with_warning(
    tmp_path,
):
    settings_path = tmp_path / ".claude" / "settings.local.json"
    state_path = tmp_path / ".memflow" / "claude-catalog-state.json"
    settings_path.parent.mkdir(parents=True)
    state_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps({"disableBundledSkills": True}),
        encoding="utf-8",
    )
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "memflow.claude_catalog.v1",
                "disable_bundled_skills_managed": True,
                "managed_skill_overrides": [],
                "native_catalog_mode": "hidden_or_minimized",
                "settings_path": ".claude/settings.local.json",
            }
        ),
        encoding="utf-8",
    )

    plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "visible"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert plan.settings_after["disableBundledSkills"] is True
    assert "disable_bundled_skills_original_missing_preserved" in plan.warnings


def test_manual_disable_bundled_change_after_apply_preserved_with_warning(tmp_path):
    hidden_plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "hidden_or_minimized"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )
    apply_claude_catalog_settings(hidden_plan)
    hidden_plan.settings_path.write_text(
        json.dumps({"disableBundledSkills": False}),
        encoding="utf-8",
    )

    visible_plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "visible"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert visible_plan.settings_after["disableBundledSkills"] is False
    assert "disable_bundled_skills_manual_change_preserved" in visible_plan.warnings


def test_invalid_and_non_object_catalog_config_fall_back_with_warning():
    invalid_mode = normalize_native_catalog_mode(
        {"claude": {"native_catalog_mode": "nope"}}
    )
    non_object_claude = normalize_native_catalog_mode({"claude": "bad"})

    assert invalid_mode.effective_mode == "hidden_or_minimized"
    assert invalid_mode.warnings == ("invalid_native_catalog_mode",)
    assert non_object_claude.effective_mode == "hidden_or_minimized"
    assert non_object_claude.warnings == ("invalid_claude_config",)


def test_discover_claude_skills_reads_project_and_user_frontmatter_names_in_order(
    tmp_path,
):
    user_skills_root = tmp_path / "home" / ".claude" / "skills"
    _write_skill(
        tmp_path / ".claude" / "skills" / "project-bravo-dir",
        name="bravo-skill",
    )
    _write_skill(
        tmp_path / ".claude" / "skills" / "project-charlie-dir",
        name="charlie-skill",
    )
    _write_skill(user_skills_root / "user-alpha-dir", name="alpha-skill")

    discovery = discover_claude_skills(
        tmp_path,
        user_skills_root=user_skills_root,
    )

    assert [(skill.name, skill.scope, skill.path) for skill in discovery.skills] == [
        (
            "alpha-skill",
            "user",
            str(user_skills_root / "user-alpha-dir" / "SKILL.md"),
        ),
        (
            "bravo-skill",
            "project",
            str(tmp_path / ".claude" / "skills" / "project-bravo-dir" / "SKILL.md"),
        ),
        (
            "charlie-skill",
            "project",
            str(tmp_path / ".claude" / "skills" / "project-charlie-dir" / "SKILL.md"),
        ),
    ]
    assert discovery.warnings == ()


def test_discover_claude_skills_falls_back_to_sanitized_directory_name(tmp_path):
    _write_skill(tmp_path / ".claude" / "skills" / "Fallback Skill! ..")

    discovery = discover_claude_skills(
        tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert [(skill.name, skill.scope) for skill in discovery.skills] == [
        ("Fallback-Skill", "project"),
    ]
    assert discovery.warnings == ()


def test_settings_patch_preserves_unrelated_settings_and_user_overrides(tmp_path):
    settings_path = tmp_path / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "theme": "dark",
                "permissions": {"allow": ["Bash(date)"]},
                "skillOverrides": {"manual-skill": "off"},
            }
        ),
        encoding="utf-8",
    )
    _write_skill(
        tmp_path / ".claude" / "skills" / "project-dir",
        name="project-skill",
    )

    plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "hidden_or_minimized"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert plan.settings_after["theme"] == "dark"
    assert plan.settings_after["permissions"] == {"allow": ["Bash(date)"]}
    assert plan.settings_after["skillOverrides"] == {
        "manual-skill": "off",
        "project-skill": "user-invocable-only",
    }


def test_reapply_after_skill_removal_cleans_only_managed_stale_overrides(tmp_path):
    settings_path = tmp_path / ".claude" / "settings.local.json"
    state_path = tmp_path / ".memflow" / "claude-catalog-state.json"
    settings_path.parent.mkdir(parents=True)
    state_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "disableBundledSkills": True,
                "skillOverrides": {
                    "manual-skill": "off",
                    "old-managed-skill": "user-invocable-only",
                },
            }
        ),
        encoding="utf-8",
    )
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "memflow.claude_catalog.v1",
                "disable_bundled_skills_managed": True,
                "managed_skill_overrides": ["old-managed-skill"],
                "managed_skill_override_values": {
                    "old-managed-skill": "user-invocable-only"
                },
                "native_catalog_mode": "hidden_or_minimized",
                "settings_path": ".claude/settings.local.json",
            }
        ),
        encoding="utf-8",
    )
    _write_skill(tmp_path / ".claude" / "skills" / "new-managed-skill")

    plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "hidden_or_minimized"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert plan.settings_after["skillOverrides"] == {
        "manual-skill": "off",
        "new-managed-skill": "user-invocable-only",
    }
    assert plan.removed_managed_skill_overrides == ("old-managed-skill",)
    assert plan.managed_skill_overrides == ("new-managed-skill",)


def test_schema_mismatch_matching_settings_path_salvages_managed_overrides_on_visible(
    tmp_path,
):
    settings_path = tmp_path / ".claude" / "settings.local.json"
    state_path = tmp_path / ".memflow" / "claude-catalog-state.json"
    settings_path.parent.mkdir(parents=True)
    state_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "disableBundledSkills": True,
                "skillOverrides": {
                    "manual-skill": "off",
                    "old-managed-skill": "user-invocable-only",
                },
            }
        ),
        encoding="utf-8",
    )
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "memflow.claude_catalog.v0",
                "disable_bundled_skills_managed": True,
                "disable_bundled_skills_original": {
                    "present": True,
                    "value": False,
                },
                "managed_skill_overrides": ["old-managed-skill"],
                "managed_skill_override_values": {
                    "old-managed-skill": "user-invocable-only"
                },
                "native_catalog_mode": "hidden_or_minimized",
                "settings_path": ".claude/settings.local.json",
            }
        ),
        encoding="utf-8",
    )

    plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "visible"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert plan.settings_after["disableBundledSkills"] is True
    assert plan.settings_after["skillOverrides"] == {"manual-skill": "off"}
    assert plan.removed_managed_skill_overrides == ("old-managed-skill",)
    assert "catalog_state_schema_version_mismatch_ignored" in plan.warnings
    assert (
        "catalog_state_unavailable_existing_catalog_settings_preserved" in plan.warnings
    )
    assert "disable_bundled_skills_manual_change_preserved" not in plan.warnings


def test_visible_preserves_managed_override_after_manual_value_change(tmp_path):
    _write_skill(
        tmp_path / ".claude" / "skills" / "project-dir",
        name="project-skill",
    )
    hidden_plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "hidden_or_minimized"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )
    apply_claude_catalog_settings(hidden_plan)
    settings = _read_json(hidden_plan.settings_path)
    settings["skillOverrides"]["project-skill"] = "off"
    hidden_plan.settings_path.write_text(json.dumps(settings), encoding="utf-8")

    visible_plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "visible"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert visible_plan.settings_after["skillOverrides"] == {"project-skill": "off"}
    assert visible_plan.removed_managed_skill_overrides == ()
    assert (
        "managed_skill_override_manual_change_preserved:project-skill"
        in visible_plan.warnings
    )


def test_name_only_managed_state_preserves_overrides_with_warning(tmp_path):
    settings_path = tmp_path / ".claude" / "settings.local.json"
    state_path = tmp_path / ".memflow" / "claude-catalog-state.json"
    settings_path.parent.mkdir(parents=True)
    state_path.parent.mkdir(parents=True)
    original_settings = {
        "skillOverrides": {"old-managed-skill": "user-invocable-only"},
    }
    settings_path.write_text(json.dumps(original_settings), encoding="utf-8")
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "memflow.claude_catalog.v1",
                "managed_skill_overrides": ["old-managed-skill"],
                "native_catalog_mode": "hidden_or_minimized",
                "settings_path": ".claude/settings.local.json",
            }
        ),
        encoding="utf-8",
    )

    plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "visible"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert plan.settings_after == original_settings
    assert plan.removed_managed_skill_overrides == ()
    assert (
        "managed_skill_override_value_missing_preserved:old-managed-skill"
        in plan.warnings
    )


def test_schema_mismatch_matching_settings_path_salvage_is_visible_only(tmp_path):
    settings_path = tmp_path / ".claude" / "settings.local.json"
    state_path = tmp_path / ".memflow" / "claude-catalog-state.json"
    settings_path.parent.mkdir(parents=True)
    state_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "skillOverrides": {
                    "manual-skill": "off",
                    "old-managed-skill": "user-invocable-only",
                },
            }
        ),
        encoding="utf-8",
    )
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "memflow.claude_catalog.v0",
                "managed_skill_overrides": ["old-managed-skill"],
                "managed_skill_override_values": {
                    "old-managed-skill": "user-invocable-only"
                },
                "native_catalog_mode": "hidden_or_minimized",
                "settings_path": ".claude/settings.local.json",
            }
        ),
        encoding="utf-8",
    )

    plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "hidden_or_minimized"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert plan.settings_after["skillOverrides"] == {
        "manual-skill": "off",
        "old-managed-skill": "user-invocable-only",
    }
    assert plan.removed_managed_skill_overrides == ()
    assert "catalog_state_schema_version_mismatch_ignored" in plan.warnings


def test_schema_mismatch_settings_path_mismatch_does_not_salvage_user_overrides(
    tmp_path,
):
    settings_path = tmp_path / ".claude" / "settings.local.json"
    state_path = tmp_path / ".memflow" / "claude-catalog-state.json"
    settings_path.parent.mkdir(parents=True)
    state_path.parent.mkdir(parents=True)
    original_settings = {
        "skillOverrides": {"manual-skill": "off"},
    }
    settings_path.write_text(json.dumps(original_settings), encoding="utf-8")
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "memflow.claude_catalog.v0",
                "managed_skill_overrides": ["manual-skill"],
                "native_catalog_mode": "hidden_or_minimized",
                "settings_path": "other/settings.local.json",
            }
        ),
        encoding="utf-8",
    )

    plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "visible"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert plan.settings_after == original_settings
    assert plan.removed_managed_skill_overrides == ()
    assert "catalog_state_schema_version_mismatch_ignored" in plan.warnings
    assert "catalog_state_settings_path_mismatch_ignored" in plan.warnings


def test_state_for_different_settings_path_does_not_remove_user_overrides(tmp_path):
    settings_path = tmp_path / ".claude" / "settings.local.json"
    state_path = tmp_path / ".memflow" / "claude-catalog-state.json"
    settings_path.parent.mkdir(parents=True)
    state_path.parent.mkdir(parents=True)
    original_settings = {
        "disableBundledSkills": True,
        "skillOverrides": {"manual-skill": "off"},
    }
    settings_path.write_text(json.dumps(original_settings), encoding="utf-8")
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "memflow.claude_catalog.v1",
                "disable_bundled_skills_managed": True,
                "managed_skill_overrides": ["manual-skill"],
                "native_catalog_mode": "hidden_or_minimized",
                "settings_path": "other/settings.local.json",
            }
        ),
        encoding="utf-8",
    )

    plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "visible"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert plan.settings_after == original_settings
    assert plan.removed_managed_skill_overrides == ()
    assert "catalog_state_settings_path_mismatch_ignored" in plan.warnings


def test_apply_writes_settings_and_state_while_dry_run_does_not(tmp_path):
    _write_skill(tmp_path / ".claude" / "skills" / "project-skill")
    plan = build_claude_catalog_settings_plan(
        {"claude": {"native_catalog_mode": "hidden_or_minimized"}},
        project_root=tmp_path,
        user_skills_root=tmp_path / "missing-user-skills",
    )

    assert not plan.settings_path.exists()
    assert not plan.state_path.exists()

    apply_claude_catalog_settings(plan)

    assert _read_json(plan.settings_path) == plan.settings_after
    assert _read_json(plan.state_path) == plan.state_after
