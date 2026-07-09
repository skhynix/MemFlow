# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from memflow.claude_catalog import (
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


def test_missing_config_uses_default_effective_mode():
    mode = normalize_native_catalog_mode({})

    assert mode.raw_mode is None
    assert mode.effective_mode == "hidden_or_minimized"
    assert mode.warnings == ()


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
