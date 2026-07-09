# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code native skill catalog settings management."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memflow.skills import parse_skill_frontmatter

DEFAULT_NATIVE_CATALOG_MODE = "hidden_or_minimized"
SUPPORTED_NATIVE_CATALOG_MODES = {
    "visible",
    "hidden_or_minimized",
    "disabled",
}


@dataclass(frozen=True)
class CatalogModeResolution:
    raw_mode: Any
    effective_mode: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscoveredClaudeSkill:
    name: str
    scope: str
    path: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClaudeSkillDiscovery:
    skills: tuple[DiscoveredClaudeSkill, ...]
    warnings: tuple[str, ...] = ()
    plugin_skills_out_of_scope: bool = True


def normalize_native_catalog_mode(config: Any) -> CatalogModeResolution:
    """Return the requested and effective Claude native catalog mode."""
    if not isinstance(config, dict):
        return CatalogModeResolution(
            raw_mode=None,
            effective_mode=DEFAULT_NATIVE_CATALOG_MODE,
            warnings=("invalid_config",),
        )

    if "claude" not in config:
        return CatalogModeResolution(
            raw_mode=None,
            effective_mode=DEFAULT_NATIVE_CATALOG_MODE,
        )

    claude_config = config.get("claude")
    if not isinstance(claude_config, dict):
        return CatalogModeResolution(
            raw_mode=None,
            effective_mode=DEFAULT_NATIVE_CATALOG_MODE,
            warnings=("invalid_claude_config",),
        )

    raw_mode = claude_config.get("native_catalog_mode")
    if raw_mode is None:
        return CatalogModeResolution(
            raw_mode=None,
            effective_mode=DEFAULT_NATIVE_CATALOG_MODE,
        )
    if isinstance(raw_mode, str) and raw_mode in SUPPORTED_NATIVE_CATALOG_MODES:
        return CatalogModeResolution(raw_mode=raw_mode, effective_mode=raw_mode)
    return CatalogModeResolution(
        raw_mode=raw_mode,
        effective_mode=DEFAULT_NATIVE_CATALOG_MODE,
        warnings=("invalid_native_catalog_mode",),
    )


def _mode_resolution_from_metadata(
    config: dict[str, Any] | None,
) -> CatalogModeResolution | None:
    if not isinstance(config, dict):
        return None
    metadata = config.get("_memflow_catalog_mode")
    if not isinstance(metadata, dict):
        return None
    effective_mode = metadata.get("effective")
    if (
        not isinstance(effective_mode, str)
        or effective_mode not in SUPPORTED_NATIVE_CATALOG_MODES
    ):
        return None

    raw_warnings = metadata.get("warnings", ())
    if not isinstance(raw_warnings, (list, tuple)):
        raw_warnings = (raw_warnings,)
    warnings = tuple(str(item) for item in raw_warnings if item)
    return CatalogModeResolution(
        raw_mode=metadata.get("raw"),
        effective_mode=effective_mode,
        warnings=warnings,
    )


def _safe_derived_skill_name(skill_path: Path) -> str:
    raw = skill_path.parent.name.strip()
    candidate = re.sub(r"\s+", "-", raw)
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip(".-_")
    if candidate:
        return candidate
    digest = hashlib.sha256(str(skill_path).encode("utf-8")).hexdigest()[:12]
    return f"skill-{digest}"


def _skill_name_from_file(skill_path: Path) -> tuple[str, tuple[str, ...]]:
    try:
        text = skill_path.read_text(encoding="utf-8")
        frontmatter, _body = parse_skill_frontmatter(text)
    except Exception:
        return (
            _safe_derived_skill_name(skill_path),
            (f"invalid_skill_frontmatter:{skill_path}",),
        )

    raw_name = frontmatter.get("name") if isinstance(frontmatter, dict) else None
    if raw_name is not None:
        name = str(raw_name).strip()
        if name:
            return name, ()
    return _safe_derived_skill_name(skill_path), ()


def _discover_scope(
    root: Path,
    scope: str,
) -> tuple[list[DiscoveredClaudeSkill], list[str]]:
    if not root.exists():
        return [], []
    if not root.is_dir():
        return [], [f"claude_skill_root_not_directory:{root}"]

    skills: list[DiscoveredClaudeSkill] = []
    warnings: list[str] = []
    try:
        candidates = sorted(root.rglob("SKILL.md"), key=lambda path: path.as_posix())
    except OSError:
        return [], [f"claude_skill_discovery_failed:{root}"]

    for skill_path in candidates:
        if not skill_path.is_file():
            continue
        name, skill_warnings = _skill_name_from_file(skill_path)
        warnings.extend(skill_warnings)
        skills.append(
            DiscoveredClaudeSkill(
                name=name,
                scope=scope,
                path=str(skill_path),
                warnings=skill_warnings,
            )
        )
    return skills, warnings


def discover_claude_skills(
    project_root: str | Path,
    *,
    user_skills_root: str | Path | None = None,
) -> ClaudeSkillDiscovery:
    """Discover non-plugin project and user Claude Code skills."""
    project = Path(project_root).expanduser()
    project_skills_root = project / ".claude" / "skills"
    user_root = (
        Path(user_skills_root).expanduser()
        if user_skills_root is not None
        else Path.home() / ".claude" / "skills"
    )

    project_skills, project_warnings = _discover_scope(project_skills_root, "project")
    user_skills, user_warnings = _discover_scope(user_root, "user")
    skills = sorted(
        (*project_skills, *user_skills),
        key=lambda skill: (skill.name, skill.scope, skill.path),
    )
    return ClaudeSkillDiscovery(
        skills=tuple(skills),
        warnings=tuple((*project_warnings, *user_warnings)),
    )
