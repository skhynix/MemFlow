# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code native skill catalog settings management."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
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
CATALOG_STATE_SCHEMA_VERSION = "memflow.claude_catalog.v1"
DEFAULT_CATALOG_STATE_PATH = Path(".memflow") / "claude-catalog-state.json"
DEFAULT_CLAUDE_SETTINGS_PATH = Path(".claude") / "settings.local.json"


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


@dataclass(frozen=True)
class ClaudeCatalogSettingsPlan:
    mode: CatalogModeResolution
    settings_path: Path
    state_path: Path
    discovered_skills: tuple[DiscoveredClaudeSkill, ...]
    settings_before: dict[str, Any]
    settings_after: dict[str, Any]
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    managed_skill_overrides: tuple[str, ...]
    removed_managed_skill_overrides: tuple[str, ...]
    preserved_user_skill_overrides: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def settings_changed(self) -> bool:
        return self.settings_before != self.settings_after

    @property
    def state_changed(self) -> bool:
        return self.state_before != self.state_after

    @property
    def changed(self) -> bool:
        return self.settings_changed or self.state_changed

    def to_status(self, *, applied: bool = False) -> dict[str, Any]:
        return {
            "applied": applied,
            "changed": self.changed,
            "settings_changed": self.settings_changed,
            "state_changed": self.state_changed,
            "settings_path": str(self.settings_path),
            "state_path": str(self.state_path),
            "native_catalog_mode": {
                "raw": self.mode.raw_mode,
                "effective": self.mode.effective_mode,
            },
            "discovered_skills": [
                {
                    "name": skill.name,
                    "scope": skill.scope,
                    "path": skill.path,
                }
                for skill in self.discovered_skills
            ],
            "managed_skill_overrides": list(self.managed_skill_overrides),
            "removed_managed_skill_overrides": list(
                self.removed_managed_skill_overrides
            ),
            "preserved_user_skill_overrides": list(self.preserved_user_skill_overrides),
            "warnings": list(self.warnings),
            "limitations": ["plugin_skills_not_managed_by_skillOverrides"],
            "settings_after": self.settings_after,
        }


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


def _stable_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _write_json_object(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(_stable_json(data))
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return data


def _read_state(path: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    if not path.exists():
        return {}, ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, ("invalid_catalog_state_ignored",)
    if not isinstance(data, dict):
        return {}, ("invalid_catalog_state_ignored",)
    return data, ()


def _same_settings_target(left: Path, right: Path) -> bool:
    try:
        return left.expanduser().resolve(strict=False) == right.expanduser().resolve(
            strict=False
        )
    except (OSError, RuntimeError):
        return False


def _resolve_project_relative(path: str | Path, project_root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = project_root / resolved
    return resolved


def _path_for_state(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _validated_state_for_settings(
    state: dict[str, Any],
    *,
    project_root: Path,
    settings_path: Path,
    allow_managed_skill_overrides_salvage: bool = False,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if not state:
        return {}, ()

    warnings: list[str] = []
    schema_mismatch = state.get("schema_version") != CATALOG_STATE_SCHEMA_VERSION
    if schema_mismatch:
        warnings.append("catalog_state_schema_version_mismatch_ignored")

    state_settings_path = state.get("settings_path")
    settings_path_matches = False
    if not isinstance(state_settings_path, str) or not state_settings_path.strip():
        warnings.append("catalog_state_settings_path_mismatch_ignored")
    else:
        resolved_state_settings_path = _resolve_project_relative(
            state_settings_path,
            project_root,
        )
        settings_path_matches = _same_settings_target(
            resolved_state_settings_path,
            settings_path,
        )
        if not settings_path_matches:
            warnings.append("catalog_state_settings_path_mismatch_ignored")

    if warnings:
        managed_values = state.get("managed_skill_override_values")
        if (
            allow_managed_skill_overrides_salvage
            and schema_mismatch
            and settings_path_matches
            and isinstance(managed_values, dict)
            and all(
                isinstance(name, str) and isinstance(value, str)
                for name, value in managed_values.items()
            )
        ):
            managed_overrides = sorted(managed_values)
            return {
                "managed_skill_overrides": managed_overrides,
                "managed_skill_override_values": managed_values,
            }, tuple(warnings)
        return {}, tuple(warnings)
    return state, ()


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


def _managed_names_from_state(state: dict[str, Any]) -> set[str]:
    raw_names = state.get("managed_skill_overrides")
    if not isinstance(raw_names, list):
        return set()
    return {str(name) for name in raw_names if str(name)}


def _managed_override_values_from_state(state: dict[str, Any]) -> dict[str, str]:
    raw_values = state.get("managed_skill_override_values")
    if not isinstance(raw_values, dict):
        return {}
    values: dict[str, str] = {}
    for raw_name, raw_value in raw_values.items():
        if not isinstance(raw_name, str) or not raw_name:
            continue
        if not isinstance(raw_value, str) or not raw_value:
            continue
        values[raw_name] = raw_value
    return values


def _state_manages_disable_bundled_skills(state: dict[str, Any]) -> bool:
    return bool(state.get("disable_bundled_skills_managed"))


def _disable_bundled_skills_original_from_state(
    state: dict[str, Any],
) -> dict[str, Any] | None:
    original = state.get("disable_bundled_skills_original")
    if not isinstance(original, dict):
        return None
    present = original.get("present")
    if not isinstance(present, bool):
        return None
    if not present:
        return {"present": False}
    value = original.get("value")
    if not isinstance(value, bool):
        return None
    return {"present": True, "value": value}


def _disable_bundled_skills_original_from_settings(
    settings: dict[str, Any],
) -> dict[str, Any]:
    if "disableBundledSkills" not in settings:
        return {"present": False}
    value = settings["disableBundledSkills"]
    return (
        {"present": True, "value": value}
        if isinstance(value, bool)
        else {"present": False}
    )


def _has_existing_catalog_restrictions(settings: dict[str, Any]) -> bool:
    if settings.get("disableBundledSkills") is True:
        return True
    skill_overrides = settings.get("skillOverrides")
    return isinstance(skill_overrides, dict) and any(
        value in {"user-invocable-only", "off"} for value in skill_overrides.values()
    )


def _catalog_state(
    *,
    project_root: Path,
    settings_path: Path,
    mode: str,
    managed_skill_override_values: dict[str, str],
    disable_bundled_skills_managed: bool,
    disable_bundled_skills_original: dict[str, Any] | None = None,
) -> dict[str, Any]:
    managed_skill_overrides = tuple(sorted(managed_skill_override_values))
    state = {
        "schema_version": CATALOG_STATE_SCHEMA_VERSION,
        "settings_path": _path_for_state(settings_path, project_root),
        "native_catalog_mode": mode,
        "disable_bundled_skills_managed": disable_bundled_skills_managed,
        "managed_skill_overrides": list(managed_skill_overrides),
        "managed_skill_override_values": {
            name: managed_skill_override_values[name]
            for name in managed_skill_overrides
        },
    }
    if disable_bundled_skills_original is not None:
        state["disable_bundled_skills_original"] = disable_bundled_skills_original
    return state


def build_claude_catalog_settings_plan(
    config: dict[str, Any] | None = None,
    *,
    project_root: str | Path = ".",
    mode: str | None = None,
    settings_path: str | Path | None = None,
    state_path: str | Path | None = None,
    settings_before: dict[str, Any] | None = None,
    user_skills_root: str | Path | None = None,
) -> ClaudeCatalogSettingsPlan:
    """Build a deterministic Claude settings patch without writing it."""
    project = Path(project_root).expanduser()
    resolved_settings_path = _resolve_project_relative(
        settings_path or DEFAULT_CLAUDE_SETTINGS_PATH,
        project,
    )
    resolved_state_path = _resolve_project_relative(
        state_path or DEFAULT_CATALOG_STATE_PATH,
        project,
    )
    if mode is not None:
        mode_config: Any = {"claude": {"native_catalog_mode": mode}}
        mode_resolution = normalize_native_catalog_mode(mode_config)
    else:
        mode_resolution = _mode_resolution_from_metadata(config)
        if mode_resolution is None:
            mode_resolution = normalize_native_catalog_mode(config or {})
    discovery = discover_claude_skills(project, user_skills_root=user_skills_root)

    if settings_before is None:
        resolved_settings_before = _read_json_object(
            resolved_settings_path,
            label="Claude settings",
        )
    elif isinstance(settings_before, dict):
        resolved_settings_before = copy.deepcopy(settings_before)
    else:
        raise ValueError("Claude settings must be a JSON object")
    state_before, state_warnings = _read_state(resolved_state_path)
    trusted_state, validation_warnings = _validated_state_for_settings(
        state_before,
        project_root=project,
        settings_path=resolved_settings_path,
        allow_managed_skill_overrides_salvage=mode_resolution.effective_mode
        == "visible",
    )
    settings_after = copy.deepcopy(resolved_settings_before)
    previous_managed_values = _managed_override_values_from_state(trusted_state)
    previous_managed_names = _managed_names_from_state(trusted_state)
    previous_disable_bundled_original = _disable_bundled_skills_original_from_state(
        trusted_state
    )
    warnings = [
        *mode_resolution.warnings,
        *discovery.warnings,
        *state_warnings,
        *validation_warnings,
    ]
    catalog_state_is_trusted = bool(trusted_state) and not (
        state_warnings or validation_warnings
    )
    if not catalog_state_is_trusted and _has_existing_catalog_restrictions(
        resolved_settings_before
    ):
        warnings.append("catalog_state_unavailable_existing_catalog_settings_preserved")

    effective_mode = mode_resolution.effective_mode
    raw_overrides = settings_after.get("skillOverrides", {})
    skill_overrides: dict[str, Any] | None
    if raw_overrides is None:
        skill_overrides = {}
    elif isinstance(raw_overrides, dict):
        skill_overrides = dict(raw_overrides)
    else:
        warnings.append(
            "invalid_skill_overrides_ignored"
            if effective_mode == "visible"
            else "invalid_skill_overrides_replaced"
        )
        skill_overrides = {} if effective_mode != "visible" else None
    original_override_names = set(skill_overrides or {})

    removed_managed: list[str] = []
    preserved_manual_overrides: set[str] = set()
    if skill_overrides is not None:
        for name, expected_value in sorted(previous_managed_values.items()):
            current_value = skill_overrides.get(name)
            if current_value == expected_value:
                del skill_overrides[name]
                removed_managed.append(name)
            elif name in skill_overrides:
                preserved_manual_overrides.add(name)
                warnings.append(
                    f"managed_skill_override_manual_change_preserved:{name}"
                )
        for name in sorted(previous_managed_names - set(previous_managed_values)):
            if name in skill_overrides:
                preserved_manual_overrides.add(name)
                warnings.append(
                    f"managed_skill_override_value_missing_preserved:{name}"
                )

    managed_after: list[str] = []
    managed_after_values: dict[str, str] = {}
    preserved_user_overrides: list[str] = []
    disable_bundled_original: dict[str, Any] | None = None
    if effective_mode == "visible":
        if _state_manages_disable_bundled_skills(trusted_state):
            current_disable_bundled = settings_after.get("disableBundledSkills")
            if previous_disable_bundled_original is None:
                warnings.append("disable_bundled_skills_original_missing_preserved")
            elif current_disable_bundled is True:
                if previous_disable_bundled_original["present"]:
                    settings_after["disableBundledSkills"] = (
                        previous_disable_bundled_original["value"]
                    )
                else:
                    del settings_after["disableBundledSkills"]
            else:
                warnings.append("disable_bundled_skills_manual_change_preserved")
        disable_bundled_managed = False
    else:
        disable_bundled_original = (
            previous_disable_bundled_original
            or _disable_bundled_skills_original_from_settings(resolved_settings_before)
        )
        settings_after["disableBundledSkills"] = True
        override_value = (
            "off" if effective_mode == "disabled" else "user-invocable-only"
        )
        for name in sorted({skill.name for skill in discovery.skills}):
            if name in preserved_manual_overrides:
                preserved_user_overrides.append(name)
                continue
            if name in original_override_names and name not in previous_managed_values:
                preserved_user_overrides.append(name)
                continue
            if skill_overrides is None:
                continue
            skill_overrides[name] = override_value
            managed_after.append(name)
            managed_after_values[name] = override_value
        disable_bundled_managed = True

    if skill_overrides is not None:
        if skill_overrides:
            settings_after["skillOverrides"] = {
                key: skill_overrides[key] for key in sorted(skill_overrides)
            }
        elif "skillOverrides" in settings_after:
            del settings_after["skillOverrides"]

    managed_after_tuple = tuple(sorted(managed_after))
    if effective_mode == "visible" and not state_before:
        state_after = {}
    else:
        state_after = _catalog_state(
            project_root=project,
            settings_path=resolved_settings_path,
            mode=effective_mode,
            managed_skill_override_values=managed_after_values,
            disable_bundled_skills_managed=disable_bundled_managed,
            disable_bundled_skills_original=disable_bundled_original,
        )

    return ClaudeCatalogSettingsPlan(
        mode=mode_resolution,
        settings_path=resolved_settings_path,
        state_path=resolved_state_path,
        discovered_skills=discovery.skills,
        settings_before=resolved_settings_before,
        settings_after=settings_after,
        state_before=state_before,
        state_after=state_after,
        managed_skill_overrides=managed_after_tuple,
        removed_managed_skill_overrides=tuple(sorted(removed_managed)),
        preserved_user_skill_overrides=tuple(sorted(preserved_user_overrides)),
        warnings=tuple(warnings),
    )


def apply_claude_catalog_settings(plan: ClaudeCatalogSettingsPlan) -> None:
    """Apply a previously built Claude catalog settings plan."""
    if plan.settings_changed:
        _write_json_object(plan.settings_path, plan.settings_after)
    if plan.state_changed:
        _write_json_object(plan.state_path, plan.state_after)
