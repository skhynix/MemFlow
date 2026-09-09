# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Setup commands for MemFlow's Claude Code integration."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO

from memflow.claude_catalog import (
    DEFAULT_CATALOG_STATE_PATH,
    DEFAULT_CLAUDE_SETTINGS_PATH,
    SUPPORTED_NATIVE_CATALOG_MODES,
    ClaudeCatalogSettingsPlan,
    build_claude_catalog_settings_plan,
    normalize_native_catalog_mode,
)
from memflow.claude_hook import DEFAULT_CONFIG, DEFAULT_CONFIG_PATH
from memflow.claude_mcp import (
    MCPSettingsPlan,
    apply_mcp_settings_plan,
    build_mcp_settings_plan,
    local_mcp_server,
)
from memflow.skill_runtime import resolve_env_file

CLAUDE_HOOK_EVENT = "UserPromptSubmit"
MANAGED_HOOK_MARKER = "# memflow-managed:claude-hook"


@dataclass(frozen=True)
class HookSettingsPlan:
    settings_after: dict[str, Any]
    installed_before: bool
    installed_after: bool
    commands_before: tuple[str, ...]
    commands_after: tuple[str, ...]
    removed_count: int
    added: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClaudeSetupPlan:
    project_root: Path
    config_path: Path
    settings_path: Path
    state_path: Path
    config_before: dict[str, Any]
    config_after: dict[str, Any]
    settings_before: dict[str, Any]
    settings_after: dict[str, Any]
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    hook_action: str | None
    hook_command: str | None
    hook_plan: HookSettingsPlan
    catalog_mode: str | None
    catalog_plan: ClaudeCatalogSettingsPlan | None
    mcp_plan: MCPSettingsPlan | None = None
    warnings: tuple[str, ...] = ()

    @property
    def config_changed(self) -> bool:
        return self.config_before != self.config_after

    @property
    def settings_changed(self) -> bool:
        return self.settings_before != self.settings_after

    @property
    def state_changed(self) -> bool:
        return self.state_before != self.state_after

    @property
    def changed(self) -> bool:
        return (
            self.config_changed
            or self.settings_changed
            or self.state_changed
            or (self.mcp_plan is not None and self.mcp_plan.changed)
        )

    def to_status(self, *, applied: bool) -> dict[str, Any]:
        mode = normalize_native_catalog_mode(self.config_after)
        return {
            "applied": applied,
            "changed": self.changed,
            "config_changed": self.config_changed,
            "settings_changed": self.settings_changed,
            "state_changed": self.state_changed,
            "project_root": str(self.project_root),
            "config_path": str(self.config_path),
            "settings_path": str(self.settings_path),
            "state_path": str(self.state_path),
            "env_file": str(
                resolve_env_file(
                    project_root=self.project_root, config=self.config_after
                )
            ),
            "mcp": self.mcp_plan.to_status() if self.mcp_plan is not None else None,
            "hook": {
                "requested": self.hook_action,
                "installed_before": self.hook_plan.installed_before,
                "installed_after": self.hook_plan.installed_after,
                "commands_before": list(self.hook_plan.commands_before),
                "commands_after": list(self.hook_plan.commands_after),
                "command": self.hook_command,
                "removed": self.hook_plan.removed_count,
                "added": self.hook_plan.added,
            },
            "native_catalog_mode": {
                "requested": self.catalog_mode,
                "raw": mode.raw_mode,
                "effective": mode.effective_mode,
            },
            "applied_catalog_settings": _catalog_settings_view(self.settings_after),
            "managed_skill_overrides": list(
                self.catalog_plan.managed_skill_overrides
                if self.catalog_plan is not None
                else ()
            ),
            "removed_managed_skill_overrides": list(
                self.catalog_plan.removed_managed_skill_overrides
                if self.catalog_plan is not None
                else ()
            ),
            "state_after": self.state_after,
            "warnings": list(self.warnings),
            "limitations": ["plugin_skills_not_managed_by_skillOverrides"],
        }


def _stable_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


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


def _read_catalog_state_for_status(
    path: Path,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if not path.exists():
        return {}, ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, ("invalid_catalog_state_ignored",)
    if not isinstance(data, dict):
        return {}, ("invalid_catalog_state_ignored",)
    return data, ()


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


def _resolve_project_root(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve(strict=False)


def _resolve_project_relative(path: str | Path, project_root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = project_root / resolved
    return resolved


def _path_options(
    *,
    project_root: str | Path = ".",
    config_path: str | Path | None = None,
    settings_path: str | Path | None = None,
    state_path: str | Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    project = _resolve_project_root(project_root)
    return (
        project,
        _resolve_project_relative(config_path or DEFAULT_CONFIG_PATH, project),
        _resolve_project_relative(
            settings_path or DEFAULT_CLAUDE_SETTINGS_PATH,
            project,
        ),
        _resolve_project_relative(state_path or DEFAULT_CATALOG_STATE_PATH, project),
    )


def default_hook_command(config_path: str | Path) -> str:
    quoted_python = shlex.quote(sys.executable)
    quoted_config = shlex.quote(str(config_path))
    return f"{quoted_python} -m memflow.claude_hook --config {quoted_config}"


def _marked_hook_command(command: str) -> str:
    command = command.strip()
    if MANAGED_HOOK_MARKER in command:
        return command
    return f"{command} {MANAGED_HOOK_MARKER}"


def _is_managed_hook_command(command: Any) -> bool:
    if not isinstance(command, str):
        return False
    return MANAGED_HOOK_MARKER in command


def _is_managed_command_hook(hook: Any) -> bool:
    return (
        isinstance(hook, dict)
        and hook.get("type") == "command"
        and _is_managed_hook_command(hook.get("command"))
    )


def _collect_hook_commands(
    settings: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    hooks = settings.get("hooks")
    if hooks is None:
        return (), ()
    if not isinstance(hooks, dict):
        return (), ("invalid_hooks_ignored",)

    entries = hooks.get(CLAUDE_HOOK_EVENT, [])
    if entries is None:
        return (), ()
    if not isinstance(entries, list):
        return (), ("invalid_user_prompt_submit_hooks_ignored",)

    commands: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            warnings.append("invalid_user_prompt_submit_hook_entry_ignored")
            continue
        entry_hooks = entry.get("hooks", [])
        if not isinstance(entry_hooks, list):
            warnings.append("invalid_user_prompt_submit_hook_commands_ignored")
            continue
        for hook in entry_hooks:
            command = hook.get("command") if isinstance(hook, dict) else None
            if isinstance(command, str) and _is_managed_command_hook(hook):
                commands.append(command)
    return tuple(commands), tuple(dict.fromkeys(warnings))


def _remove_managed_hook_entries(
    settings: dict[str, Any],
    *,
    create: bool,
) -> tuple[dict[str, Any], int, tuple[str, ...]]:
    settings_after = copy.deepcopy(settings)
    warnings: list[str] = []
    hooks = settings_after.get("hooks")
    if hooks is None:
        if create:
            hooks = {}
            settings_after["hooks"] = hooks
        else:
            return settings_after, 0, ()
    elif not isinstance(hooks, dict):
        if not create:
            return settings_after, 0, ("invalid_hooks_ignored",)
        hooks = {}
        settings_after["hooks"] = hooks
        warnings.append("invalid_hooks_replaced")

    entries = hooks.get(CLAUDE_HOOK_EVENT, [])
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        if not create:
            return settings_after, 0, ("invalid_user_prompt_submit_hooks_ignored",)
        entries = []
        warnings.append("invalid_user_prompt_submit_hooks_replaced")

    removed = 0
    kept_entries: list[Any] = []
    for entry in entries:
        if not isinstance(entry, dict):
            kept_entries.append(entry)
            continue

        entry_hooks = entry.get("hooks", [])
        if not isinstance(entry_hooks, list):
            kept_entries.append(entry)
            continue

        kept_hooks = []
        for hook in entry_hooks:
            if _is_managed_command_hook(hook):
                removed += 1
                continue
            kept_hooks.append(hook)

        if len(kept_hooks) == len(entry_hooks):
            kept_entries.append(entry)
        elif kept_hooks:
            updated_entry = dict(entry)
            updated_entry["hooks"] = kept_hooks
            kept_entries.append(updated_entry)

    if kept_entries:
        hooks[CLAUDE_HOOK_EVENT] = kept_entries
    elif CLAUDE_HOOK_EVENT in hooks:
        del hooks[CLAUDE_HOOK_EVENT]

    if not hooks:
        del settings_after["hooks"]

    return settings_after, removed, tuple(warnings)


def _install_hook_entry(settings: dict[str, Any], command: str) -> dict[str, Any]:
    settings_after = copy.deepcopy(settings)
    hooks = settings_after.setdefault("hooks", {})
    entries = hooks.setdefault(CLAUDE_HOOK_EVENT, [])
    entries.append(
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": command,
                }
            ],
        }
    )
    return settings_after


def build_hook_settings_plan(
    settings_before: dict[str, Any],
    *,
    action: str | None,
    command: str | None = None,
) -> HookSettingsPlan:
    commands_before, collect_warnings = _collect_hook_commands(settings_before)
    settings_after = copy.deepcopy(settings_before)
    warnings: list[str] = [*collect_warnings]
    removed = 0
    added = False

    if action == "off":
        settings_after, removed, remove_warnings = _remove_managed_hook_entries(
            settings_after,
            create=False,
        )
        warnings.extend(remove_warnings)
    elif action == "on":
        if command is None:
            raise ValueError("hook command is required when enabling the hook")
        settings_after, removed, remove_warnings = _remove_managed_hook_entries(
            settings_after,
            create=True,
        )
        warnings.extend(remove_warnings)
        settings_after = _install_hook_entry(settings_after, command)
        added = True
    elif action is not None:
        raise ValueError(f"unsupported hook action: {action}")

    commands_after, after_warnings = _collect_hook_commands(settings_after)
    warnings.extend(after_warnings)
    return HookSettingsPlan(
        settings_after=settings_after,
        installed_before=bool(commands_before),
        installed_after=bool(commands_after),
        commands_before=commands_before,
        commands_after=commands_after,
        removed_count=removed,
        added=added,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _config_for_edit(
    config_before: dict[str, Any],
    *,
    config_exists: bool,
    should_create: bool,
    include_catalog_defaults: bool,
) -> dict[str, Any]:
    if config_exists:
        return copy.deepcopy(config_before)
    if should_create:
        config = copy.deepcopy(DEFAULT_CONFIG)
        if not include_catalog_defaults:
            config.pop("claude", None)
        return config
    return {}


def _has_explicit_catalog_mode(config: dict[str, Any]) -> bool:
    claude_config = config.get("claude")
    return isinstance(claude_config, dict) and "native_catalog_mode" in claude_config


def _set_catalog_mode(
    config: dict[str, Any],
    mode: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    config_after = copy.deepcopy(config)
    claude_config = config_after.get("claude")
    warnings: tuple[str, ...] = ()
    if not isinstance(claude_config, dict):
        if claude_config is not None:
            warnings = ("invalid_claude_config_replaced",)
        claude_config = {}
    else:
        claude_config = dict(claude_config)
    claude_config["native_catalog_mode"] = mode
    config_after["claude"] = claude_config
    return config_after, warnings


def build_claude_setup_plan(
    *,
    project_root: str | Path = ".",
    config_path: str | Path | None = None,
    settings_path: str | Path | None = None,
    state_path: str | Path | None = None,
    hook: str | None = None,
    catalog: str | None = None,
    hook_command: str | None = None,
    env_file: str | Path | None = None,
    mcp: str | None = None,
) -> ClaudeSetupPlan:
    if hook is None and catalog is None and mcp is None and env_file is None:
        hook = mcp = "on"

    project, resolved_config, resolved_settings, resolved_state = _path_options(
        project_root=project_root,
        config_path=config_path,
        settings_path=settings_path,
        state_path=state_path,
    )
    config_before = _read_json_object(resolved_config, label="Claude hook config")
    settings_before = _read_json_object(resolved_settings, label="Claude settings")
    config_exists = resolved_config.exists()

    config_after = _config_for_edit(
        config_before,
        config_exists=config_exists,
        should_create=hook == "on"
        or mcp == "on"
        or catalog is not None
        or env_file is not None,
        include_catalog_defaults=catalog is not None,
    )
    warnings: list[str] = []
    if env_file is not None or hook == "on" or mcp == "on":
        resolved_env = resolve_env_file(
            env_file, project_root=project, config=config_before
        )
        if not resolved_env.is_file():
            raise ValueError(f"environment file is not a file: {resolved_env}")
        memflow_config = config_after.setdefault("memflow", {})
        if not isinstance(memflow_config, dict):
            raise ValueError("skill config memflow must be a JSON object")
        memflow_config["env_file"] = str(resolved_env)

    mcp_plan = None
    if mcp is not None:
        claude_config = config_after.get("claude", {})
        if not isinstance(claude_config, dict):
            raise ValueError("skill config claude must be a JSON object")
        server = {
            "type": "stdio",
            "command": sys.executable,
            "args": ["-m", "memflow.mcp_server", "--config", str(resolved_config)],
            "env": {},
        }
        mcp_plan = build_mcp_settings_plan(
            project,
            action=mcp,
            server=server,
            managed_server=claude_config.get("mcp_server"),
        )
        if mcp == "on":
            claude_config["mcp_server"] = server
            config_after["claude"] = claude_config
        else:
            claude_config.pop("mcp_server", None)
    if catalog is not None:
        config_after, config_warnings = _set_catalog_mode(config_after, catalog)
        warnings.extend(config_warnings)

    if catalog is not None:
        catalog_plan = build_claude_catalog_settings_plan(
            config_after,
            project_root=project,
            mode=catalog,
            settings_path=resolved_settings,
            state_path=resolved_state,
            settings_before=settings_before,
        )
        settings_after = catalog_plan.settings_after
        state_before = catalog_plan.state_before
        state_after = catalog_plan.state_after
        warnings.extend(catalog_plan.warnings)
    else:
        catalog_plan = None
        settings_after = copy.deepcopy(settings_before)
        state_before = {}
        state_after = {}

    desired_hook_command = None
    if hook == "on":
        desired_hook_command = _marked_hook_command(
            hook_command or default_hook_command(resolved_config)
        )
    elif hook_command:
        desired_hook_command = _marked_hook_command(hook_command)

    hook_plan = build_hook_settings_plan(
        settings_after,
        action=hook,
        command=desired_hook_command,
    )
    warnings.extend(hook_plan.warnings)

    return ClaudeSetupPlan(
        project_root=project,
        config_path=resolved_config,
        settings_path=resolved_settings,
        state_path=resolved_state,
        config_before=config_before,
        config_after=config_after,
        settings_before=settings_before,
        settings_after=hook_plan.settings_after,
        state_before=state_before,
        state_after=state_after,
        hook_action=hook,
        hook_command=desired_hook_command,
        hook_plan=hook_plan,
        catalog_mode=catalog,
        catalog_plan=catalog_plan,
        mcp_plan=mcp_plan,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def apply_claude_setup_plan(plan: ClaudeSetupPlan) -> None:
    if plan.mcp_plan is not None:
        apply_mcp_settings_plan(plan.mcp_plan)
    if plan.config_changed:
        _write_json_object(plan.config_path, plan.config_after)
    if plan.settings_changed:
        _write_json_object(plan.settings_path, plan.settings_after)
    if plan.state_changed:
        _write_json_object(plan.state_path, plan.state_after)


def _catalog_settings_view(settings: dict[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {}
    if "disableBundledSkills" in settings:
        view["disableBundledSkills"] = settings["disableBundledSkills"]
    if "skillOverrides" in settings:
        view["skillOverrides"] = settings["skillOverrides"]
    return view


def build_status(
    *,
    project_root: str | Path = ".",
    config_path: str | Path | None = None,
    settings_path: str | Path | None = None,
    state_path: str | Path | None = None,
) -> dict[str, Any]:
    project, resolved_config, resolved_settings, resolved_state = _path_options(
        project_root=project_root,
        config_path=config_path,
        settings_path=settings_path,
        state_path=state_path,
    )
    config = _read_json_object(resolved_config, label="Claude hook config")
    settings = _read_json_object(resolved_settings, label="Claude settings")
    mode = normalize_native_catalog_mode(config)
    commands, hook_warnings = _collect_hook_commands(settings)

    mismatches: list[str] = []
    catalog_warnings: tuple[str, ...] = ()
    state_before: dict[str, Any] = {}
    desired_catalog_settings: dict[str, Any] = {}
    managed_skill_overrides: tuple[str, ...] = ()
    removed_managed_skill_overrides: tuple[str, ...] = ()
    if _has_explicit_catalog_mode(config):
        catalog_plan = build_claude_catalog_settings_plan(
            config,
            project_root=project,
            settings_path=resolved_settings,
            state_path=resolved_state,
            settings_before=settings,
        )
        state_before = catalog_plan.state_before
        desired_catalog_settings = _catalog_settings_view(catalog_plan.settings_after)
        managed_skill_overrides = catalog_plan.managed_skill_overrides
        removed_managed_skill_overrides = catalog_plan.removed_managed_skill_overrides
        catalog_warnings = catalog_plan.warnings
        if catalog_plan.settings_changed:
            mismatches.append("catalog_settings")
        if catalog_plan.state_changed:
            mismatches.append("catalog_state")
    else:
        state_before, catalog_warnings = _read_catalog_state_for_status(resolved_state)

    warnings = [*mode.warnings, *catalog_warnings, *hook_warnings]
    configured_env = resolve_env_file(project_root=project, config=config)
    mcp_server = local_mcp_server(project)
    claude_config = config.get("claude", {})
    expected_mcp = (
        claude_config.get("mcp_server") if isinstance(claude_config, dict) else None
    )
    if expected_mcp is not None and mcp_server != expected_mcp:
        mismatches.append("mcp_settings")
    return {
        "project_root": str(project),
        "config_path": str(resolved_config),
        "settings_path": str(resolved_settings),
        "state_path": str(resolved_state),
        "env_file": str(configured_env),
        "env_file_exists": configured_env.is_file(),
        "mcp": {
            "scope": "local",
            "installed": mcp_server is not None,
            "matches_config": mcp_server == expected_mcp
            if expected_mcp is not None
            else None,
        },
        "hook": {
            "installed": bool(commands),
            "commands": list(commands),
        },
        "native_catalog_mode": {
            "raw": mode.raw_mode,
            "effective": mode.effective_mode,
        },
        "applied_catalog_settings": _catalog_settings_view(settings),
        "desired_catalog_settings": desired_catalog_settings,
        "managed_skill_overrides": list(managed_skill_overrides),
        "removed_managed_skill_overrides": list(removed_managed_skill_overrides),
        "state": {
            "exists": resolved_state.exists(),
            "data": state_before,
        },
        "mismatches": mismatches,
        "warnings": list(dict.fromkeys(warnings)),
        "limitations": ["plugin_skills_not_managed_by_skillOverrides"],
    }


def _write_status(stdout: TextIO, status: dict[str, Any]) -> None:
    stdout.write(_stable_json(status))


def _add_common_path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-root",
        default=".",
        help="target project root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--config-path",
        help="hook config JSON path. Defaults to .memflow/claude-hook.json.",
    )
    parser.add_argument(
        "--settings-path",
        help="Claude settings JSON path. Defaults to .claude/settings.local.json.",
    )
    parser.add_argument(
        "--state-path",
        help="catalog state JSON path. Defaults to .memflow/claude-catalog-state.json.",
    )


def _run_status(
    args: argparse.Namespace, *, stdout: TextIO, stderr: TextIO, **_: Any
) -> int:
    try:
        status = build_status(
            project_root=args.project_root,
            config_path=args.config_path,
            settings_path=args.settings_path,
            state_path=args.state_path,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=stderr)
        return 1
    _write_status(stdout, status)
    return 0


def _run_configure(
    args: argparse.Namespace, *, stdout: TextIO, stderr: TextIO, **_: Any
) -> int:
    try:
        plan = build_claude_setup_plan(
            project_root=args.project_root,
            config_path=args.config_path,
            settings_path=args.settings_path,
            state_path=args.state_path,
            hook=args.hook,
            catalog=args.catalog,
            hook_command=args.hook_command,
            env_file=args.env_file,
            mcp=args.mcp,
        )
        status = plan.to_status(applied=args.apply)
        if args.apply:
            apply_claude_setup_plan(plan)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=stderr)
        return 1
    _write_status(stdout, status)
    return 0


def add_claude_subcommands(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="claude_command", metavar="command")

    status = subparsers.add_parser("status", help="show Claude integration status")
    _add_common_path_args(status)
    status.set_defaults(handler=_run_status)

    configure = subparsers.add_parser(
        "configure",
        help="set up the Claude hook and MCP server, or change individual settings",
        description=(
            "Enable the hook and project-local MCP server when no --hook, --mcp, "
            "--catalog, or --env-file option is given. "
            "Use those options to change individual settings. "
            "To activate both integrations with a custom environment file, "
            "use --hook on --mcp on --env-file PATH. "
            "Preview changes unless --apply is given."
        ),
    )
    _add_common_path_args(configure)
    configure.add_argument("--hook", choices=("on", "off"))
    configure.add_argument("--mcp", choices=("on", "off"))
    configure.add_argument(
        "--env-file",
        help=(
            "save this environment file for the project without changing hook, "
            "MCP, or catalog settings unless requested separately"
        ),
    )
    configure.add_argument("--catalog", choices=sorted(SUPPORTED_NATIVE_CATALOG_MODES))
    configure.add_argument(
        "--hook-command",
        help="command string to install for the MemFlow UserPromptSubmit hook",
    )
    action = configure.add_mutually_exclusive_group()
    action.add_argument(
        "--apply",
        action="store_true",
        help="write the planned config, settings, and state changes",
    )
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="preview changes without writing. This is the default.",
    )
    configure.set_defaults(handler=_run_configure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memflow claude",
        description="Manage MemFlow's Claude Code integration.",
    )
    add_claude_subcommands(parser)
    return parser


def main(
    argv: Iterable[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out = stdout or sys.stdout
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help(file=out)
        return 0
    return handler(args, stdout=out, stderr=stderr or sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
