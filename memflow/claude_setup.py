# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Setup helpers for MemFlow's Claude Code integration."""

from __future__ import annotations

import copy
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
