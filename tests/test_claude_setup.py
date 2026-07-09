# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from memflow.claude_setup import MANAGED_HOOK_MARKER, build_hook_settings_plan


def test_build_hook_settings_plan_hook_on_installs_marked_command_and_preserves_settings():
    command = f"python -m memflow.claude_hook {MANAGED_HOOK_MARKER}"
    settings_before = {
        "theme": "dark",
        "hooks": {
            "UserPromptSubmit": [
                {
                    "matcher": "",
                    "hooks": [{"type": "command", "command": "echo user hook"}],
                }
            ],
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": "echo pre tool"}],
                }
            ],
        },
    }

    plan = build_hook_settings_plan(settings_before, action="on", command=command)

    assert plan.installed_before is False
    assert plan.installed_after is True
    assert plan.commands_after == (command,)
    assert plan.added is True
    assert plan.removed_count == 0
    assert plan.settings_after["theme"] == "dark"
    assert plan.settings_after["hooks"]["PreToolUse"] == [
        {
            "matcher": "Bash",
            "hooks": [{"type": "command", "command": "echo pre tool"}],
        }
    ]
    assert plan.settings_after["hooks"]["UserPromptSubmit"] == [
        {
            "matcher": "",
            "hooks": [{"type": "command", "command": "echo user hook"}],
        },
        {
            "matcher": "",
            "hooks": [{"type": "command", "command": command}],
        },
    ]


def test_build_hook_settings_plan_hook_off_removes_only_marker_managed_hooks():
    marked_command = f"uv run -m memflow.claude_hook {MANAGED_HOOK_MARKER}"
    unmarked_command = "uv run -m memflow.claude_hook --config user-owned.json"
    settings_before = {
        "hooks": {
            "UserPromptSubmit": [
                {
                    "matcher": "",
                    "hooks": [
                        {"type": "command", "command": "echo user hook"},
                        {"type": "command", "command": unmarked_command},
                        {"type": "command", "command": marked_command},
                    ],
                }
            ],
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": "echo pre tool"}],
                }
            ],
        }
    }

    plan = build_hook_settings_plan(settings_before, action="off")

    assert plan.installed_before is True
    assert plan.installed_after is False
    assert plan.commands_before == (marked_command,)
    assert plan.commands_after == ()
    assert plan.removed_count == 1
    assert plan.added is False
    assert plan.settings_after["hooks"]["UserPromptSubmit"] == [
        {
            "matcher": "",
            "hooks": [
                {"type": "command", "command": "echo user hook"},
                {"type": "command", "command": unmarked_command},
            ],
        }
    ]
    assert plan.settings_after["hooks"]["PreToolUse"] == [
        {
            "matcher": "Bash",
            "hooks": [{"type": "command", "command": "echo pre tool"}],
        }
    ]


def test_build_hook_settings_plan_ignores_missing_and_non_string_hook_commands():
    settings_before = {
        "hooks": {
            "UserPromptSubmit": [
                {
                    "matcher": "",
                    "hooks": [
                        {"type": "command"},
                        {
                            "type": "command",
                            "command": ["python", "-m", "memflow.claude_hook"],
                        },
                    ],
                }
            ]
        }
    }

    plan = build_hook_settings_plan(settings_before, action=None)

    assert plan.installed_before is False
    assert plan.installed_after is False
    assert plan.commands_before == ()
    assert plan.commands_after == ()
    assert plan.settings_after == settings_before
