# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import shlex
import sys

from memflow.claude_setup import MANAGED_HOOK_MARKER, build_hook_settings_plan
from memflow.cli import main as cli_main


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


def _run_memflow_claude(args, *, tmp_path):
    stdout = io.StringIO()
    rc = cli_main(["claude", *args, "--project-root", str(tmp_path)], stdout=stdout)
    return rc, json.loads(stdout.getvalue())


def _memflow_hook_commands(settings):
    commands = []
    for entry in settings.get("hooks", {}).get("UserPromptSubmit", []):
        for hook in entry.get("hooks", []):
            command = hook.get("command")
            if isinstance(command, str) and MANAGED_HOOK_MARKER in command:
                commands.append(command)
    return commands


def _marked_default_hook_command(config_path):
    return (
        f"{shlex.quote(sys.executable)} -m memflow.claude_hook "
        f"--config {shlex.quote(str(config_path))} {MANAGED_HOOK_MARKER}"
    )


def test_configure_catalog_disabled_apply_updates_config_settings_and_state(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path / ".claude" / "skills" / "project-skill")

    rc, status = _run_memflow_claude(
        ["configure", "--catalog", "disabled", "--apply"],
        tmp_path=tmp_path,
    )

    config = _read_json(tmp_path / ".memflow" / "claude-hook.json")
    settings = _read_json(tmp_path / ".claude" / "settings.local.json")
    state = _read_json(tmp_path / ".memflow" / "claude-catalog-state.json")
    assert rc == 0
    assert status["applied"] is True
    assert config["claude"]["native_catalog_mode"] == "disabled"
    assert settings["disableBundledSkills"] is True
    assert settings["skillOverrides"] == {"project-skill": "off"}
    assert state["native_catalog_mode"] == "disabled"
    assert state["managed_skill_overrides"] == ["project-skill"]


def test_configure_dry_run_does_not_write_config_settings_or_state(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path / ".claude" / "skills" / "project-skill")

    rc, status = _run_memflow_claude(
        ["configure", "--hook", "on", "--catalog", "disabled"],
        tmp_path=tmp_path,
    )

    assert rc == 0
    assert status["applied"] is False
    assert status["config_changed"] is True
    assert status["settings_changed"] is True
    assert not (tmp_path / ".memflow" / "claude-hook.json").exists()
    assert not (tmp_path / ".claude" / "settings.local.json").exists()
    assert not (tmp_path / ".memflow" / "claude-catalog-state.json").exists()


def test_hook_on_adds_user_prompt_submit_hook_and_preserves_settings(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    settings_path = tmp_path / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )

    rc, status = _run_memflow_claude(
        ["configure", "--hook", "on", "--apply"],
        tmp_path=tmp_path,
    )

    settings = _read_json(settings_path)
    expected_command = _marked_default_hook_command(
        tmp_path / ".memflow" / "claude-hook.json"
    )
    assert rc == 0
    assert status["hook"]["installed_after"] is True
    assert status["hook"]["command"] == expected_command
    assert settings["theme"] == "dark"
    assert settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "echo pre tool"
    user_prompt_commands = [
        hook["command"]
        for entry in settings["hooks"]["UserPromptSubmit"]
        for hook in entry["hooks"]
    ]
    assert "echo user hook" in user_prompt_commands
    assert _memflow_hook_commands(settings) == [expected_command]


def test_hook_only_setup_does_not_create_catalog_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path / ".claude" / "skills" / "project-skill")

    rc, configure_status = _run_memflow_claude(
        ["configure", "--hook", "on", "--apply"],
        tmp_path=tmp_path,
    )
    status_rc, status = _run_memflow_claude(["status"], tmp_path=tmp_path)

    config = _read_json(tmp_path / ".memflow" / "claude-hook.json")
    assert rc == 0
    assert status_rc == 0
    assert configure_status["hook"]["installed_after"] is True
    assert "native_catalog_mode" not in config.get("claude", {})
    assert status["hook"]["installed"] is True
    assert status["desired_catalog_settings"] == {}
    assert status["mismatches"] == []
    assert not (tmp_path / ".memflow" / "claude-catalog-state.json").exists()


def test_hook_command_override_is_marked_and_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    rc, status = _run_memflow_claude(
        [
            "configure",
            "--hook",
            "on",
            "--hook-command",
            "python custom-hook.py",
            "--apply",
        ],
        tmp_path=tmp_path,
    )

    settings = _read_json(tmp_path / ".claude" / "settings.local.json")
    expected_command = f"python custom-hook.py {MANAGED_HOOK_MARKER}"
    assert rc == 0
    assert status["hook"]["command"] == expected_command
    assert _memflow_hook_commands(settings) == [expected_command]


def test_hook_on_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    _run_memflow_claude(["configure", "--hook", "on", "--apply"], tmp_path=tmp_path)
    settings_path = tmp_path / ".claude" / "settings.local.json"
    first_settings = _read_json(settings_path)

    _run_memflow_claude(["configure", "--hook", "on", "--apply"], tmp_path=tmp_path)

    second_settings = _read_json(settings_path)
    assert second_settings == first_settings
    assert len(_memflow_hook_commands(second_settings)) == 1


def test_hook_off_removes_only_marked_memflow_hook(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    settings_path = tmp_path / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    marked_command = (
        "uv run -m memflow.claude_hook --config .memflow/claude-hook.json "
        f"{MANAGED_HOOK_MARKER}"
    )
    unmarked_user_command = "uv run -m memflow.claude_hook --config user-owned.json"
    settings_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "UserPromptSubmit": [
                        {
                            "matcher": "",
                            "hooks": [
                                {"type": "command", "command": "echo user hook"},
                                {
                                    "type": "command",
                                    "command": unmarked_user_command,
                                },
                                {
                                    "type": "command",
                                    "command": marked_command,
                                },
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
        ),
        encoding="utf-8",
    )

    rc, status = _run_memflow_claude(
        ["configure", "--hook", "off", "--apply"],
        tmp_path=tmp_path,
    )

    settings = _read_json(settings_path)
    assert rc == 0
    assert status["hook"]["installed_after"] is False
    assert _memflow_hook_commands(settings) == []
    assert settings["hooks"]["UserPromptSubmit"][0]["hooks"] == [
        {"type": "command", "command": "echo user hook"},
        {"type": "command", "command": unmarked_user_command},
    ]
    assert settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "echo pre tool"


def test_missing_and_non_string_hook_commands_are_not_reported_installed(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    settings_path = tmp_path / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "UserPromptSubmit": [
                        {
                            "matcher": "",
                            "hooks": [
                                {"type": "command"},
                                {
                                    "type": "command",
                                    "command": [
                                        "python",
                                        "-m",
                                        "memflow.claude_hook",
                                    ],
                                },
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    rc, status = _run_memflow_claude(["status"], tmp_path=tmp_path)

    assert rc == 0
    assert status["hook"]["installed"] is False
    assert status["hook"]["commands"] == []


def test_combined_hook_and_catalog_configure_merges_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path / ".claude" / "skills" / "project-skill")
    settings_path = tmp_path / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            {
                "theme": "dark",
                "hooks": {
                    "UserPromptSubmit": [
                        {
                            "matcher": "",
                            "hooks": [{"type": "command", "command": "echo user hook"}],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    rc, status = _run_memflow_claude(
        ["configure", "--hook", "on", "--catalog", "disabled", "--apply"],
        tmp_path=tmp_path,
    )

    settings = _read_json(settings_path)
    assert rc == 0
    assert status["applied"] is True
    assert settings["theme"] == "dark"
    assert settings["disableBundledSkills"] is True
    assert settings["skillOverrides"] == {"project-skill": "off"}
    assert len(_memflow_hook_commands(settings)) == 1
    assert any(
        hook["command"] == "echo user hook"
        for entry in settings["hooks"]["UserPromptSubmit"]
        for hook in entry["hooks"]
    )


def test_status_reports_hook_catalog_state_and_mismatches(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path / ".claude" / "skills" / "project-skill")
    _run_memflow_claude(
        ["configure", "--hook", "on", "--catalog", "disabled", "--apply"],
        tmp_path=tmp_path,
    )

    rc, status = _run_memflow_claude(["status"], tmp_path=tmp_path)

    assert rc == 0
    assert status["hook"]["installed"] is True
    assert status["native_catalog_mode"]["effective"] == "disabled"
    assert status["applied_catalog_settings"]["disableBundledSkills"] is True
    assert status["applied_catalog_settings"]["skillOverrides"] == {
        "project-skill": "off"
    }
    assert status["state"]["exists"] is True
    assert status["managed_skill_overrides"] == ["project-skill"]
    assert status["mismatches"] == []


def test_status_reports_catalog_drift_when_skill_is_added_after_apply(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    configure_rc, _ = _run_memflow_claude(
        ["configure", "--catalog", "disabled", "--apply"],
        tmp_path=tmp_path,
    )
    _write_skill(tmp_path / ".claude" / "skills" / "late-skill")

    status_rc, status = _run_memflow_claude(["status"], tmp_path=tmp_path)

    assert configure_rc == 0
    assert status_rc == 0
    assert status["applied_catalog_settings"] == {"disableBundledSkills": True}
    assert status["desired_catalog_settings"] == {
        "disableBundledSkills": True,
        "skillOverrides": {"late-skill": "off"},
    }
    assert status["managed_skill_overrides"] == ["late-skill"]
    assert status["mismatches"] == ["catalog_settings", "catalog_state"]


def test_missing_catalog_state_warns_when_existing_settings_are_preserved(
    tmp_path,
    monkeypatch,
):
    warning = "catalog_state_unavailable_existing_catalog_settings_preserved"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path / ".claude" / "skills" / "project-skill")
    configure_rc, _ = _run_memflow_claude(
        ["configure", "--catalog", "hidden_or_minimized", "--apply"],
        tmp_path=tmp_path,
    )
    state_path = tmp_path / ".memflow" / "claude-catalog-state.json"
    state_path.unlink()

    status_rc, status = _run_memflow_claude(["status"], tmp_path=tmp_path)
    disabled_rc, disabled_status = _run_memflow_claude(
        ["configure", "--catalog", "disabled", "--apply"],
        tmp_path=tmp_path,
    )

    settings = _read_json(tmp_path / ".claude" / "settings.local.json")
    assert configure_rc == 0
    assert status_rc == 0
    assert status["warnings"].count(warning) == 1
    assert disabled_rc == 0
    assert disabled_status["applied"] is True
    assert disabled_status["native_catalog_mode"]["effective"] == "disabled"
    assert disabled_status["warnings"].count(warning) == 1
    assert disabled_status["applied_catalog_settings"] == {
        "disableBundledSkills": True,
        "skillOverrides": {"project-skill": "user-invocable-only"},
    }
    assert settings["skillOverrides"] == {"project-skill": "user-invocable-only"}


def test_status_reports_invalid_config_catalog_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config_path = tmp_path / ".memflow" / "claude-hook.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps({"claude": {"native_catalog_mode": "bad"}}),
        encoding="utf-8",
    )

    rc, status = _run_memflow_claude(["status"], tmp_path=tmp_path)

    assert rc == 0
    assert status["native_catalog_mode"] == {
        "raw": "bad",
        "effective": "hidden_or_minimized",
    }
    assert "invalid_native_catalog_mode" in status["warnings"]
