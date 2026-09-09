# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Register the project's private MCP server through the Claude Code CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def claude_config_path() -> Path:
    config_dir = os.getenv("CLAUDE_CONFIG_DIR")
    return (
        Path(config_dir).expanduser().resolve() / ".claude.json"
        if config_dir
        else Path.home() / ".claude.json"
    )


def _read_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Claude MCP config must be valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Claude MCP config must be a JSON object: {path}")
    return value


def _object_field(value: dict[str, Any], name: str) -> dict[str, Any]:
    result = value.get(name, {})
    if not isinstance(result, dict):
        raise ValueError(f"Claude MCP config {name} must be a JSON object")
    return result


def local_mcp_server(project: Path) -> dict[str, Any] | None:
    config = _read_object(claude_config_path())
    projects = _object_field(config, "projects")
    entry = _object_field(projects, str(project))
    server = _object_field(entry, "mcpServers").get("memflow")
    if server is not None and not isinstance(server, dict):
        raise ValueError("Claude MCP server memflow must be a JSON object")
    return server


@dataclass(frozen=True)
class MCPSettingsPlan:
    project_root: Path
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    action: str

    @property
    def changed(self) -> bool:
        return self.before != self.after

    def to_status(self) -> dict[str, Any]:
        return {
            "requested": self.action,
            "scope": "local",
            "installed_before": self.before is not None,
            "installed_after": self.after is not None,
            "changed": self.changed,
        }


def build_mcp_settings_plan(
    project: Path,
    *,
    action: str,
    server: dict[str, Any],
    managed_server: dict[str, Any] | None,
) -> MCPSettingsPlan:
    if action not in {"on", "off"}:
        raise ValueError(f"unsupported MCP action: {action}")
    before = local_mcp_server(project)
    after = server if action == "on" else None
    if before is not None and before != after and before != managed_server:
        raise ValueError(
            "a different local MCP server named memflow already exists; "
            "inspect it with 'claude mcp get memflow' before replacing it"
        )
    if before is None and action == "on":
        global_config = _read_object(claude_config_path())
        for config in (_read_object(project / ".mcp.json"), global_config):
            existing = _object_field(config, "mcpServers").get("memflow")
            if existing is not None and existing != after:
                raise ValueError(
                    "a project or user MCP server named memflow already exists; "
                    "inspect it with 'claude mcp get memflow' before replacing it"
                )
    return MCPSettingsPlan(project, before, after, action)


def _run_claude(claude: str, project: Path, args: list[str]) -> None:
    env = os.environ.copy()
    if env.get("CLAUDE_CONFIG_DIR"):
        env["CLAUDE_CONFIG_DIR"] = str(claude_config_path().parent)
    try:
        result = subprocess.run(
            [claude, "mcp", *args],
            cwd=project,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Claude MCP setup timed out after 30 seconds") from exc
    except OSError as exc:
        raise RuntimeError(f"could not run Claude MCP setup: {exc.strerror}") from exc
    if result.returncode:
        raise RuntimeError(
            f"Claude MCP {args[0]} failed (exit {result.returncode}); "
            "check 'claude mcp get memflow' "
            "in the target project and rerun configure"
        )


def _add_server(claude: str, project: Path, server: dict[str, Any]) -> None:
    _run_claude(
        claude,
        project,
        ["add-json", "--scope", "local", "memflow", json.dumps(server)],
    )


def apply_mcp_settings_plan(plan: MCPSettingsPlan) -> None:
    if local_mcp_server(plan.project_root) != plan.before:
        raise RuntimeError("MCP configuration changed during setup; rerun configure")
    if not plan.changed:
        return
    claude = shutil.which("claude")
    if claude is None:
        raise ValueError("claude was not found on PATH; install Claude Code first")
    if plan.before is not None:
        _run_claude(
            claude, plan.project_root, ["remove", "--scope", "local", "memflow"]
        )
    try:
        if plan.after is not None:
            _add_server(claude, plan.project_root, plan.after)
    except RuntimeError:
        # Restore a managed registration if replacing it failed before writing.
        if plan.before is not None and local_mcp_server(plan.project_root) is None:
            _add_server(claude, plan.project_root, plan.before)
        raise
    if local_mcp_server(plan.project_root) != plan.after:
        raise RuntimeError("Claude MCP configuration was not saved; rerun configure")
