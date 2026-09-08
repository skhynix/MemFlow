# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Local stdio MCP server for stored MemFlow skills."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

if TYPE_CHECKING:
    from memflow.manager import MemFlow
    from memflow.store import BaseStore

SERVER_NAME = "memflow"

server = FastMCP(
    SERVER_NAME,
    instructions="Read skills selected by the MemFlow Claude Code hook.",
)

_manager: MemFlow | None = None
_env_file: str | None = None


def _create_manager(env_file: str | None) -> MemFlow:
    from memflow.skill_cli import _create_skill_manager

    return _create_skill_manager(env_file)


def _get_manager() -> MemFlow:
    global _manager
    if _manager is None:
        _manager = _create_manager(_env_file)
    return _manager


def read_skill_from_store(store: BaseStore, skill_id: str) -> dict[str, str]:
    """Return one complete, non-blocked skill selected by its exact ID."""
    if not skill_id:
        raise ValueError("skill_id must not be empty")

    procedure = store.get(skill_id)
    if procedure is None or procedure.id != skill_id:
        raise ValueError(f"skill not found: {skill_id}")
    if procedure.kind != "skill":
        raise ValueError(f"record is not a skill: {skill_id}")

    skill = procedure.metadata.get("skill", {})
    if not isinstance(skill, dict):
        skill = {}
    governance = procedure.metadata.get("governance", {})
    if not isinstance(governance, dict):
        governance = {}

    trust_state = str(governance.get("trust_state") or "unknown")
    stored_mode = str(governance.get("mode") or "data")
    if trust_state == "blocked" or stored_mode == "blocked":
        raise PermissionError(f"skill is blocked: {skill_id}")

    return {
        "id": procedure.id,
        "name": str(skill.get("name") or procedure.title),
        "trust_mode": "instruction" if stored_mode == "instruction" else "data",
        "content": procedure.content,
    }


@server.tool(
    name="read_skill",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    structured_output=True,
)
def read_skill(skill_id: str) -> dict[str, str]:
    """Read the complete stored SKILL.md for an exact MemFlow skill ID."""
    try:
        store = _get_manager().store
    except Exception:
        raise RuntimeError("MemFlow skill store is unavailable") from None
    return read_skill_from_store(store, skill_id)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memflow-mcp",
        description="Run the local MemFlow MCP server over stdio.",
    )
    parser.add_argument(
        "--env-file",
        metavar="PATH",
        help="load Qdrant configuration from this environment file",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the MemFlow MCP server over stdio."""
    global _env_file
    args: Any = _build_parser().parse_args(argv)
    _env_file = args.env_file
    server.run(transport="stdio")


if __name__ == "__main__":
    main()


__all__ = ["main", "read_skill", "read_skill_from_store", "server"]
