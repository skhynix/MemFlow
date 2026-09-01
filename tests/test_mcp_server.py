# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the local MemFlow MCP skill server."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import memflow.mcp_server as mcp_server_module
from memflow.mcp_server import read_skill_from_store, server
from memflow.models import Procedure
from memflow.store import EmulatedStore


def _procedure(
    *,
    skill_id: str = "skill:commit-craft",
    kind: str = "skill",
    trust_state: str = "trusted",
    trust_mode: str = "instruction",
) -> Procedure:
    return Procedure(
        id=skill_id,
        title="fallback-name",
        kind=kind,
        content="---\nname: commit-craft\n---\n\n# Complete skill\n",
        source_path="/private/source/commit-craft/SKILL.md",
        metadata={
            "skill": {
                "name": "commit-craft",
                "root_path": "/private/source/commit-craft",
                "source_path": "/private/source/commit-craft/SKILL.md",
            },
            "governance": {
                "trust_state": trust_state,
                "mode": trust_mode,
            },
        },
    )


def test_read_skill_returns_complete_content_and_minimal_metadata():
    store = EmulatedStore()
    procedure = _procedure()
    store.add(procedure)

    result = read_skill_from_store(store, procedure.id)

    assert result == {
        "id": procedure.id,
        "name": "commit-craft",
        "trust_mode": "instruction",
        "content": procedure.content,
    }
    assert "/private/source" not in repr(result)


def test_read_skill_uses_exact_id_lookup_only():
    store = EmulatedStore()
    store.add(_procedure())

    with pytest.raises(ValueError, match="skill not found"):
        read_skill_from_store(store, "commit-craft")


def test_read_skill_rejects_non_skill_records():
    store = EmulatedStore()
    store.add(_procedure(skill_id="procedure:one", kind="procedure"))

    with pytest.raises(ValueError, match="record is not a skill"):
        read_skill_from_store(store, "procedure:one")


@pytest.mark.parametrize(
    ("trust_state", "trust_mode"),
    [("blocked", "data"), ("unknown", "blocked")],
)
def test_read_skill_rejects_blocked_skills(trust_state, trust_mode):
    store = EmulatedStore()
    store.add(_procedure(trust_state=trust_state, trust_mode=trust_mode))

    with pytest.raises(PermissionError, match="skill is blocked"):
        read_skill_from_store(store, "skill:commit-craft")


def test_read_skill_defaults_unrecognized_trust_modes_to_data():
    store = EmulatedStore()
    store.add(_procedure(trust_state="unknown", trust_mode="unexpected"))

    result = read_skill_from_store(store, "skill:commit-craft")

    assert result["trust_mode"] == "data"


def test_server_exposes_one_read_only_tool():
    tools = asyncio.run(server.list_tools())

    assert [tool.name for tool in tools] == ["read_skill"]
    assert tools[0].annotations is not None
    assert tools[0].annotations.readOnlyHint is True
    assert tools[0].annotations.destructiveHint is False
    assert tools[0].annotations.idempotentHint is True
    assert tools[0].annotations.openWorldHint is False


def test_registered_tool_returns_text_and_structured_skill(monkeypatch):
    store = EmulatedStore()
    procedure = _procedure()
    store.add(procedure)
    monkeypatch.setattr(
        mcp_server_module,
        "_manager",
        SimpleNamespace(store=store),
    )

    content, structured = asyncio.run(
        server.call_tool("read_skill", {"skill_id": procedure.id})
    )

    expected = {
        "id": procedure.id,
        "name": "commit-craft",
        "trust_mode": "instruction",
        "content": procedure.content,
    }
    assert structured == expected
    assert len(content) == 1
    assert json.loads(content[0].text) == expected


def test_tool_hides_store_initialization_details(monkeypatch):
    def fail_to_create_manager(_env_file):
        raise RuntimeError("connection failed at secret-host:6333")

    monkeypatch.setattr(mcp_server_module, "_manager", None)
    monkeypatch.setattr(mcp_server_module, "_create_manager", fail_to_create_manager)

    with pytest.raises(RuntimeError, match="^MemFlow skill store is unavailable$"):
        mcp_server_module.read_skill("skill:commit-craft")
