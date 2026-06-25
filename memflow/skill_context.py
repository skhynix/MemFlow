# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Agent-agnostic request and response boundaries for skill context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SkillContextRequest:
    prompt: str
    cwd: str
    agent: str
    adapter: str
    session_id: str
    transcript_path: str
    user_id: str
    project_scope: str


@dataclass(frozen=True)
class SkillContextResponse:
    trace_id: str
    selected_skills: tuple[dict[str, Any], ...]
    rendered_context: str
    warnings: tuple[str, ...]
    status: str
    latency_ms: int


__all__ = ["SkillContextRequest", "SkillContextResponse"]
