# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Data models for MemFlow.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SkillRecord:
    """A stored skill snapshot and its retrieval metadata."""

    title: str
    content: str  # Stored Markdown snapshot
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default"
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    kind: str = "skill"
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


def _text_list(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _markdown_body_without_frontmatter(text: str) -> str:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return text
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "".join(lines[index + 1 :])
    return text


def skill_search_text(record: SkillRecord) -> str:
    """Return full-body-derived text used to index and rank a skill."""
    skill = record.metadata.get("skill", {})
    if not isinstance(skill, dict):
        skill = {}
    parts = [
        record.title,
        skill.get("name", ""),
        skill.get("description", ""),
        record.category,
        _text_list(record.tags),
        _text_list(skill.get("aliases", [])),
        _text_list(skill.get("file_patterns", [])),
        _text_list(skill.get("tools", [])),
        skill.get("relative_path", ""),
        _markdown_body_without_frontmatter(record.content),
    ]
    return "\n".join(str(part) for part in parts if part)


@dataclass
class SearchResult:
    """A skill retrieved from search with its relevance score."""

    skill: SkillRecord
    score: float
