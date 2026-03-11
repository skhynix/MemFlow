"""
Data models for MemFlow.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Procedure:
    """A stored procedural memory entry."""
    title: str
    content: str  # Markdown text with numbered steps
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default"
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SearchResult:
    """A procedure retrieved from search with its relevance score."""
    procedure: Procedure
    score: float
