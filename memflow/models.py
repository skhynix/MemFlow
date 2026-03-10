"""
Data models for MemFlow.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


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


# ---------------------------------------------------------------------------
# Phase 3 — Plan / Execute / Learn
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """A single executable step in a TaskPlan."""
    tool: str         # "llm" | "bash" | "http" | custom registered name
    description: str  # human-readable purpose of this step
    args: dict        # tool-specific keyword arguments
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        return cls(
            tool=data.get("tool", "llm"),
            description=data.get("description", ""),
            args=data.get("args", {}),
        )


@dataclass
class TaskPlan:
    """A decomposed task ready for execution."""
    task: str
    jobs: list[Job]
    context: str = ""  # procedure content retrieved and used during planning


@dataclass
class JobResult:
    """Result of executing a single Job."""
    job: Job
    success: bool
    output: str
    error: str = ""


@dataclass
class RunResult:
    """Aggregate result of executing a full TaskPlan."""
    plan: TaskPlan
    job_results: list[JobResult]
    learned: Procedure | None = None  # procedure extracted and stored after execution
