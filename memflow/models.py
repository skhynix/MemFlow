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


# ---------------------------------------------------------------------------
# Phase 3 — Plan / Execute / Learn
# ---------------------------------------------------------------------------

class StepType:
    """Step type: PLAN (sub-plan recursion) or TOOL (external call).

    Note: PLAN type is reserved for future hierarchical decomposition feature.
    Currently only TOOL type is implemented and used in Phase 3.
    """
    PLAN = "plan"  # TODO: Not yet implemented - reserved for future extension
    TOOL = "tool"


@dataclass
class StepResult:
    """Result of executing a single Step."""
    step_id: str
    success: bool
    output: str = ""
    error: str = ""
    retryable: bool = True  # Whether this step can be retried


@dataclass
class Step:
    """A single step in a TaskPlan with execution status tracking."""
    id: str
    goal: str
    type: str  # StepType.PLAN or StepType.TOOL
    tool_name: str | None = None
    status: str = "pending"  # "pending", "done", "failed"
    result: StepResult | None = None
    args: dict = field(default_factory=dict)


@dataclass
class TaskPlan:
    """A decomposed task ready for execution."""
    task: str
    steps: list[Step]
    context: str = ""  # procedure content retrieved and used during planning


@dataclass
class RunResult:
    """Aggregate result of executing a full TaskPlan."""
    plan: TaskPlan
    step_results: list[StepResult]
    learned: Procedure | None = None  # procedure extracted and stored after execution
