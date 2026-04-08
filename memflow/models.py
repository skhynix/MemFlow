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

    @classmethod
    def from_job(cls, job: "Job") -> "Step":
        """Convert legacy Job to Step for backward compatibility."""
        return cls(
            id=job.id,
            goal=job.description,
            type=StepType.TOOL,
            tool_name=job.tool,
            args=job.args,
        )


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
    steps: list[Step]
    context: str = ""  # procedure content retrieved and used during planning

    # Backward compatibility
    @property
    def jobs(self) -> list[Job]:
        """Legacy property for backward compatibility."""
        return [Job(id=s.id, tool=s.tool_name or "llm", description=s.goal, args=s.args) for s in self.steps]


@dataclass
class JobResult:
    """Result of executing a single Job (legacy)."""
    job: Job
    success: bool
    output: str
    error: str = ""

    @classmethod
    def from_step_result(cls, step: Step, step_result: StepResult) -> "JobResult":
        """Convert StepResult to JobResult for backward compatibility."""
        job = Job(id=step.id, tool=step.tool_name or "llm", description=step.goal, args=step.args)
        return cls(
            job=job,
            success=step_result.success,
            output=step_result.output,
            error=step_result.error,
        )


@dataclass
class RunResult:
    """Aggregate result of executing a full TaskPlan."""
    plan: TaskPlan
    step_results: list[StepResult]
    learned: Procedure | None = None  # procedure extracted and stored after execution

    # Backward compatibility
    @property
    def job_results(self) -> list[JobResult]:
        """Legacy property for backward compatibility."""
        return [
            JobResult.from_step_result(self.plan.steps[i], r)
            for i, r in enumerate(self.step_results)
        ]
