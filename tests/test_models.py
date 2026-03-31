"""Unit tests for MemFlow data models."""

import pytest
from memflow.models import Procedure, Step, StepResult, TaskPlan, RunResult, SearchResult, StepType


class TestProcedure:
    """Tests for Procedure model."""

    def test_procedure_minimal(self):
        """Test creating a procedure with minimum required fields."""
        proc = Procedure(title="Test Title", content="1. Step one")
        assert proc.title == "Test Title"
        assert proc.content == "1. Step one"
        assert proc.id is not None
        assert proc.user_id == "default"
        assert proc.category == "general"
        assert proc.tags == []
        assert proc.created_at is not None

    def test_procedure_full(self):
        """Test creating a procedure with all fields."""
        proc = Procedure(
            title="Full Procedure",
            content="1. First\n2. Second",
            user_id="user123",
            category="workflow",
            tags=["tag1", "tag2"],
            created_at="2026-03-31T10:00:00"
        )
        assert proc.title == "Full Procedure"
        assert proc.content == "1. First\n2. Second"
        assert proc.user_id == "user123"
        assert proc.category == "workflow"
        assert proc.tags == ["tag1", "tag2"]
        assert proc.created_at == "2026-03-31T10:00:00"

    def test_procedure_unique_ids(self):
        """Test that each procedure gets a unique ID."""
        proc1 = Procedure(title="Test 1", content="1. Step")
        proc2 = Procedure(title="Test 2", content="1. Step")
        assert proc1.id != proc2.id


class TestStepResult:
    """Tests for StepResult model."""

    def test_step_result_success(self):
        """Test successful step result."""
        result = StepResult(step_id="step-123", success=True, output="Success output")
        assert result.success is True
        assert result.output == "Success output"
        assert result.error == ""
        assert result.retryable is True

    def test_step_result_failure(self):
        """Test failed step result."""
        result = StepResult(
            step_id="step-123",
            success=False,
            output="",
            error="Error message",
            retryable=False
        )
        assert result.success is False
        assert result.error == "Error message"
        assert result.retryable is False


class TestStep:
    """Tests for Step model."""

    def test_step_minimal(self):
        """Test creating a step with minimum fields."""
        step = Step(id="step-123", goal="Run command", type=StepType.TOOL, tool_name="bash")
        assert step.id == "step-123"
        assert step.goal == "Run command"
        assert step.type == StepType.TOOL
        assert step.tool_name == "bash"
        assert step.status == "pending"
        assert step.result is None
        assert step.args == {}

    def test_step_from_dict_parsing(self):
        """Test Step creation from dict (simulating LLM response parsing)."""
        data = {
            "tool": "bash",
            "description": "Run a command",
            "args": {"command": "echo hello"}
        }
        step = Step(
            id="step-123",
            goal=data.get("description", ""),
            type=StepType.TOOL,
            tool_name=data.get("tool", "llm"),
            args=data.get("args", {})
        )
        assert step.tool_name == "bash"
        assert step.goal == "Run a command"
        assert step.args == {"command": "echo hello"}

    def test_step_with_result(self):
        """Test step with attached result."""
        step = Step(id="step-123", goal="Test", type=StepType.TOOL, tool_name="llm")
        result = StepResult(step_id=step.id, success=True, output="Done")
        step.result = result
        step.status = "done"

        assert step.result.success is True
        assert step.status == "done"


class TestTaskPlan:
    """Tests for TaskPlan model."""

    def test_task_plan(self):
        """Test creating a task plan."""
        steps = [Step(id="step-1", goal="Step 1", type=StepType.TOOL, tool_name="bash")]
        plan = TaskPlan(task="Deploy app", steps=steps, context="Some context")
        assert plan.task == "Deploy app"
        assert len(plan.steps) == 1
        assert plan.context == "Some context"


class TestRunResult:
    """Tests for RunResult model."""

    def test_run_result(self):
        """Test creating a run result."""
        plan = TaskPlan(task="Test", steps=[])
        step_results = []
        learned = Procedure(title="Learned", content="1. Step")
        result = RunResult(plan=plan, step_results=step_results, learned=learned)
        assert result.plan == plan
        assert result.step_results == []
        assert result.learned == learned


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_search_result(self):
        """Test creating a search result."""
        proc = Procedure(title="Test", content="1. Step")
        result = SearchResult(procedure=proc, score=0.85)
        assert result.procedure == proc
        assert result.score == 0.85

    def test_search_result_score_zero(self):
        """Test search result with zero score."""
        proc = Procedure(title="Test", content="1. Step")
        result = SearchResult(procedure=proc, score=0.0)
        assert result.score == 0.0

    def test_search_result_score_one(self):
        """Test search result with perfect score."""
        proc = Procedure(title="Test", content="1. Step")
        result = SearchResult(procedure=proc, score=1.0)
        assert result.score == 1.0

    def test_search_result_different_procedures(self):
        """Test search results with different procedures have different scores."""
        proc1 = Procedure(title="Deploy guide", content="1. Run deploy.sh")
        proc2 = Procedure(title="Install guide", content="1. Run install.sh")
        result1 = SearchResult(procedure=proc1, score=0.9)
        result2 = SearchResult(procedure=proc2, score=0.7)
        assert result1.procedure.title != result2.procedure.title
        assert result1.score != result2.score
