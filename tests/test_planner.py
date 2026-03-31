"""Unit tests for MemFlow planner."""

import pytest
from unittest.mock import MagicMock

from memflow.planner import LLMPlanner, DEFAULT_TOOLS
from memflow.models import Step, TaskPlan, StepType


class TestLLMPlanner:
    """Tests for LLMPlanner."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.generate.return_value = '{"steps": [{"tool": "bash", "description": "Run deploy", "args": {"command": "./deploy.sh"}}]}'
        return llm

    def test_plan_with_context(self, mock_llm):
        """Test planning with procedure context."""
        planner = LLMPlanner(mock_llm)
        context = "### How to deploy\n1. Run deploy.sh"

        plan = planner.plan("Deploy the application", context=context)

        assert isinstance(plan, TaskPlan)
        assert plan.task == "Deploy the application"
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "bash"
        assert plan.context == context

    def test_plan_without_context(self, mock_llm):
        """Test planning without context."""
        planner = LLMPlanner(mock_llm)

        plan = planner.plan("Deploy the application")

        assert plan.context == ""
        assert len(plan.steps) == 1

    def test_plan_with_multiple_steps(self, mock_llm):
        """Test planning that produces multiple steps."""
        mock_llm.generate.return_value = (
            '{"steps": ['
            '{"tool": "bash", "description": "Install deps", "args": {"command": "pip install -r requirements.txt"}},'
            '{"tool": "bash", "description": "Run tests", "args": {"command": "pytest"}}'
            "]}"
        )
        planner = LLMPlanner(mock_llm)

        plan = planner.plan("Setup and test")

        assert len(plan.steps) == 2
        assert plan.steps[0].tool_name == "bash"
        assert plan.steps[1].tool_name == "bash"

    def test_plan_with_empty_steps(self, mock_llm):
        """Test planning with empty steps array."""
        mock_llm.generate.return_value = '{"steps": []}'
        planner = LLMPlanner(mock_llm)

        plan = planner.plan("Some task")

        assert len(plan.steps) == 0

    def test_plan_with_extra_tools(self, mock_llm):
        """Test planner with custom extra tools."""
        mock_llm.generate.return_value = '{"steps": []}'

        extra_tools = [
            {"name": "custom_tool", "description": "A custom tool"},
            {"name": "another_tool", "description": "Another tool"}
        ]
        planner = LLMPlanner(mock_llm, extra_tools=extra_tools)
        planner.plan("Test task")

        # Verify extra tools are included in the LLM prompt
        messages = mock_llm.generate.call_args[0][0]
        system_prompt = messages[0]["content"]
        assert "custom_tool" in system_prompt
        assert "another_tool" in system_prompt

    def test_default_tools_in_planner(self, mock_llm):
        """Test that default tools are included in the prompt."""
        mock_llm.generate.return_value = '{"steps": []}'

        planner = LLMPlanner(mock_llm)
        planner.plan("Test task")

        # Verify default tools are included in the LLM prompt
        messages = mock_llm.generate.call_args[0][0]
        system_prompt = messages[0]["content"]
        assert "llm" in system_prompt
        assert "bash" in system_prompt
        assert "http" in system_prompt

    def test_plan_calls_llm_generate(self, mock_llm):
        """Test that planning calls LLM generate."""
        planner = LLMPlanner(mock_llm)
        planner.plan("Test task")

        mock_llm.generate.assert_called_once()
        messages = mock_llm.generate.call_args[0][0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Respond with JSON only" in messages[1]["content"]

    def test_plan_passes_task_to_llm(self, mock_llm):
        """Test that task is included in LLM prompt."""
        planner = LLMPlanner(mock_llm)
        planner.plan("My specific task")

        messages = mock_llm.generate.call_args[0][0]
        system_prompt = messages[0]["content"]
        assert "My specific task" in system_prompt

    def test_plan_step_has_correct_type(self, mock_llm):
        """Test that planned steps have TOOL type."""
        planner = LLMPlanner(mock_llm)
        plan = planner.plan("Test task")

        if plan.steps:
            assert plan.steps[0].type == StepType.TOOL

    def test_plan_step_has_uuid(self, mock_llm):
        """Test that planned steps have UUID."""
        planner = LLMPlanner(mock_llm)
        plan = planner.plan("Test task")

        if plan.steps:
            assert plan.steps[0].id is not None
            assert len(plan.steps[0].id) > 0
