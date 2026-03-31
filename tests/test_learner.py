"""Unit tests for MemFlow learner."""

import pytest
from unittest.mock import MagicMock

from memflow.learner import Learner
from memflow.models import Step, StepResult, Procedure, StepType


class TestLearner:
    """Tests for Learner."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.generate.return_value = (
            '{"has_procedure": true, "title": "Learned Procedure", '
            '"category": "workflow", "content": "1. Step one\\n2. Step two"}'
        )
        return llm

    def test_extract_with_successful_steps(self, mock_llm):
        """Test extraction from successful step results."""
        learner = Learner(mock_llm)

        successful_step = Step(
            id="step-1",
            goal="Deploy",
            type=StepType.TOOL,
            tool_name="bash",
            args={"command": "./deploy.sh"}
        )
        successful_step.result = StepResult(
            step_id="step-1",
            success=True,
            output="Deployment successful"
        )

        procedure = learner.extract("Deploy app", [successful_step], user_id="user1")

        assert procedure is not None
        assert procedure.title == "Learned Procedure"
        assert procedure.category == "workflow"
        assert "Step one" in procedure.content
        assert procedure.user_id == "user1"

    def test_extract_with_no_successful_steps(self, mock_llm):
        """Test extraction when all steps failed."""
        learner = Learner(mock_llm)

        failed_step = Step(
            id="step-1",
            goal="Deploy",
            type=StepType.TOOL,
            tool_name="bash",
            args={}
        )
        failed_step.result = StepResult(
            step_id="step-1",
            success=False,
            output="",
            error="Command failed"
        )

        procedure = learner.extract("Deploy app", [failed_step])

        assert procedure is None
        mock_llm.generate.assert_not_called()

    def test_extract_with_empty_steps(self, mock_llm):
        """Test extraction with empty steps list."""
        learner = Learner(mock_llm)

        procedure = learner.extract("Deploy app", [])

        assert procedure is None
        mock_llm.generate.assert_not_called()

    def test_extract_with_multiple_successful_steps(self, mock_llm):
        """Test extraction from multiple successful steps."""
        learner = Learner(mock_llm)

        step1 = Step(id="step-1", goal="Install", type=StepType.TOOL, tool_name="bash")
        step1.result = StepResult(step_id="step-1", success=True, output="Installed")

        step2 = Step(id="step-2", goal="Configure", type=StepType.TOOL, tool_name="bash")
        step2.result = StepResult(step_id="step-2", success=True, output="Configured")

        step3 = Step(id="step-3", goal="Deploy", type=StepType.TOOL, tool_name="bash")
        step3.result = StepResult(step_id="step-3", success=True, output="Deployed")

        procedure = learner.extract("Full deploy", [step1, step2, step3])

        assert procedure is not None
        assert mock_llm.generate.called

    def test_extract_with_mixed_results(self, mock_llm):
        """Test extraction ignores failed steps."""
        learner = Learner(mock_llm)

        success_step = Step(id="step-1", goal="Success step", type=StepType.TOOL, tool_name="bash")
        success_step.result = StepResult(step_id="step-1", success=True, output="Success")

        fail_step = Step(id="step-2", goal="Fail step", type=StepType.TOOL, tool_name="bash")
        fail_step.result = StepResult(step_id="step-2", success=False, output="", error="Failed")

        procedure = learner.extract("Mixed task", [success_step, fail_step])

        assert procedure is not None
        # Verify only successful step was included in the prompt
        messages = mock_llm.generate.call_args[0][0]
        system_prompt = messages[0]["content"]
        assert "Success step" in system_prompt
        assert "Fail step" not in system_prompt

    def test_extract_with_invalid_llm_response(self, mock_llm):
        """Test extraction when LLM returns invalid JSON."""
        mock_llm.generate.return_value = "not valid json"
        learner = Learner(mock_llm)

        step = Step(id="step-1", goal="Test", type=StepType.TOOL, tool_name="bash")
        step.result = StepResult(step_id="step-1", success=True, output="Output")

        procedure = learner.extract("Test task", [step])

        assert procedure is None

    def test_extract_with_has_procedure_false(self, mock_llm):
        """Test extraction when LLM says no procedure found."""
        mock_llm.generate.return_value = '{"has_procedure": false}'
        learner = Learner(mock_llm)

        step = Step(id="step-1", goal="Test", type=StepType.TOOL, tool_name="bash")
        step.result = StepResult(step_id="step-1", success=True, output="Output")

        procedure = learner.extract("Test task", [step])

        assert procedure is None

    def test_extract_default_user_id(self, mock_llm):
        """Test extraction uses default user_id when not specified."""
        learner = Learner(mock_llm)

        step = Step(id="step-1", goal="Test", type=StepType.TOOL, tool_name="bash")
        step.result = StepResult(step_id="step-1", success=True, output="Output")

        procedure = learner.extract("Test task", [step])

        assert procedure.user_id == "default"

    def test_extract_preserves_task_in_title(self, mock_llm):
        """Test that task is used as title fallback."""
        mock_llm.generate.return_value = (
            '{"has_procedure": true, "title": "My Procedure", '
            '"category": "other", "content": "1. Step"}'
        )
        learner = Learner(mock_llm)

        step = Step(id="step-1", goal="Test", type=StepType.TOOL, tool_name="bash")
        step.result = StepResult(step_id="step-1", success=True, output="Output")

        procedure = learner.extract("Specific deployment task", [step])

        assert procedure.title == "My Procedure"

    def test_extract_calls_llm_with_correct_prompt(self, mock_llm):
        """Test that extraction calls LLM with correct prompt structure."""
        learner = Learner(mock_llm)

        step = Step(
            id="step-1",
            goal="Deploy step",
            type=StepType.TOOL,
            tool_name="bash",
            args={}
        )
        step.result = StepResult(
            step_id="step-1",
            success=True,
            output="Deployed successfully"
        )

        learner.extract("Deploy task", [step])

        messages = mock_llm.generate.call_args[0][0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Deploy task" in messages[0]["content"]
        assert "Deploy step" in messages[0]["content"]
