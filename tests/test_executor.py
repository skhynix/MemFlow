# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MemFlow executor."""

import pytest
from unittest.mock import MagicMock, patch

from memflow.executor import ToolRegistry, _bash_tool, _http_tool, _make_llm_tool
from memflow.models import Step, StepResult, StepType


class TestToolRegistry:
    """Tests for ToolRegistry."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.generate.return_value = "LLM response"
        return llm

    def test_builtin_tools_registered(self, mock_llm):
        """Test that built-in tools are registered."""
        registry = ToolRegistry(llm=mock_llm)

        tools = registry.available_tools()
        assert "llm" in tools
        assert "bash" in tools
        assert "http" in tools

    def test_bash_tool_without_llm(self):
        """Test ToolRegistry works without LLM (bash only)."""
        registry = ToolRegistry()

        tools = registry.available_tools()
        assert "llm" not in tools
        assert "bash" in tools
        assert "http" in tools

    def test_register_custom_tool(self, mock_llm):
        """Test registering a custom tool."""
        registry = ToolRegistry(llm=mock_llm)

        def my_custom_tool(arg1: str) -> str:
            return f"Result: {arg1}"

        registry.register("my_tool", my_custom_tool)

        assert "my_tool" in registry.available_tools()

    def test_execute_custom_tool(self, mock_llm):
        """Test executing a custom registered tool."""
        registry = ToolRegistry(llm=mock_llm)

        def echo_tool(message: str) -> str:
            return f"Echo: {message}"

        registry.register("echo", echo_tool)

        step = Step(
            id="step-1",
            goal="Test echo",
            type=StepType.TOOL,
            tool_name="echo",
            args={"message": "Hello"}
        )
        result = registry.execute_step(step)

        assert result.success is True
        assert result.output == "Echo: Hello"

    def test_execute_unknown_tool(self, mock_llm):
        """Test executing with unknown tool name."""
        registry = ToolRegistry(llm=mock_llm)

        step = Step(
            id="step-1",
            goal="Test",
            type=StepType.TOOL,
            tool_name="unknown_tool",
            args={}
        )
        result = registry.execute_step(step)

        assert result.success is False
        assert "Unknown tool" in result.error
        assert "unknown_tool" in result.error

    def test_execute_tool_with_error(self, mock_llm):
        """Test tool execution that raises an exception."""
        registry = ToolRegistry(llm=mock_llm)

        def failing_tool() -> str:
            raise ValueError("Tool failed!")

        registry.register("failing", failing_tool)

        step = Step(
            id="step-1",
            goal="Test",
            type=StepType.TOOL,
            tool_name="failing",
            args={}
        )
        result = registry.execute_step(step)

        assert result.success is False
        assert "Tool failed!" in result.error

    def test_register_overwrites_existing_tool(self, mock_llm):
        """Test that registering a tool with same name overwrites it."""
        registry = ToolRegistry(llm=mock_llm)

        def original_tool() -> str:
            return "Original"

        def replaced_tool() -> str:
            return "Replaced"

        registry.register("my_tool", original_tool)
        registry.register("my_tool", replaced_tool)

        step = Step(
            id="step-1",
            goal="Test",
            type=StepType.TOOL,
            tool_name="my_tool",
            args={}
        )
        result = registry.execute_step(step)

        assert result.output == "Replaced"

    def test_execute_llm_tool(self, mock_llm):
        """Test executing the built-in LLM tool."""
        mock_llm.generate.return_value = "Generated response"
        registry = ToolRegistry(llm=mock_llm)

        step = Step(
            id="step-1",
            goal="Ask question",
            type=StepType.TOOL,
            tool_name="llm",
            args={"prompt": "What is X?"}
        )
        result = registry.execute_step(step)

        assert result.success is True
        assert result.output == "Generated response"
        mock_llm.generate.assert_called_once()

    def test_execute_bash_tool(self, mock_llm):
        """Test executing the built-in bash tool."""
        registry = ToolRegistry(llm=mock_llm)

        step = Step(
            id="step-1",
            goal="Echo test",
            type=StepType.TOOL,
            tool_name="bash",
            args={"command": "echo hello"}
        )
        result = registry.execute_step(step)

        assert result.success is True
        assert "hello" in result.output

    def test_execute_http_tool(self, mock_urlopen_context, mock_llm):
        """Test executing the built-in HTTP tool."""
        registry = ToolRegistry(llm=mock_llm)

        step = Step(
            id="step-1",
            goal="GET request",
            type=StepType.TOOL,
            tool_name="http",
            args={"url": "https://example.com/api", "method": "GET"}
        )
        result = registry.execute_step(step)

        assert result.success is True
        assert result.output == '{"status": "ok"}'

    def test_execute_plan_step_returns_failure(self, mock_llm):
        """Test that PLAN type steps return failure (not yet implemented)."""
        registry = ToolRegistry(llm=mock_llm)

        step = Step(
            id="step-1",
            goal="Sub-plan",
            type=StepType.PLAN,
            tool_name=None,
            args={}
        )
        result = registry.execute_step(step)

        assert result.success is False
        assert "PLAN type steps are not yet implemented" in result.error
        assert result.retryable is False


class TestBashTool:
    """Tests for _bash_tool function."""

    def test_bash_echo(self):
        """Test bash tool with echo command."""
        result = _bash_tool("echo test_output")
        assert "test_output" in result

    def test_bash_command_failure(self):
        """Test bash tool with failing command returns empty output."""
        result = _bash_tool("exit 1")
        # Non-zero exit returns empty stdout/stderr, not an exception
        assert result == ""

    def test_bash_stderr_capture(self):
        """Test bash tool captures stderr."""
        result = _bash_tool("echo stderr_msg >&2")
        assert "stderr_msg" in result

    def test_bash_nonzero_exit_with_stderr(self):
        """Test bash tool captures stderr on non-zero exit."""
        result = _bash_tool("echo error_msg >&2; exit 1")
        assert "error_msg" in result


class TestHttpTool:
    """Tests for _http_tool function."""

    def test_http_tool_get(self, mock_urlopen_context):
        """Test HTTP tool with GET request."""
        result = _http_tool("https://example.com/api", method="GET")

        assert result == '{"status": "ok"}'

    def test_http_tool_post(self, mock_urlopen_context, mock_http_response):
        """Test HTTP tool with POST request."""
        mock_response = mock_http_response(b'{"status": "created"}')
        mock_urlopen_context.return_value.__enter__.return_value = mock_response

        result = _http_tool(
            "https://example.com/api",
            method="POST",
            body='{"key": "value"}'
        )

        assert result == '{"status": "created"}'

    def test_http_tool_default_method_is_get(self, mock_urlopen_context, mock_http_response):
        """Test HTTP tool defaults to GET method."""
        mock_response = mock_http_response(b"OK")
        mock_urlopen_context.return_value.__enter__.return_value = mock_response

        _http_tool("https://example.com")

        # Verify method was GET
        call_args = mock_urlopen_context.call_args[0][0]
        assert call_args.method == "GET"


class TestMakeLlmTool:
    """Tests for _make_llm_tool function."""

    def test_llm_tool_creation(self):
        """Test creating an LLM tool."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Response text"

        llm_tool = _make_llm_tool(mock_llm)
        result = llm_tool(prompt="Test prompt")

        assert result == "Response text"
        mock_llm.generate.assert_called_once_with([{"role": "user", "content": "Test prompt"}])
