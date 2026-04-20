# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tool executor for MemFlow Phase 3.

ToolRegistry holds callable tools and executes steps one at a time,
returning a StepResult for each.

Built-in tools
--------------
  llm   — delegates to the LLM (requires llm= at init time)
  bash  — runs a shell command via subprocess (30 s timeout)
  http  — makes an HTTP request via urllib (no extra dependencies)

Custom tools can be registered with ToolRegistry.register(name, fn).
Each tool function receives the Step.args dict as keyword arguments and
must return a string.
"""

from __future__ import annotations

import subprocess
import urllib.request
from typing import Callable

from memflow.llm import BaseLLM
from memflow.models import Step, StepResult


class ToolRegistry:
    """Registry of named tools that execute steps."""

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self._tools: dict[str, Callable[..., str]] = {}
        if llm is not None:
            self.register("llm", _make_llm_tool(llm))
        self.register("bash", _bash_tool)
        self.register("http", _http_tool)

    def register(self, name: str, fn: Callable[..., str]) -> None:
        """Register a custom tool.  Overwrites any existing tool with the same name."""
        self._tools[name] = fn

    def available_tools(self) -> list[str]:
        return list(self._tools)

    def execute_step(self, step: Step) -> StepResult:
        """Execute a single Step and return a StepResult.

        Note: PLAN type steps are not yet implemented. Currently only TOOL type
        steps are supported. PLAN type support is planned for future extension
        to enable hierarchical task decomposition.
        """
        if step.type == "plan":
            # TODO: Implement PLAN type step handling for hierarchical decomposition.
            # Currently returns failure as PLAN steps should be expanded by the
            # planner before execution. See design doc for future implementation.
            return StepResult(
                step_id=step.id,
                success=False,
                error="PLAN type steps are not yet implemented (TODO: expand to sub-steps)",
                retryable=False,
            )

        fn = self._tools.get(step.tool_name)
        if fn is None:
            return StepResult(
                step_id=step.id,
                success=False,
                output="",
                error=f"Unknown tool {step.tool_name!r}. Available: {self.available_tools()}",
                retryable=False,
            )
        try:
            output = fn(**step.args)
            return StepResult(step_id=step.id, success=True, output=str(output))
        except Exception as e:
            return StepResult(step_id=step.id, success=False, output="", error=str(e), retryable=True)


# ---------------------------------------------------------------------------
# Built-in tool implementations
# ---------------------------------------------------------------------------

def _make_llm_tool(llm: BaseLLM) -> Callable[..., str]:
    def _llm_tool(prompt: str) -> str:
        return llm.generate([{"role": "user", "content": prompt}])
    return _llm_tool


def _bash_tool(command: str) -> str:
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout
    if result.stderr:
        output += ("\n" if output else "") + result.stderr
    return output.strip()


def _http_tool(url: str, method: str = "GET", body: str = "") -> str:
    req = urllib.request.Request(url, method=method.upper())
    req.add_header("User-Agent", "MemFlow/1.0")
    if body:
        req.data = body.encode()
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read().decode(errors="replace")
