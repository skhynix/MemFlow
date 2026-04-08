"""
Task planner for MemFlow Phase 3.

LLMPlanner decomposes a high-level task into a list of executable Jobs,
optionally using retrieved procedures as planning context
(Retrieve → Planner back-edge).

Supports two planning modes:
  1. Single-shot planning (default) — decompose entire task at once.
  2. Multi-stage planning — plan a few steps, execute, reflect, then replan.
"""

from __future__ import annotations

import uuid

from memflow.llm import BaseLLM, parse_json
from memflow.models import Job, JobResult, TaskPlan, Step, StepType, StepResult
from memflow.prompts import PLANNING_PROMPT, REPLAN_PROMPT

# Default tool descriptions shown to the planner LLM.
DEFAULT_TOOLS = [
    {
        "name": "llm",
        "description": (
            'Ask the language model. Args: {"prompt": "your question"}'
        ),
    },
    {
        "name": "bash",
        "description": (
            'Run a shell command. Args: {"command": "echo hello"}'
        ),
    },
    {
        "name": "http",
        "description": (
            'Make an HTTP request. '
            'Args: {"url": "https://...", "method": "GET", "body": ""}'
        ),
    },
]


class LLMPlanner:
    """
    Decomposes tasks into executable job plans using an LLM.

    The planner receives relevant procedures as context so it can reuse
    existing SOPs rather than reinventing steps from scratch.

    Supports multi-stage planning with reflection:
      1. Plan initial steps (max_steps per iteration).
      2. Execute and collect results.
      3. Reflect on outcomes — identify failures or new information.
      4. Replan remaining work with updated context.
      5. Repeat until task is complete or max iterations reached.
    """

    def __init__(
        self,
        llm: BaseLLM,
        extra_tools: list[dict] | None = None,
        max_steps_per_iteration: int = 3,
        max_iterations: int = 5,
    ) -> None:
        self.llm = llm
        tools = list(DEFAULT_TOOLS)
        if extra_tools:
            tools.extend(extra_tools)
        self._tools_text = "\n".join(
            f"- {t['name']}: {t['description']}" for t in tools
        )
        self._max_steps_per_iteration = max_steps_per_iteration
        self._max_iterations = max_iterations
        self._plan_guard: "PlanGuard | None" = None

    def plan(
        self,
        task: str,
        context: str = "",
        multi_stage: bool = False,
        executed_results: list[JobResult] | None = None,
        plan_guard: "PlanGuard | None" = None,
    ) -> TaskPlan:
        """Decompose task into a TaskPlan.

        Args:
            task:             High-level task description.
            context:          Relevant procedure text retrieved from the store.
            multi_stage:      If True, plan only a few steps for iterative execution.
            executed_results: Results from previously executed jobs (for replanning).
            plan_guard:       Guard for controlling recursion depth.

        Returns:
            TaskPlan with jobs to execute.
        """
        # Set plan guard
        self._plan_guard = plan_guard

        if multi_stage and executed_results:
            # Replanning with execution history
            return self._replan(task, context, executed_results)
        elif multi_stage:
            # Initial multi-stage planning (limited steps)
            return self._plan_limited(task, context)
        else:
            # Single-shot planning (original behavior)
            return self._plan_single(task, context)

    def _plan_limited(self, task: str, context: str = "") -> TaskPlan:
        """Plan only a limited number of steps (for multi-stage planning)."""
        # Check PlanGuard for recursion depth
        if self._plan_guard and not self._plan_guard.can_recurse():
            return TaskPlan(task=task, steps=[], context=context)

        # Enter recursion depth
        if self._plan_guard:
            self._plan_guard.enter()

        try:
            messages = [
                {
                    "role": "system",
                    "content": PLANNING_PROMPT.format(
                        procedures=context or "No relevant procedures found.",
                        tools=self._tools_text,
                        task=task,
                        max_steps=self._max_steps_per_iteration,
                    ),
                },
                {"role": "user", "content": "Respond with JSON only."},
            ]
            response = self.llm.generate(messages)
            data = parse_json(response)

            # Handle case where LLM returns a list directly instead of {"steps": [...]}
            if isinstance(data, list):
                step_dicts = data
            elif isinstance(data, dict):
                step_dicts = data.get("steps", [])
            else:
                step_dicts = []

            # Convert to Step objects
            steps = []
            for s in step_dicts:
                tool = s.get("tool", "llm")
                steps.append(Step(
                    id=str(uuid.uuid4()),
                    goal=s.get("description", ""),
                    type=StepType.TOOL,
                    tool_name=tool,
                    args=s.get("args", {}),
                ))
            return TaskPlan(task=task, steps=steps, context=context)
        finally:
            # Exit recursion depth
            if self._plan_guard:
                self._plan_guard.exit()

    def _plan_single(self, task: str, context: str = "") -> TaskPlan:
        """Original single-shot planning - plan all steps at once."""
        # Check PlanGuard for recursion depth
        if self._plan_guard and not self._plan_guard.can_recurse():
            return TaskPlan(task=task, steps=[], context=context)

        # Enter recursion depth
        if self._plan_guard:
            self._plan_guard.enter()

        try:
            messages = [
                {
                    "role": "system",
                    "content": PLANNING_PROMPT.format(
                        procedures=context or "No relevant procedures found.",
                        tools=self._tools_text,
                        task=task,
                        max_steps=10,  # Plan all steps at once
                    ),
                },
                {"role": "user", "content": "Plan ALL steps needed to complete the task. Respond with JSON only."},
            ]
            response = self.llm.generate(messages)
            data = parse_json(response)

            # Handle case where LLM returns a list directly instead of {"steps": [...]}
            if isinstance(data, list):
                step_dicts = data
            elif isinstance(data, dict):
                step_dicts = data.get("steps", [])
            else:
                step_dicts = []

            # Convert to Step objects
            steps = []
            for s in step_dicts:
                steps.append(Step(
                    id=str(uuid.uuid4()),
                    goal=s.get("description", ""),
                    type=StepType.TOOL,
                    tool_name=s.get("tool", "llm"),
                    args=s.get("args", {}),
                ))
            return TaskPlan(task=task, steps=steps, context=context)
        finally:
            # Exit recursion depth
            if self._plan_guard:
                self._plan_guard.exit()

    def _replan(
        self,
        task: str,
        context: str,
        executed_results: list[JobResult],
    ) -> TaskPlan:
        """Replan based on execution results (Reflect → Replan)."""
        # Check PlanGuard for recursion depth
        if self._plan_guard and not self._plan_guard.can_recurse():
            return TaskPlan(task=task, steps=[], context=context)

        # Enter recursion depth
        if self._plan_guard:
            self._plan_guard.enter()

        try:
            # Summarize what has been done
            history = []
            for result in executed_results:
                status = "SUCCESS" if result.success else "FAILED"
                history.append(
                    f"[{status}] {result.job.description}\n"
                    f"  Output: {result.output or result.error}"
                )
            history_text = "\n\n".join(history)

            messages = [
                {
                    "role": "system",
                    "content": REPLAN_PROMPT.format(
                        procedures=context or "No relevant procedures found.",
                        tools=self._tools_text,
                        task=task,
                        history=history_text,
                        max_steps=self._max_steps_per_iteration,
                    ),
                },
                {"role": "user", "content": "Respond with JSON only."},
            ]
            response = self.llm.generate(messages)
            data = parse_json(response)

            # Handle case where LLM returns a list directly instead of {"steps": [...]}
            if isinstance(data, list):
                step_dicts = data
            elif isinstance(data, dict):
                step_dicts = data.get("steps", [])
            else:
                step_dicts = []

            # Convert to Step objects
            steps = []
            for s in step_dicts:
                steps.append(Step(
                    id=str(uuid.uuid4()),
                    goal=s.get("description", ""),
                    type=StepType.TOOL,
                    tool_name=s.get("tool", "llm"),
                    args=s.get("args", {}),
                ))
            return TaskPlan(task=task, steps=steps, context=context)
        finally:
            # Exit recursion depth
            if self._plan_guard:
                self._plan_guard.exit()
