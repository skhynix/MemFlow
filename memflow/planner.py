"""
Task planner for MemFlow Phase 3.

LLMPlanner decomposes a high-level task into a list of executable Jobs,
optionally using retrieved procedures as planning context
(Retrieve → Planner back-edge).
"""

from __future__ import annotations

from memflow.llm import BaseLLM, parse_json
from memflow.models import Job, TaskPlan
from memflow.prompts import PLANNING_PROMPT

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
    """

    def __init__(
        self,
        llm: BaseLLM,
        extra_tools: list[dict] | None = None,
    ) -> None:
        self.llm = llm
        tools = list(DEFAULT_TOOLS)
        if extra_tools:
            tools.extend(extra_tools)
        self._tools_text = "\n".join(
            f"- {t['name']}: {t['description']}" for t in tools
        )

    def plan(self, task: str, context: str = "") -> TaskPlan:
        """Decompose task into a TaskPlan.

        Args:
            task:    High-level task description.
            context: Relevant procedure text retrieved from the store.
        """
        messages = [
            {
                "role": "system",
                "content": PLANNING_PROMPT.format(
                    procedures=context or "No relevant procedures found.",
                    tools=self._tools_text,
                    task=task,
                ),
            },
            {"role": "user", "content": "Respond with JSON only."},
        ]
        response = self.llm.generate(messages)
        data = parse_json(response)
        jobs = [Job.from_dict(s) for s in data.get("steps", [])]
        return TaskPlan(task=task, jobs=jobs, context=context)
