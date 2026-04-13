"""
Learner for MemFlow Phase 3.

After run() completes, Learner analyses the successful executed steps and asks
the LLM to extract a reusable Procedure (SOP).  If a procedure is extracted
it is returned to the caller for storage (Learn → Retrieve back-edge).
"""

from __future__ import annotations

from memflow.llm import BaseLLM, parse_json
from memflow.models import Procedure, Step
from memflow.prompts import LEARNING_PROMPT


class Learner:
    """Extracts a reusable Procedure from successful execution results."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def extract(
        self,
        task: str,
        steps: list[Step],
        user_id: str = "default",
    ) -> Procedure | None:
        """Return a Procedure extracted from successful steps, or None.

        Only successful executed steps are included in the learning prompt so the
        LLM focuses on what actually worked.
        """
        successful = [step for step in steps if step.result and step.result.success]
        if not successful:
            return None

        steps_text = "\n".join(
            f"{i + 1}. [{step.tool_name or 'llm'}] {step.goal}\n"
            f"   Output: {step.result.output[:400]}"
            for i, step in enumerate(successful)
        )

        messages = [
            {
                "role": "system",
                "content": LEARNING_PROMPT.format(task=task, steps=steps_text),
            },
            {"role": "user", "content": "Respond with JSON only."},
        ]

        try:
            response = self.llm.generate(messages)
            data = parse_json(response)
        except Exception:
            return None

        if not data.get("has_procedure"):
            return None

        return Procedure(
            title=data.get("title", task),
            content=data.get("content", ""),
            user_id=user_id,
            category=data.get("category", "workflow"),
        )
