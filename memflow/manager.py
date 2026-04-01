"""
MemFlowManager — core orchestrator for MemFlow.

Public API:
  add(messages, procedure, user_id)  — store a procedure
  search(query, user_id, top_k)      — retrieve procedures
  chat(query, user_id)               — respond using procedure context
  plan(task, user_id)                — decompose task into executable steps
  execute(plan, tools)               — execute a task plan
  run(task, user_id, tools)          — plan + execute + learn
"""

from __future__ import annotations

import threading
from typing import Callable

from memflow.executor import ToolRegistry
from memflow.learner import Learner
from memflow.llm import BaseLLM, LLMFactory, parse_json
from memflow.models import JobResult, Procedure, RunResult, SearchResult, TaskPlan
from memflow.planner import LLMPlanner
from memflow.prompts import CHAT_SYSTEM_PROMPT, CLASSIFICATION_PROMPT, EXTRACTION_PROMPT
from memflow.store import BaseStore, EmulatedStore, FileStore, MemMachineBypass, MemMachineStore

PROCEDURAL_KEYWORDS = [
    "step", "how to", "first", "then", "finally",
    "deploy", "install", "configure", "setup", "build",
    "run", "execute", "process", "workflow", "procedure",
]


class MemFlowManager:
    def __init__(
        self,
        llm: BaseLLM,
        store: BaseStore | None = None,
        bypass: MemMachineBypass | None = None,
    ) -> None:
        self.llm = llm
        self.store = store or EmulatedStore()
        self._bypass = bypass  # routes semantic/episodic content to MemMachine
        # Phase 3 components — lazily initialised on first use
        self._planner: LLMPlanner | None = None
        self._executor: ToolRegistry | None = None
        self._learner: Learner | None = None

    @classmethod
    def from_env(cls) -> "MemFlowManager":
        """Create a manager configured from environment variables.

        Environment variables:
          LLM_PROVIDER         — ollama | openai-compatible (default: ollama)
          LLM_MODEL            — model name (provider default if unset)
          LLM_API_BASE         — LLM server URL (default: http://localhost:11434)
          LLM_API_KEY          — API key for authenticated endpoints
          MEMFLOW_BACKEND      — emulated | file | memmachine (default: emulated)
          MEMFLOW_DATA_DIR     — data directory for file backend (default: ./memflow_data)
          MEMMACHINE_BASE_URL  — MemMachine server URL (default: http://localhost:8080)
          MEMMACHINE_ORG_ID    — MemMachine org ID (default: default)
          MEMMACHINE_PROJECT   — MemMachine project ID (default: memflow)
          MEMMACHINE_API_KEY   — MemMachine API key (optional)
        """
        import os
        provider = os.getenv("LLM_PROVIDER", "ollama")
        model = os.getenv("LLM_MODEL")
        api_base = os.getenv("LLM_API_BASE", "http://localhost:11434")
        api_key = os.getenv("LLM_API_KEY")
        llm = LLMFactory.create(provider, model=model, api_base=api_base, api_key=api_key)

        backend = os.getenv("MEMFLOW_BACKEND", "emulated")
        mm_url = os.getenv("MEMMACHINE_BASE_URL", "http://localhost:8080")
        mm_org = os.getenv("MEMMACHINE_ORG_ID", "default")
        mm_proj = os.getenv("MEMMACHINE_PROJECT", "memflow")
        mm_key = os.getenv("MEMMACHINE_API_KEY")

        store: BaseStore
        bypass: MemMachineBypass | None = None

        if backend == "file":
            data_dir = os.getenv("MEMFLOW_DATA_DIR", "./memflow_data")
            store = FileStore(data_dir=data_dir)
        elif backend == "memmachine":
            store = MemMachineStore(
                base_url=mm_url, org_id=mm_org, project_id=mm_proj, api_key=mm_key
            )
            bypass = MemMachineBypass(
                base_url=mm_url, org_id=mm_org, project_id=mm_proj, api_key=mm_key
            )
        else:
            store = EmulatedStore()

        return cls(llm=llm, store=store, bypass=bypass)

    # ------------------------------------------------------------------
    # add
    # ------------------------------------------------------------------

    def add(
        self,
        messages: str | list[dict] | None = None,
        procedure: Procedure | None = None,
        user_id: str = "default",
    ) -> dict:
        """Store a procedure.

        Path 1 — direct: pass a Procedure object via `procedure=`.
        Path 2 — extract: pass conversation text/messages via `messages=`.
        """
        if procedure is not None:
            self.store.add(procedure)
            return {"id": procedure.id, "title": procedure.title, "event": "ADD"}

        if messages is None:
            raise ValueError("Either 'messages' or 'procedure' must be provided")

        return self._extract_and_store(messages, user_id)

    def _extract_and_store(
        self,
        messages: str | list[dict],
        user_id: str,
    ) -> dict:
        # Normalize
        if isinstance(messages, str):
            combined = messages
            msg_list = [{"role": "user", "content": messages}]
        else:
            combined = " ".join(m.get("content", "") for m in messages)
            msg_list = messages

        # Stage 1: keyword heuristic
        if not self._is_likely_procedural(combined):
            return {"results": [], "skipped": "no procedural keywords detected"}

        # Stage 2: LLM classification
        memory_type = self._classify_memory_type(combined)
        if memory_type in ("semantic", "episodic"):
            if self._bypass is not None:
                try:
                    self._bypass.add(combined, memory_type, user_id)
                except Exception:
                    pass  # bypass failures are non-critical
                return {"results": [], "routed_to": "bypass", "type": memory_type}
            return {"results": [], "skipped": f"classified as {memory_type}"}
        if memory_type == "none":
            return {"results": [], "skipped": "classified as none"}

        # Stage 3: LLM extraction
        extraction_messages = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            *msg_list,
            {"role": "user", "content": "Extract procedural memory from the above."},
        ]
        try:
            response = self.llm.generate(extraction_messages)
            data = parse_json(response)
        except Exception as e:
            return {"results": [], "error": str(e)}

        if not data.get("has_procedure"):
            return {"results": []}

        proc = Procedure(
            title=data.get("title", "Untitled"),
            content=data.get("content", ""),
            user_id=user_id,
            category=data.get("category", "general"),
        )
        self.store.add(proc)
        return {"results": [{"id": proc.id, "title": proc.title, "event": "ADD"}]}

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        user_id: str | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Retrieve relevant procedures by similarity."""
        return self.store.search(query, top_k=top_k, user_id=user_id)

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------

    def chat(
        self,
        query: str,
        user_id: str | None = None,
        enable_auto_learn: bool = True,
    ) -> str:
        """Generate a response using retrieved procedures as context.

        If enable_auto_learn is True (default), procedures are extracted
        from the Q&A in a background thread and stored automatically.
        """
        results = self.search(query, user_id=user_id)

        if results:
            procedures_text = "\n\n---\n\n".join(
                f"### {r.procedure.title}\n{r.procedure.content}"
                for r in results
            )
        else:
            procedures_text = "No relevant procedures found."

        messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT.format(
                procedures=procedures_text,
            )},
            {"role": "user", "content": query},
        ]

        response = self.llm.generate(messages)

        if enable_auto_learn:
            self._auto_learn_async(
                messages=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response},
                ],
                user_id=user_id or "default",
            )

        return response

    # ------------------------------------------------------------------
    # plan
    # ------------------------------------------------------------------

    def plan(
        self,
        task: str,
        user_id: str | None = None,
    ) -> TaskPlan:
        """Decompose a task into executable steps.

        Relevant procedures are retrieved first and injected into the planning
        prompt so the LLM can reuse existing SOPs (Retrieve → Planner back-edge).
        """
        results = self.search(task, user_id=user_id)
        context = "\n\n---\n\n".join(
            f"### {r.procedure.title}\n{r.procedure.content}"
            for r in results
        ) if results else ""

        if self._planner is None:
            self._planner = LLMPlanner(self.llm)
        return self._planner.plan(task, context=context)

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(
        self,
        plan: TaskPlan,
        tools: dict[str, Callable[..., str]] | None = None,
    ) -> list[JobResult]:
        """Execute a task plan.

        Args:
            plan:  TaskPlan object containing jobs to execute.
            tools: Extra tools to register {name: callable}.
                   Each callable receives Job.args as keyword arguments
                   and must return a string.

        Returns:
            List of JobResult objects from executing each job.
        """
        if self._executor is None:
            self._executor = ToolRegistry(llm=self.llm)
        if tools:
            for name, fn in tools.items():
                self._executor.register(name, fn)

        job_results: list[JobResult] = [
            self._executor.execute(job) for job in plan.jobs
        ]
        return job_results

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        user_id: str | None = None,
        tools: dict[str, Callable[..., str]] | None = None,
    ) -> RunResult:
        """Plan, execute, and learn from a task.

        Steps:
          1. Retrieve relevant procedures (context for planning).
          2. Plan: LLM decomposes the task into Jobs.
          3. Execute: run each Job with the registered tools.
          4. Learn: extract a reusable Procedure from successful steps
             and store it automatically (Learn → Retrieve back-edge).

        Args:
            task:    High-level task description.
            user_id: User scope for retrieval and storage.
            tools:   Extra tools to register {name: callable}.
                     Each callable receives Job.args as keyword arguments
                     and must return a string.
        """
        task_plan = self.plan(task, user_id=user_id)
        job_results = self.execute(task_plan, tools=tools)

        if self._learner is None:
            self._learner = Learner(self.llm)
        learned = self._learner.extract(
            task, job_results, user_id=user_id or "default"
        )
        if learned is not None:
            self.store.add(learned)

        return RunResult(plan=task_plan, job_results=job_results, learned=learned)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _auto_learn_async(self, messages: list[dict], user_id: str) -> None:
        combined = " ".join(m.get("content", "") for m in messages)
        if not self._is_likely_procedural(combined):
            return
        thread = threading.Thread(
            target=self._extract_and_store,
            args=(messages, user_id),
            daemon=True,
        )
        thread.start()

    def _classify_memory_type(self, content: str) -> str:
        """Stage 2: LLM classification — returns procedural/semantic/episodic/none."""
        messages = [
            {"role": "system", "content": CLASSIFICATION_PROMPT},
            {"role": "user", "content": content},
        ]
        try:
            response = self.llm.generate(messages)
            data = parse_json(response)
            return data.get("type", "procedural")
        except Exception:
            return "procedural"  # fall back to procedural on error

    @staticmethod
    def _is_likely_procedural(text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in PROCEDURAL_KEYWORDS)


