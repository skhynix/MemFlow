# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

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

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from memflow.executor import ToolRegistry
from memflow.learner import Learner
from memflow.llm import BaseLLM, LLMFactory, parse_json
from memflow.models import Procedure, RunResult, SearchResult, TaskPlan, Step, StepResult
from memflow.planner import LLMPlanner
from memflow.prompts import CHAT_SYSTEM_PROMPT, CLASSIFICATION_PROMPT, EXTRACTION_PROMPT, INTENT_CLASSIFICATION_PROMPT
from memflow.store import BaseStore, EmulatedStore, FileStore, MemMachineStore, MemMachineBypass, PgVectorStore


def _load_env_file(env_path: str | None = None) -> None:
    """
    Load environment variables from .env file using python-dotenv.

    python-dotenv handles:
    - Inline comments (# after value)
    - Quoted values with # inside (e.g., "value#hash")
    - Escape sequences and multiline values

    Only sets variables that are not already set (priority: env > .env).

    Args:
        env_path: Path to .env file. If None, searches in current directory.
    """
    from dotenv import load_dotenv

    if env_path is None:
        env_path = ".env"

    path = Path(env_path)
    if not path.exists():
        return

    # override=False keeps existing environment variables (env has priority)
    load_dotenv(dotenv_path=path, override=False)


# ---------------------------------------------------------------------------
# Guard structures for execution control
# ---------------------------------------------------------------------------

@dataclass
class GlobalGuard:
    """Guard at run() level - controls overall execution.

    Design document specification:
    - max_attempts: maximum replan attempts
    - cycle detection: detect same goal+failure repetition via hash
    """
    max_attempts: int = 5
    goal_fingerprints: dict = None

    def __post_init__(self):
        if self.goal_fingerprints is None:
            self.goal_fingerprints = {}

    def is_cycle_detected(self, goal: str, failure: str) -> bool:
        """Detect if same goal+failure is repeating."""
        key = (goal, failure)
        if key in self.goal_fingerprints:
            return True
        self.goal_fingerprints[key] = True
        return False

    def can_attempt(self, attempt: int) -> bool:
        return attempt < self.max_attempts


@dataclass
class PlanGuard:
    """Guard at plan() level - controls recursion depth.

    Design document specification:
    - max_depth: maximum vertical recursion depth for plan decomposition
    """
    max_depth: int = 3
    current_depth: int = 0

    def can_recurse(self) -> bool:
        return self.current_depth < self.max_depth

    def enter(self):
        self.current_depth += 1

    def exit(self):
        self.current_depth -= 1


@dataclass
class ToolGuard:
    """Guard at tool call level - controls retry count.

    Design document specification:
    - max_retry: maximum retry for transient failures (network timeout, etc.)
    - retry is for same method, replan is for different approach
    """
    max_retry: int = 3

    def should_retry(self, attempt: int) -> bool:
        return attempt < self.max_retry


class MemFlowManager:
    """
    Core orchestrator for MemFlow operations.

    Coordinates LLM, storage, and Phase 3 components (planner, executor, learner).
    Supports automatic bypass routing for non-procedural memories.

    Environment variables (read when use_env=True):
        Priorities: explicit env var > .env file > fallback defaults

        LLM_PROVIDER              — LLM provider: ollama | openai-compatible
        LLM_MODEL                 — Model name
        LLM_API_BASE              — LLM server URL
        LLM_API_KEY               — API key for authenticated endpoints
        MEMFLOW_BACKEND           — Storage backend: emulated | file | memmachine | pgvector
        MEMFLOW_FILE_DIR          — File directory for FileStore
        PGVECTOR_BASE_URL         — PostgreSQL URL for PgVectorStore
        PGVECTOR_EMBEDDING_MODEL  — Embedding model
        PGVECTOR_EMBEDDING_API_BASE — Embedding API base URL
        PGVECTOR_EMBEDDING_API_KEY  — Embedding API key
        PGVECTOR_EMBEDDING_DIMENSIONS — Embedding dimensions
        MEMMACHINE_BASE_URL       — MemMachine server URL
        MEMMACHINE_ORG_ID         — MemMachine organization ID
        MEMMACHINE_PROJECT        — MemMachine project ID
        MEMMACHINE_API_KEY        — MemMachine API key (optional)

    Note:
        When use_env=True, automatically loads .env file from current directory
        if it exists. Values in .env are used only if not already set as
        environment variables.
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
        store: BaseStore | None = None,
        bypass: MemMachineBypass | None = None,
        max_steps_per_iteration: int | None = None,
        max_plan_iterations: int | None = None,
        use_env: bool = True,
    ) -> None:
        # Track which components are explicitly provided (should not be overwritten by .env)
        llm_provided = llm is not None
        store_provided = store is not None
        bypass_provided = bypass is not None

        # Load .env file first (if enabled) so environment variables are available
        if use_env:
            _load_env_file()

        # Validate LLM is provided when use_env=False
        if not use_env and llm is None:
            raise ValueError("LLM must be provided when use_env=False")

        # Determine backend: explicit store type takes priority, then .env, then default
        if store_provided:
            # Infer backend from explicitly provided store type
            if isinstance(store, PgVectorStore):
                backend = "pgvector"
            elif isinstance(store, MemMachineStore):
                backend = "memmachine"
            elif isinstance(store, FileStore):
                backend = "file"
            else:
                backend = "emulated"
        else:
            backend = os.getenv("MEMFLOW_BACKEND", "emulated")

        # LLM Configuration - only from .env if not explicitly provided
        if not llm_provided and use_env:
            llm_provider = os.getenv("LLM_PROVIDER", "ollama")
            llm_model = os.getenv("LLM_MODEL")
            llm_api_base = os.getenv("LLM_API_BASE", "http://localhost:11434")
            llm_api_key = os.getenv("LLM_API_KEY")
            llm = LLMFactory.create(llm_provider, model=llm_model, api_base=llm_api_base, api_key=llm_api_key)

        # Storage Backend
        backend = os.getenv("MEMFLOW_BACKEND", "emulated")
        file_dir = os.getenv("MEMFLOW_FILE_DIR", "./file_data")

        # MemMachine Configuration
        mm_url = os.getenv("MEMMACHINE_BASE_URL", "http://localhost:8080")
        mm_org = os.getenv("MEMMACHINE_ORG_ID", "default")
        mm_proj = os.getenv("MEMMACHINE_PROJECT", "memflow")
        mm_key = os.getenv("MEMMACHINE_API_KEY")

        # PgVector Store Configuration
        pg_url = os.getenv("PGVECTOR_BASE_URL", "postgresql://pgvector:pgvector_password@localhost:5433/pgvector")
        pg_emb = os.getenv("PGVECTOR_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
        pg_emb_api_base = os.getenv("PGVECTOR_EMBEDDING_API_BASE")  # No default - must be set
        pg_emb_api_key = os.getenv("PGVECTOR_EMBEDDING_API_KEY", "EMPTY")
        pg_emb_dim = os.getenv("PGVECTOR_EMBEDDING_DIMENSIONS", "2560")

        if not store_provided and use_env:
            if backend == "file":
                store = FileStore(file_dir=file_dir)
            elif backend == "memmachine":
                store = MemMachineStore(
                    base_url=mm_url, org_id=mm_org, project_id=mm_proj, api_key=mm_key
                )
            elif backend == "pgvector":
                store = PgVectorStore(
                    base_url=pg_url,
                    emb_model=pg_emb,
                    emb_api_base=pg_emb_api_base,
                    emb_api_key=pg_emb_api_key,
                    emb_dim=int(pg_emb_dim),
                )
            else:
                store = EmulatedStore()

        if not bypass_provided and use_env and backend in ("memmachine", "pgvector"):
            bypass_kwargs = {
                "base_url": mm_url,
                "org_id": mm_org,
                "project_id": mm_proj,
                "api_key": mm_key,
            }
            if backend == "pgvector" and store is not None:
                bypass_kwargs["pgvector_store"] = store
            bypass = MemMachineBypass(**bypass_kwargs)

        self.llm = llm
        self.store = store or EmulatedStore()
        self._bypass = bypass  # routes semantic/episodic content to MemMachine
        self._max_steps_per_iteration = max_steps_per_iteration
        self._max_plan_iterations = max_plan_iterations
        # Phase 3 components — lazily initialised on first use
        self._planner: LLMPlanner | None = None
        self._executor: ToolRegistry | None = None
        self._learner: Learner | None = None

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

        # Stage 1: LLM classification (memory type routing)
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

        # Stage 2: LLM extraction (procedure extraction)
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
    # chat - Primary user interface
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        user_id: str | None = None,
        history: list[dict] | None = None,
        allow_execute: bool = False,
    ) -> dict:
        """
        Primary user interface for MemFlow.

        Classifies user intent and routes to appropriate handler:
        - SEARCH: Retrieve and respond with relevant procedures
        - ADD: Extract and store procedural knowledge
        - EXECUTE: Run the task (requires allow_execute=True)
        - CONVERSATION: Respond naturally without memory operations

        Args:
            message:       User's message
            user_id:       User scope for memory operations
            history:       Previous conversation messages for context
            allow_execute: If True, EXECUTE intent will run the task.
                          If False, asks for confirmation first.

        Returns:
            dict with response and metadata:
            - response: str - The response text
            - intents: list[str] - All detected intents
            - primary_intent: str - Main intent for response formatting
            - handler_results: dict - Results from each intent handler
            - requires_confirmation: bool - (optional) True when EXECUTE needs confirmation
        """
        # Combine current message with history for context
        if history:
            context_messages = history[-5:]  # Last 5 messages for context
            full_context = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in context_messages
            ) + f"\nuser: {message}"
        else:
            full_context = message

        # Classify intents (may return multiple)
        intents_data = self._classify_intents(full_context)
        intents = intents_data.get("intents", ["CONVERSATION"])
        primary_intent = intents_data.get("primary", intents[0] if intents else "CONVERSATION")

        # Check EXECUTE confirmation before processing
        if "EXECUTE" in intents and not allow_execute:
            # Ask for confirmation before executing
            return {
                "response": f"I can help you with: {message}\n\nThis will execute a task. Would you like me to proceed? (confirm to execute)",
                "intents": intents,
                "primary_intent": primary_intent,
                "requires_confirmation": True,
            }

        # Process intents in order
        responses = []
        handler_results = {}

        handlers = {
            "SEARCH": self._handle_search,
            "ADD": self._handle_add,
            "EXECUTE": self._handle_execute,
            "CONVERSATION": self._handle_conversation,
        }

        for intent in intents:
            handler = handlers.get(intent, self._handle_conversation)
            result = handler(message, user_id, history)
            responses.append(result.get("response", ""))
            handler_results[intent] = result

        # Combine responses
        combined_response = "\n\n".join(responses) if len(responses) > 1 else responses[0]

        return {
            "response": combined_response,
            "intents": intents,
            "primary_intent": primary_intent,
            "handler_results": handler_results,
        }

    def _classify_intents(self, context: str) -> dict:
        """Classify user intents using LLM. Returns multiple intents if applicable."""
        messages = [
            {"role": "system", "content": INTENT_CLASSIFICATION_PROMPT},
            {"role": "user", "content": context},
        ]
        try:
            response = self.llm.generate(messages)
            data = parse_json(response)
            intents = data.get("intents", [])
            primary = data.get("primary", "CONVERSATION")

            # Validate intents
            valid_intents = []
            for intent in intents:
                if intent in ("SEARCH", "ADD", "EXECUTE", "CONVERSATION"):
                    valid_intents.append(intent)

            if not valid_intents:
                return {"intents": ["CONVERSATION"], "primary": "CONVERSATION"}

            # If CONVERSATION is mixed with others, remove it (others take priority)
            if len(valid_intents) > 1 and "CONVERSATION" in valid_intents:
                valid_intents.remove("CONVERSATION")

            return {"intents": valid_intents, "primary": primary}
        except Exception:
            return {"intents": ["CONVERSATION"], "primary": "CONVERSATION"}

    def _handle_search(self, message: str, user_id: str | None, history: list[dict] | None) -> dict:
        """Handle SEARCH intent - retrieve and respond."""
        results = self.search(message, user_id=user_id, top_k=3)

        if results:
            procedures_text = "\n\n".join(
                f"**{r.procedure.title}** (Category: {r.procedure.category})\n{r.procedure.content}"
                for r in results
            )
            response = f"Found {len(results)} relevant procedure(s):\n\n{procedures_text}"
        else:
            response = "I couldn't find any relevant procedures for that. Let me help with what I know..."
            # Still provide LLM response
            chat_response = self._generate_chat_response(message, [])
            response += f"\n\n{chat_response}"

        return {"response": response, "intent": "SEARCH", "results": results}

    def _handle_add(self, message: str, user_id: str | None, history: list[dict] | None) -> dict:
        """Handle ADD intent - extract and store procedure."""
        # Use existing extraction logic
        result = self._extract_and_store(message, user_id or "default")
        if result.get("results"):
            return {
                "response": f"Saved: {result['results'][0].get('title', 'Procedure')}",
                "intent": "ADD",
                "data": result,
            }
        elif result.get("skipped"):
            return {
                "response": f"I couldn't extract a procedure from that. Reason: {result['skipped']}",
                "intent": "ADD",
                "data": result,
            }
        return {"response": "I couldn't extract a procedure. Please provide clearer step-by-step instructions.", "intent": "ADD"}

    def _handle_execute(self, message: str, user_id: str | None, history: list[dict] | None) -> dict:
        """Handle EXECUTE intent - run the task."""
        try:
            result = self.run(message, user_id=user_id, multi_stage=True)
            # Format execution result
            success_count = sum(1 for r in result.step_results if r.success)
            total = len(result.step_results)

            steps_output = []
            for i, r in enumerate(result.step_results):
                status = "✓" if r.success else "✗"
                step = result.plan.steps[i] if i < len(result.plan.steps) else None
                desc = step.goal if step else "Unknown step"
                steps_output.append(f"{status} {desc}")
                if r.output:
                    steps_output.append(f"    Output: {r.output[:100]}")

            response = f"Executed {total} step(s), {success_count} succeeded:\n\n" + "\n".join(steps_output)

            if result.learned:
                response += f"\n\nLearned: {result.learned.title}"

            return {"response": response, "intent": "EXECUTE", "data": {"result": result}}
        except Exception as e:
            return {"response": f"Execution failed: {str(e)}", "intent": "EXECUTE", "error": str(e)}

    def _handle_conversation(self, message: str, user_id: str | None, history: list[dict] | None) -> dict:
        """Handle CONVERSATION intent - respond naturally."""
        # Search for relevant context
        results = self.search(message, user_id=user_id, top_k=2)
        procedures_text = "\n\n".join(
            f"### {r.procedure.title}\n{r.procedure.content}"
            for r in results
        ) if results else "No relevant procedures found."

        response = self._generate_chat_response(message, results)
        return {"response": response, "intent": "CONVERSATION"}

    def _generate_chat_response(self, message: str, search_results: list) -> str:
        """Generate a natural language response using LLM."""
        procedures_text = "\n\n".join(
            f"### {r.procedure.title}\n{r.procedure.content}"
            for r in search_results
        ) if search_results else "No relevant procedures found."

        messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT.format(procedures=procedures_text)},
            {"role": "user", "content": message},
        ]
        return self.llm.generate(messages)

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
    # plan
    # ------------------------------------------------------------------

    def plan(
        self,
        task: str,
        user_id: str | None = None,
        multi_stage: bool = False,
        executed_steps: list[Step] | None = None,
        max_depth: int | None = None,
    ) -> TaskPlan:
        """Decompose a task into executable steps.

        Relevant procedures are retrieved first and injected into the planning
        prompt so the LLM can reuse existing SOPs (Retrieve → Planner back-edge).

        Args:
            task:             High-level task description.
            user_id:          User scope for retrieval.
            multi_stage:      If True, plan only a few steps for iterative execution.
            executed_steps:   Previously executed steps with attached results.
            max_depth:        Maximum recursion depth for plan decomposition.

        Returns:
            TaskPlan with steps to execute.
        """
        results = self.search(task, user_id=user_id)
        context = "\n\n---\n\n".join(
            f"### {r.procedure.title}\n{r.procedure.content}"
            for r in results
        ) if results else ""

        if self._planner is None:
            self._planner = LLMPlanner(
                llm=self.llm,
                max_steps_per_iteration=self._max_steps_per_iteration or 1,
                max_iterations=self._max_plan_iterations or 5,
            )

        # PlanGuard for max_depth control
        plan_guard = PlanGuard(max_depth=max_depth or 3)

        return self._planner.plan(
            task,
            context=context,
            multi_stage=multi_stage,
            executed_steps=executed_steps,
            plan_guard=plan_guard,
        )

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(
        self,
        plan: TaskPlan,
        tools: dict[str, Callable[..., str]] | None = None,
    ) -> list[StepResult]:
        """Execute a task plan.

        Args:
            plan:  TaskPlan object containing steps to execute.
            tools: Extra tools to register {name: callable}.
                   Each callable receives Step.args as keyword arguments
                   and must return a string.

        Returns:
            List of StepResult objects from executing each step.
        """
        if self._executor is None:
            self._executor = ToolRegistry(llm=self.llm)
        if tools:
            for name, fn in tools.items():
                self._executor.register(name, fn)

        step_results: list[StepResult] = []
        for step in plan.steps:
            result = self._executor.execute_step(step)
            self._attach_step_result(step, result)
            step_results.append(result)
        return step_results

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        user_id: str | None = None,
        tools: dict[str, Callable[..., str]] | None = None,
        multi_stage: bool = True,
    ) -> RunResult:
        """Plan, execute, and learn from a task.

        This is the unified entry point for task execution. Use multi_stage=True
        (default) for adaptive planning that learns from execution results, or
        multi_stage=False for simple single-shot planning.

        Steps:
          1. Retrieve relevant procedures (context for planning).
          2. Plan: LLM decomposes the task into steps.
          3. Execute: run each step with the registered tools.
          4. Learn: extract a reusable Procedure from successful steps
             and store it automatically (Learn → Retrieve back-edge).

        Args:
            task:       High-level task description.
            user_id:    User scope for retrieval and storage.
            tools:      Extra tools to register {name: callable}.
                        Each callable receives Step.args as keyword arguments
                        and must return a string.
            multi_stage: If True, use Reflect-and-Refine loop (plan a few steps,
                        execute, replan). If False, single-shot plan then execute.

        Returns:
            RunResult with plan, step results, and learned procedure.
        """
        if multi_stage:
            return self._run_with_partial_replan(task, user_id=user_id, tools=tools)
        else:
            return self._run_single_shot(task, user_id=user_id, tools=tools)

    def _run_single_shot(
        self,
        task: str,
        user_id: str | None = None,
        tools: dict[str, Callable[..., str]] | None = None,
    ) -> RunResult:
        """Single-shot execution: plan once, execute, learn (internal)."""
        task_plan = self.plan(task, user_id=user_id)
        step_results = self.execute(task_plan, tools=tools)

        if self._learner is None:
            self._learner = Learner(self.llm)

        learned = self._learner.extract(
            task, task_plan.steps, user_id=user_id or "default"
        )
        if learned is not None:
            self.store.add(learned)

        return RunResult(plan=task_plan, step_results=step_results, learned=learned)

    # ------------------------------------------------------------------
    # run_with_partial_replan - Design document compliant execution
    # ------------------------------------------------------------------

    def _run_with_partial_replan(
        self,
        task: str,
        user_id: str | None = None,
        tools: dict[str, Callable[..., str]] | None = None,
    ) -> RunResult:
        """
        Execute task with partial replan for failed steps only.

        Design document implementation:
        - Replan only the failed step's goal, not global goal
        - Keep successful steps, replace only failed ones
        - GlobalGuard prevents infinite loops via cycle detection

        Steps:
          1. Initial plan for the full task
          2. Execute step by step
          3. On failure: replan only that step with its goal
          4. On success: mark step as done, continue to next
          5. Repeat until all steps done or max attempts reached
        """
        # Retrieve context
        results = self.search(task, user_id=user_id)
        context = "\n\n---\n\n".join(
            f"### {r.procedure.title}\n{r.procedure.content}"
            for r in results
        ) if results else ""

        # Initialize components
        if self._planner is None:
            self._planner = LLMPlanner(self.llm)
        if self._executor is None:
            self._executor = ToolRegistry(llm=self.llm)
        if tools:
            for name, fn in tools.items():
                self._executor.register(name, fn)

        # Guards - instance level for cycle detection across replan calls
        if not hasattr(self, '_global_guard'):
            self._global_guard = GlobalGuard(max_attempts=5)
        global_guard = self._global_guard
        attempt = 0

        # Initial plan
        plan = self._planner.plan(task, context=context, multi_stage=False)
        all_step_results: list[StepResult] = []
        executed_steps: list[Step] = []

        # Execute with partial replan
        step_index = 0
        while step_index < len(plan.steps):
            step = plan.steps[step_index]

            # Skip already completed steps
            if step.status == 'done':
                step_index += 1
                continue

            # Execute current step
            step_result = self._execute_step_with_guard(step, tools)
            self._attach_step_result(step, step_result)
            all_step_results.append(step_result)
            executed_steps.append(step)

            if step_result.success:
                step_index += 1
            else:
                # Step failed - replan only this step
                attempt += 1
                if not global_guard.can_attempt(attempt):
                    break  # Max attempts reached

                # Check for cycle (same goal + same failure)
                if global_guard.is_cycle_detected(step.goal, step_result.error):
                    break  # Cycle detected

                # Replan with failed step's goal (not global goal!)
                # Pass the failure context for alternative approach
                replan_context = f"Previous attempt failed: {step_result.error}"
                new_plan = self._planner.plan(
                    step.goal,  # Only this step's goal, not global task
                    context=f"{context}\n\n{replan_context}",
                    multi_stage=False,
                )

                if new_plan.steps:
                    # Replace failed step with new subplan
                    # Insert new steps at current position
                    plan.steps = (
                        plan.steps[:step_index] +
                        new_plan.steps +
                        plan.steps[step_index + 1:]
                    )
                    # Don't increment step_index - process new first step
                else:
                    # LLM couldn't find an alternative.
                    step_index += 1

        # Mark remaining steps as not executed
        for i in range(step_index, len(plan.steps)):
            if plan.steps[i].status == 'pending':
                plan.steps[i].status = 'failed'

        # Learn from successful execution
        learned = None
        successful_steps = [
            step for step in executed_steps if step.result and step.result.success
        ]
        if successful_steps:
            if self._learner is None:
                self._learner = Learner(self.llm)

            learned = self._learner.extract(
                task, executed_steps, user_id=user_id or "default"
            )
            if learned is not None:
                self.store.add(learned)

        return RunResult(
            plan=plan,
            step_results=all_step_results,
            learned=learned,
        )

    def _execute_step_with_guard(self, step: Step, tools: dict | None = None) -> StepResult:
        """Execute a single step with ToolGuard retry logic."""
        tool_guard = ToolGuard(max_retry=3)
        attempt = 0

        while tool_guard.should_retry(attempt):
            result = self._executor.execute_step(step)

            if result.success:
                return result

            if not result.retryable:
                return result  # Non-retryable error

            attempt += 1

        # All retries exhausted
        return StepResult(
            step_id=step.id,
            success=False,
            output="",
            error=f"Failed after {attempt} retries",
            retryable=False,
        )

    @staticmethod
    def _attach_step_result(step: Step, result: StepResult) -> None:
        step.result = result
        step.status = "done" if result.success else "failed"

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _auto_learn_async(self, messages: list[dict], user_id: str) -> None:
        # Keyword filter removed - always attempt extraction for async learning
        thread = threading.Thread(
            target=self._extract_and_store,
            args=(messages, user_id),
            daemon=True,
        )
        thread.start()

    def _classify_memory_type(self, content: str) -> str:
        """Stage 1: LLM classification — returns procedural/semantic/episodic/none."""
        messages = [
            {"role": "system", "content": CLASSIFICATION_PROMPT},
            {"role": "user", "content": content},
        ]
        try:
            response = self.llm.generate(messages)
            data = parse_json(response)
            return data.get("type", "procedural")
        except Exception:
            return "none"  # fall back to none on error (safe default)

    def _validate_step_output(self, step: Step) -> bool:
        """
        Validate that a step result has meaningful output.
        Returns False if output is empty or meaningless.

        Special handling for file operations - if the command creates a file,
        verify the file exists even if output is empty.
        """
        if step.result is None:
            return False

        if not step.result.success:
            return True  # Don't validate failed steps

        output = step.result.output.strip() if step.result.output else ""
        command = step.args.get("command", "") if step.args else ""

        # Empty output - check if it's a file/directory operation that succeeded
        if not output:
            if command:
                import os
                import re

                # Directory creation commands (mkdir) - verify directory exists
                if command.strip().startswith('mkdir '):
                    parts = command.split()
                    # Handle mkdir -p dirname or mkdir dirname
                    dirnames = [p for p in parts if p != 'mkdir' and p != '-p']
                    for dirname in dirnames:
                        if os.path.isdir(dirname):
                            return True

                # Extract filename from common patterns
                filename_match = re.search(r'>\s*(\S+)', command)
                if filename_match:
                    filename = filename_match.group(1)
                    if os.path.exists(filename):
                        # Verify file has content
                        if os.path.getsize(filename) > 0:
                            return True  # File was created with content

                # Touch command
                if command.strip().startswith('touch '):
                    parts = command.split()
                    if len(parts) > 1:
                        filename = parts[-1]
                        if os.path.exists(filename):
                            return True

            return False  # No output and no file/directory created

        # Single character or very short output might be meaningless
        if len(output) < 2:
            return False

        # Common meaningless outputs
        meaningless_patterns = ["$", "%", "#", ">", "", " ", "\n"]
        if output in meaningless_patterns:
            return False

        return True

    def _is_task_complete(self, task: str, steps: list[Step]) -> bool:
        """
        Use LLM to verify if the task has been actually completed.
        Be conservative - only return True if clearly complete.
        """
        # First, check for obvious success patterns
        successful_outputs = [
            step.result.output
            for step in steps
            if step.result and step.result.success and step.result.output
        ]

        # For tasks with "and", check if multiple distinct things were accomplished
        if " and " in task.lower():
            parts = [p.strip() for p in task.lower().split(" and ")]
            # Need at least one successful output per part
            if len(parts) > len(successful_outputs):
                return False  # Not enough outputs to cover all parts

        # Summarize what was accomplished
        summary_parts = []
        for step in steps:
            if step.result is None:
                continue

            desc = step.goal or f"Step {step.id[:8]}"
            if step.result.success and step.result.output:
                summary_parts.append(f"- {desc}: {step.result.output[:200]}")
            elif not step.result.success:
                summary_parts.append(f"- {desc}: FAILED ({step.result.error[:100]})")

        summary = "\n".join(summary_parts) if summary_parts else "No results yet."

        # Ask LLM to verify task completion with stricter criteria
        verification_prompt = f"""
Task: {task}

Execution Summary:
{summary}

Has the task been completed successfully?
Criteria:
1. ALL parts of the task must be done (check for "and", commas, multiple requirements)
2. Steps must produce meaningful output (not empty, not errors)
3. File operations must have created actual files
4. If the same output appears multiple times, count it only ONCE
5. If ANY part is incomplete or failed, answer NO

Respond ONLY with "YES" or "NO".
"""
        messages = [{"role": "user", "content": verification_prompt}]
        try:
            response = self.llm.generate(messages).strip().upper()
            return "YES" in response
        except Exception:
            # On error, assume task is not complete (conservative)
            return False
