"""
MemFlowManager — core orchestrator for MemFlow.

Public API:
  add(messages, procedure, user_id)  — store a procedure
  search(query, user_id, top_k)      — retrieve procedures
  chat(query, user_id)               — respond using procedure context
"""

from __future__ import annotations

import json
import re
import threading

from memflow.llm import BaseLLM, LLMFactory
from memflow.models import Procedure, SearchResult
from memflow.prompts import CHAT_SYSTEM_PROMPT, CLASSIFICATION_PROMPT, EXTRACTION_PROMPT
from memflow.store import BaseStore, EmulatedStore, FileStore

PROCEDURAL_KEYWORDS = [
    "step", "how to", "first", "then", "finally",
    "deploy", "install", "configure", "setup", "build",
    "run", "execute", "process", "workflow", "procedure",
]


class MemFlowManager:
    def __init__(self, llm: BaseLLM, store: BaseStore | None = None) -> None:
        self.llm = llm
        self.store = store or EmulatedStore()

    @classmethod
    def from_env(cls) -> "MemFlowManager":
        """Create a manager configured from environment variables.

        Environment variables:
          LLM_PROVIDER      — ollama | openai-compatible (default: ollama)
          LLM_MODEL         — model name (provider default if unset)
          LLM_API_BASE      — LLM server URL (default: http://localhost:11434)
          LLM_API_KEY       — API key for authenticated endpoints
          MEMFLOW_BACKEND   — emulated | file | memmachine (default: emulated)
          MEMFLOW_DATA_DIR  — data directory for file backend (default: ./memflow_data)
        """
        import os
        provider = os.getenv("LLM_PROVIDER", "ollama")
        model = os.getenv("LLM_MODEL")
        api_base = os.getenv("LLM_API_BASE", "http://localhost:11434")
        api_key = os.getenv("LLM_API_KEY")
        llm = LLMFactory.create(provider, model=model, api_base=api_base, api_key=api_key)

        backend = os.getenv("MEMFLOW_BACKEND", "emulated")
        if backend == "file":
            data_dir = os.getenv("MEMFLOW_DATA_DIR", "./memflow_data")
            store: BaseStore = FileStore(data_dir=data_dir)
        else:
            store = EmulatedStore()

        return cls(llm=llm, store=store)

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
        if memory_type != "procedural":
            return {"results": [], "skipped": f"classified as {memory_type}"}

        # Stage 3: LLM extraction
        extraction_messages = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            *msg_list,
            {"role": "user", "content": "Extract procedural memory from the above."},
        ]
        try:
            response = self.llm.generate(extraction_messages)
            data = _parse_json(response)
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
            data = _parse_json(response)
            return data.get("type", "procedural")
        except Exception:
            return "procedural"  # fall back to procedural on error

    @staticmethod
    def _is_likely_procedural(text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in PROCEDURAL_KEYWORDS)


def _parse_json(text: str) -> dict:
    """Extract a JSON object from LLM output, stripping markdown fences and triple quotes."""
    # Remove markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Replace triple quotes with single quotes
    text = re.sub(r'"""\s*', '"', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object by matching braces
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            json_str = text[start:end + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    return {}
