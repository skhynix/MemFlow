"""
Storage backends for MemFlow.

EmulatedStore  — in-memory dict, word-overlap search (testing / demos)
FileStore      — Markdown files on disk, word-overlap search (local dev)
MemMachineStore — MemMachine VectorDB, semantic search (production)
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from memflow.models import Procedure, SearchResult


def _text_score(text: str, query: str) -> float:
    """Word-overlap relevance score in [0, 1]."""
    if not query.strip():
        return 1.0
    text_words = set(text.lower().split())
    query_words = set(query.lower().split())
    if not query_words:
        return 0.0
    return len(text_words & query_words) / len(query_words)


class BaseStore(ABC):
    """Abstract base for all storage backends."""

    @abstractmethod
    def add(self, procedure: Procedure) -> None: ...

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[SearchResult]: ...

    @abstractmethod
    def get(self, id: str) -> Procedure | None: ...

    @abstractmethod
    def delete(self, id: str) -> bool: ...

    @abstractmethod
    def list_all(self, user_id: str | None = None) -> list[Procedure]: ...


class EmulatedStore(BaseStore):
    """
    In-memory store with word-overlap search.

    All data is lost on process restart.
    Suitable for Phase 1 validation and testing.
    """

    def __init__(self) -> None:
        self._store: dict[str, Procedure] = {}

    def add(self, procedure: Procedure) -> None:
        self._store[procedure.id] = procedure

    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[SearchResult]:
        results = []
        for proc in self._store.values():
            if user_id and proc.user_id != user_id:
                continue
            score = _text_score(proc.title + " " + proc.content, query)
            if score > 0:
                results.append(SearchResult(procedure=proc, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def get(self, id: str) -> Procedure | None:
        return self._store.get(id)

    def delete(self, id: str) -> bool:
        if id in self._store:
            del self._store[id]
            return True
        return False

    def list_all(self, user_id: str | None = None) -> list[Procedure]:
        procs = list(self._store.values())
        if user_id:
            procs = [p for p in procs if p.user_id == user_id]
        return procs


class FileStore(BaseStore):
    """
    File-based store persisting each procedure as a Markdown file.

    File format — frontmatter (key: value) followed by titled content:

        ---
        id: <uuid>
        user_id: <user>
        category: <category>
        tags: ["tag1", "tag2"]
        created_at: <iso-timestamp>
        ---
        # <title>

        <content>

    Persists across process restarts. Suitable for local development.
    """

    def __init__(self, data_dir: str = "./memflow_data") -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, id: str) -> Path:
        return self._dir / f"{id}.md"

    def _serialize(self, procedure: Procedure) -> str:
        return (
            "---\n"
            f"id: {procedure.id}\n"
            f"user_id: {procedure.user_id}\n"
            f"category: {procedure.category}\n"
            f"tags: {json.dumps(procedure.tags, ensure_ascii=False)}\n"
            f"created_at: {procedure.created_at}\n"
            "---\n"
            f"# {procedure.title}\n"
            "\n"
            f"{procedure.content}\n"
        )

    def _deserialize(self, text: str) -> Procedure | None:
        if not text.startswith("---"):
            return None
        parts = text.split("---", 2)
        if len(parts) < 3:
            return None

        meta: dict[str, str] = {}
        for line in parts[1].strip().splitlines():
            if ": " in line:
                k, v = line.split(": ", 1)
                meta[k.strip()] = v.strip()

        body = parts[2].strip()
        lines = body.splitlines()
        title = ""
        content_start = 0
        for i, line in enumerate(lines):
            if line.startswith("# "):
                title = line[2:].strip()
                content_start = i + 1
                break

        while content_start < len(lines) and not lines[content_start].strip():
            content_start += 1
        content = "\n".join(lines[content_start:])

        try:
            tags = json.loads(meta.get("tags", "[]"))
        except Exception:
            tags = []

        return Procedure(
            id=meta.get("id", ""),
            title=title,
            content=content,
            user_id=meta.get("user_id", "default"),
            category=meta.get("category", "general"),
            tags=tags,
            created_at=meta.get("created_at", ""),
        )

    def _load_all(self) -> list[Procedure]:
        procs = []
        for path in sorted(self._dir.glob("*.md")):
            proc = self._deserialize(path.read_text(encoding="utf-8"))
            if proc and proc.id:
                procs.append(proc)
        return procs

    def add(self, procedure: Procedure) -> None:
        self._path(procedure.id).write_text(
            self._serialize(procedure), encoding="utf-8"
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[SearchResult]:
        results = []
        for proc in self._load_all():
            if user_id and proc.user_id != user_id:
                continue
            score = _text_score(proc.title + " " + proc.content, query)
            if score > 0:
                results.append(SearchResult(procedure=proc, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def get(self, id: str) -> Procedure | None:
        path = self._path(id)
        if not path.exists():
            return None
        return self._deserialize(path.read_text(encoding="utf-8"))

    def delete(self, id: str) -> bool:
        path = self._path(id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_all(self, user_id: str | None = None) -> list[Procedure]:
        procs = self._load_all()
        if user_id:
            procs = [p for p in procs if p.user_id == user_id]
        return procs


class MemMachineBypass:
    """
    Write-only bridge that routes non-procedural content to MemMachine.

    When MemFlowManager classifies content as semantic or episodic, it forwards
    the content here so MemMachine can store it in the appropriate backend
    (VectorDB for semantic, GraphDB for episodic).

    Requires the `memmachine-client` Python package and a running MemMachine server.
    Connection is deferred to first use (lazy initialization).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        org_id: str = "default",
        project_id: str = "memflow",
        api_key: str | None = None,
    ) -> None:
        self._base_url = base_url
        self._org_id = org_id
        self._project_id = project_id
        self._api_key = api_key
        self._memory: Any = None
        self._lock = threading.Lock()

    def _get_memory(self) -> Any:
        if self._memory is not None:
            return self._memory
        with self._lock:
            if self._memory is None:
                import memmachine_client as memmachine
                kwargs: dict[str, Any] = {"base_url": self._base_url}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                client = memmachine.MemMachineClient(**kwargs)
                project = client.get_or_create_project(
                    org_id=self._org_id, project_id=self._project_id
                )
                self._memory = project.memory()
        return self._memory

    def add(self, content: str, memory_type: str, user_id: str) -> None:
        """Store content in MemMachine tagged with the given memory type."""
        meta = {"mm_type": memory_type, "user_id": user_id}
        self._get_memory().add(content=content, metadata=meta)


class MemMachineStore(BaseStore):
    """
    MemMachine-backed store for procedural memory.

    Procedures are stored as episodic memories with metadata tag
    mm_type='procedural', which distinguishes them from semantic and episodic
    memories also residing in the same MemMachine project.

    An in-memory index (procedure.id → MemMachine episode id) is populated as
    a side-effect of add() and search() to allow O(1) delete without a full scan.
    On a cache-miss in delete(), list_all() is called once to hydrate the index.

    Requires the `memmachine-client` Python package and a running MemMachine server.
    Connection is deferred to first use (lazy initialization).
    """

    _MM_TYPE = "procedural"

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        org_id: str = "default",
        project_id: str = "memflow",
        api_key: str | None = None,
    ) -> None:
        self._base_url = base_url
        self._org_id = org_id
        self._project_id = project_id
        self._api_key = api_key
        self._memory: Any = None
        self._lock = threading.Lock()
        self._index: dict[str, str] = {}  # procedure.id → MemMachine episode id

    def _get_memory(self) -> Any:
        if self._memory is not None:
            return self._memory
        with self._lock:
            if self._memory is None:
                import memmachine_client as memmachine
                kwargs: dict[str, Any] = {"base_url": self._base_url}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                client = memmachine.MemMachineClient(**kwargs)
                project = client.get_or_create_project(
                    org_id=self._org_id, project_id=self._project_id
                )
                self._memory = project.memory()
        return self._memory

    @staticmethod
    def _sanitize(meta: dict) -> dict:
        """MemMachine requires all metadata values to be strings."""
        result = {}
        for k, v in meta.items():
            if v is None:
                continue
            result[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, list) else str(v)
        return result

    def _to_text(self, procedure: Procedure) -> str:
        return f"# {procedure.title}\n\n{procedure.content}"

    def _to_metadata(self, procedure: Procedure) -> dict:
        return self._sanitize({
            "mm_type": self._MM_TYPE,
            "record_id": procedure.id,
            "user_id": procedure.user_id,
            "category": procedure.category,
            "tags": procedure.tags,
            "created_at": procedure.created_at,
        })

    def _parse_item(self, item: Any) -> tuple[Procedure | None, str]:
        """Parse a MemMachine search result → (Procedure | None, episode_id)."""
        if isinstance(item, dict):
            ep_id = str(item.get("id", ""))
            content = item.get("content", "")
            meta = item.get("metadata", {}) or {}
        else:
            ep_id = str(getattr(item, "id", ""))
            content = str(getattr(item, "content", "") or "")
            meta = getattr(item, "metadata", {}) or {}

        if meta.get("mm_type") != self._MM_TYPE:
            return None, ep_id

        lines = content.strip().splitlines()
        title = lines[0].lstrip("# ").strip() if lines else ""
        start = 1
        while start < len(lines) and not lines[start].strip():
            start += 1
        body = "\n".join(lines[start:])

        try:
            tags = json.loads(meta.get("tags", "[]"))
        except Exception:
            tags = []

        proc = Procedure(
            id=meta.get("record_id", ep_id),
            title=title,
            content=body,
            user_id=meta.get("user_id", "default"),
            category=meta.get("category", "general"),
            tags=tags,
            created_at=meta.get("created_at", ""),
        )
        return proc, ep_id

    def add(self, procedure: Procedure) -> None:
        result = self._get_memory().add(
            content=self._to_text(procedure),
            metadata=self._to_metadata(procedure),
        )
        if isinstance(result, dict):
            ep_id = str(result.get("id", procedure.id))
        elif result is not None:
            ep_id = str(getattr(result, "id", procedure.id))
        else:
            ep_id = procedure.id
        self._index[procedure.id] = ep_id

    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[SearchResult]:
        raw = self._get_memory().search(query=query, limit=top_k * 3)
        results = []
        for item in raw or []:
            score = float(
                item.get("score", 1.0) if isinstance(item, dict)
                else getattr(item, "score", 1.0)
            )
            proc, ep_id = self._parse_item(item)
            if proc is None:
                continue
            if ep_id:
                self._index[proc.id] = ep_id
            if user_id and proc.user_id != user_id:
                continue
            results.append(SearchResult(procedure=proc, score=score))
        return results[:top_k]

    def get(self, id: str) -> Procedure | None:
        for proc in self.list_all():
            if proc.id == id:
                return proc
        return None

    def delete(self, id: str) -> bool:
        if id not in self._index:
            self.list_all()  # hydrate index
        ep_id = self._index.get(id)
        if not ep_id:
            return False
        try:
            self._get_memory().delete(ep_id)
            self._index.pop(id, None)
            return True
        except Exception:
            return False

    def list_all(self, user_id: str | None = None) -> list[Procedure]:
        raw = self._get_memory().search(query="", limit=10_000)
        procs = []
        for item in raw or []:
            proc, ep_id = self._parse_item(item)
            if proc is None:
                continue
            if ep_id:
                self._index[proc.id] = ep_id
            if user_id and proc.user_id != user_id:
                continue
            procs.append(proc)
        return procs
