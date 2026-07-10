# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Storage backends for MemFlow.

EmulatedStore     — in-memory dict, word-overlap search (testing / demos)
FileStore         — Markdown files on disk, word-overlap search (local dev)
MemMachineStore   — MemMachine VectorDB, semantic search (production)
PgVectorStore     — PostgreSQL + pgvector VectorDB, cosine similarity search (production)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

# Register pgvector for proper vector type handling
from pgvector.psycopg2 import register_vector
from sqlalchemy import create_engine, text

# Default constants for batch operations
DEFAULT_MAX_BATCHES = 32
DEFAULT_MAX_WORKERS = 48

# Default token limit for the embedding model.
# Qwen3-Embedding-4B (.env.example default) supports 8192 tokens. Use char-based
# estimation only — see _count_tokens. Chunk boundaries are approximate; quality
# impact is bounded since chunks are mean-pooled before indexing.
DEFAULT_EMBEDDING_MAX_TOKENS = 8192

from memflow.models import Procedure, SearchResult, procedure_search_text  # noqa: E402

logger = logging.getLogger(__name__)


def _text_score(text: str, query: str) -> float:
    """Word-overlap relevance score in [0, 1]."""
    if not query.strip():
        return 1.0
    text_words = set(text.lower().split())
    query_words = set(query.lower().split())
    if not query_words:
        return 0.0
    return len(text_words & query_words) / len(query_words)


def _matches_filters(
    procedure: Procedure,
    user_id: str | None = None,
    kind: str | None = None,
) -> bool:
    if user_id and procedure.user_id != user_id:
        return False
    if kind is not None and procedure.kind != kind:
        return False
    return True


def _metadata_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _metadata_scalar(value: str) -> str:
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, str) else value
    except Exception:
        return value


def _is_raw_skill_snapshot(procedure: Procedure) -> bool:
    if procedure.kind != "skill":
        return False
    skill = procedure.metadata.get("skill")
    if not isinstance(skill, dict) or not skill:
        return False
    return bool(skill.get("sha256") or procedure.source_path)


def _split_file_record(text: str) -> tuple[str, str] | None:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None

    offset = len(lines[0])
    for line in lines[1:]:
        if line.strip() == "---":
            return text[len(lines[0]) : offset], text[offset + len(line) :]
        offset += len(line)
    return None


class BaseStore(ABC):
    """Abstract base for all storage backends."""

    @abstractmethod
    def add(self, procedure: Procedure | list[Procedure]) -> int: ...

    @abstractmethod
    def search(
        self,
        query: str | list[str],
        top_k: int = 5,
        user_id: str | None = None,
        kind: str | None = "skill",
    ) -> list[SearchResult] | list[list[SearchResult]]: ...

    @abstractmethod
    def get(self, id: str) -> Procedure | None: ...

    @abstractmethod
    def delete(self, id: str | list[str]) -> int: ...

    @abstractmethod
    def list(self, user_id: str | None = None) -> list[Procedure]: ...

    # Async methods - only PgVectorStore implements these
    async def add_async(
        self,
        procedure: Procedure | list[Procedure],
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> int:
        raise NotImplementedError(
            "Async operations are only supported by PgVectorStore"
        )

    async def search_async(
        self,
        query: str | list[str],
        top_k: int = 5,
        user_id: str | None = None,
        kind: str | None = "skill",
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> list[SearchResult] | list[list[SearchResult]]:
        raise NotImplementedError(
            "Async operations are only supported by PgVectorStore"
        )

    async def delete_async(
        self,
        id: str | list[str],
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> int:
        raise NotImplementedError(
            "Async operations are only supported by PgVectorStore"
        )


class EmulatedStore(BaseStore):
    """
    In-memory store with word-overlap search.

    All data is lost on process restart.
    Suitable for Phase 1 validation and testing.
    """

    def __init__(self) -> None:
        self._store: dict[str, Procedure] = {}

    def add(self, procedure: Procedure | list[Procedure]) -> int:
        """Add a procedure or procedures."""
        if isinstance(procedure, list):
            for proc in procedure:
                self._store[proc.id] = proc
            return len(procedure)
        else:
            self._store[procedure.id] = procedure
            return 1

    def search(
        self,
        query: str | list[str],
        top_k: int = 5,
        user_id: str | None = None,
        kind: str | None = "skill",
    ) -> list[SearchResult] | list[list[SearchResult]]:
        """Search using word-overlap scoring."""
        if isinstance(query, list):
            all_results = []
            for q in query:
                results = []
                for proc in self._store.values():
                    if user_id and proc.user_id != user_id:
                        continue
                    if kind is not None and proc.kind != kind:
                        continue
                    score = _text_score(procedure_search_text(proc), q)
                    if score > 0:
                        results.append(SearchResult(procedure=proc, score=score))
                results.sort(key=lambda r: r.score, reverse=True)
                all_results.append(results[:top_k])
            return all_results
        else:
            results = []
            for proc in self._store.values():
                if user_id and proc.user_id != user_id:
                    continue
                if kind is not None and proc.kind != kind:
                    continue
                score = _text_score(procedure_search_text(proc), query)
                if score > 0:
                    results.append(SearchResult(procedure=proc, score=score))
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:top_k]

    def get(self, id: str) -> Procedure | None:
        """Get a single procedure by ID."""
        return self._store.get(id)

    def delete(self, id: str | list[str]) -> int:
        """Delete a procedure or procedures by ID."""
        if isinstance(id, list):
            num_deleted = 0
            for i in id:
                if i in self._store:
                    del self._store[i]
                    num_deleted += 1
            return num_deleted
        else:
            if id in self._store:
                del self._store[id]
                return 1
            return 0

    def list(self, user_id: str | None = None) -> list[Procedure]:
        """Get all procedures, optionally filtered by user_id."""
        procs = list(self._store.values())
        if user_id:
            return [p for p in procs if p.user_id == user_id]
        return procs


class FileStore(BaseStore):
    """
    File-based store persisting each procedure as a Markdown file.

    File format — simplified frontmatter followed by titled content:

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

    def __init__(self, file_dir: str = "./file_data") -> None:
        self._dir = Path(file_dir)
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

        created_at = meta.get("created_at", "")
        updated_at = meta.get("updated_at", created_at)

        return Procedure(
            id=meta.get("id", ""),
            title=title,
            content=content,
            user_id=meta.get("user_id", "default"),
            category=meta.get("category", "general"),
            tags=tags,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _load_all(self) -> list[Procedure]:
        procs = []
        for path in sorted(self._dir.glob("*.md")):
            proc = self._deserialize(path.read_text(encoding="utf-8"))
            if proc and proc.id:
                procs.append(proc)
        return procs

    def add(self, procedure: Procedure | list[Procedure]) -> int:
        """Add a procedure or procedures."""
        if isinstance(procedure, list):
            for proc in procedure:
                self._path(proc.id).write_text(self._serialize(proc), encoding="utf-8")
            return len(procedure)
        else:
            self._path(procedure.id).write_text(
                self._serialize(procedure), encoding="utf-8"
            )
            return 1

    def search(
        self,
        query: str | list[str],
        top_k: int = 5,
        user_id: str | None = None,
        kind: str | None = "skill",
    ) -> list[SearchResult] | list[list[SearchResult]]:
        """Search using word-overlap scoring."""
        if isinstance(query, list):
            all_results = []
            for q in query:
                results = []
                for proc in self._load_all():
                    if user_id and proc.user_id != user_id:
                        continue
                    if kind is not None and proc.kind != kind:
                        continue
                    score = _text_score(procedure_search_text(proc), q)
                    if score > 0:
                        results.append(SearchResult(procedure=proc, score=score))
                results.sort(key=lambda r: r.score, reverse=True)
                all_results.append(results[:top_k])
            return all_results
        else:
            results = []
            for proc in self._load_all():
                if user_id and proc.user_id != user_id:
                    continue
                if kind is not None and proc.kind != kind:
                    continue
                score = _text_score(procedure_search_text(proc), query)
                if score > 0:
                    results.append(SearchResult(procedure=proc, score=score))
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:top_k]

    def get(self, id: str) -> Procedure | None:
        """Get a single procedure by ID."""
        path = self._path(id)
        if path.exists():
            return self._deserialize(path.read_text(encoding="utf-8"))
        return None

    def delete(self, id: str | list[str]) -> int:
        """Delete a procedure or procedures by ID."""
        if isinstance(id, list):
            num_deleted = 0
            for i in id:
                path = self._path(i)
                if path.exists():
                    path.unlink()
                    num_deleted += 1
            return num_deleted
        else:
            path = self._path(id)
            if path.exists():
                path.unlink()
                return 1
            return 0

    def list(self, user_id: str | None = None) -> list[Procedure]:
        """Get all procedures, optionally filtered by user_id."""
        procs = self._load_all()
        if user_id:
            return [p for p in procs if p.user_id == user_id]
        return procs


class MemMachineBypass:
    """
    Write-only bridge that routes non-procedural content to MemMachine.

    When MemFlow classifies content as semantic or episodic, it forwards
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
        pgvector_store: "PgVectorStore | None" = None,
    ) -> None:
        self._base_url = base_url
        self._org_id = org_id
        self._project_id = project_id
        self._api_key = api_key
        self._pgvector_store = pgvector_store  # for procedural memory
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
        if memory_type == "procedural":
            # Route procedural memory to PgVectorStore
            if self._pgvector_store is not None:
                proc = Procedure(
                    id=str(uuid.uuid4()),
                    title=f"Procedural: {user_id}",
                    content=content,
                    user_id=user_id,
                    category="procedural",
                    kind="procedure",
                )
                self._pgvector_store.add(proc)
        else:
            # Route episodic/semantic to MemMachine
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
            result[k] = (
                json.dumps(v, ensure_ascii=False)
                if isinstance(v, (dict, list))
                else str(v)
            )
        return result

    def _to_text(self, procedure: Procedure) -> str:
        return f"# {procedure.title}\n\n{procedure.content}"

    def _to_metadata(self, procedure: Procedure) -> dict:
        return self._sanitize(
            {
                "mm_type": self._MM_TYPE,
                "record_id": procedure.id,
                "user_id": procedure.user_id,
                "category": procedure.category,
                "tags": procedure.tags,
                "kind": procedure.kind,
                "source_path": procedure.source_path,
                "metadata": procedure.metadata,
                "created_at": procedure.created_at,
                "updated_at": procedure.updated_at,
            }
        )

    def _extract_episodes(self, raw: Any) -> list[Any]:
        """Extract episodes from SearchResult (both long-term and short-term)."""
        episodes = []
        if raw is None or raw.content is None or raw.content.episodic_memory is None:
            return episodes
        if raw.content.episodic_memory.long_term_memory is not None:
            episodes.extend(raw.content.episodic_memory.long_term_memory.episodes)
        if raw.content.episodic_memory.short_term_memory is not None:
            episodes.extend(raw.content.episodic_memory.short_term_memory.episodes)
        return episodes

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
        metadata = _metadata_json(meta.get("metadata", "{}"))
        source_path = meta.get("source_path") or None
        created_at = meta.get("created_at", "")
        updated_at = meta.get("updated_at") or created_at

        proc = Procedure(
            id=meta.get("record_id", ep_id),
            title=title,
            content=body,
            user_id=meta.get("user_id", "default"),
            category=meta.get("category", "general"),
            tags=tags,
            kind=meta.get("kind", "skill"),
            source_path=source_path,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
        )
        return proc, ep_id

    def add(
        self,
        procedure: Procedure | list[Procedure],
        batch_size: int = 50,
    ) -> int:
        if isinstance(procedure, list):
            for proc in procedure:
                result = self._get_memory().add(
                    content=self._to_text(proc),
                    metadata=self._to_metadata(proc),
                )
                if isinstance(result, dict):
                    ep_id = str(result.get("id", proc.id))
                elif result is not None:
                    ep_id = str(getattr(result, "id", proc.id))
                else:
                    ep_id = proc.id
                self._index[proc.id] = ep_id
            return len(procedure)
        else:
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
            return 1

    def search(
        self,
        query: str | list[str],
        top_k: int = 5,
        user_id: str | None = None,
        kind: str | None = "skill",
    ) -> list[SearchResult] | list[list[SearchResult]]:
        """Search using MemMachine semantic search."""
        if isinstance(query, list):
            all_results = []
            for q in query:
                raw = self._get_memory().search(query=q, limit=top_k * 3)
                results = []
                for item in self._extract_episodes(raw):
                    score = float(item.score) if item.score is not None else 0.0
                    proc, ep_id = self._parse_item(item)
                    if proc is None:
                        continue
                    if ep_id:
                        self._index[proc.id] = ep_id
                    if not _matches_filters(proc, user_id=user_id, kind=kind):
                        continue
                    results.append(SearchResult(procedure=proc, score=score))
                all_results.append(results[:top_k])
            return all_results
        else:
            raw = self._get_memory().search(query=query, limit=top_k * 3)
            results = []
            for item in self._extract_episodes(raw):
                score = float(item.score) if item.score is not None else 0.0
                proc, ep_id = self._parse_item(item)
                if proc is None:
                    continue
                if ep_id:
                    self._index[proc.id] = ep_id
                if not _matches_filters(proc, user_id=user_id, kind=kind):
                    continue
                results.append(SearchResult(procedure=proc, score=score))
            return results[:top_k]

    def get(self, id: str) -> Procedure | None:
        for proc in self.list():
            if proc.id == id:
                return proc
        return None

    def delete(
        self,
        id: str | list[str],
    ) -> int:
        if isinstance(id, list):
            num_deleted = 0
            for i in id:
                if i not in self._index:
                    self.list()  # hydrate index
                ep_id = self._index.get(i)
                if not ep_id:
                    continue
                try:
                    self._get_memory().delete(ep_id)
                    self._index.pop(i, None)
                    num_deleted += 1
                except Exception:
                    pass
            return num_deleted
        else:
            if id not in self._index:
                self.list()  # hydrate index
            ep_id = self._index.get(id)
            if not ep_id:
                return 0
            try:
                self._get_memory().delete(ep_id)
                self._index.pop(id, None)
                return 1
            except Exception:
                return 0

    def list(self, user_id: str | None = None) -> list[Procedure]:
        raw = self._get_memory().search(query="", limit=10_000)
        procs = []

        for item in self._extract_episodes(raw):
            proc, ep_id = self._parse_item(item)
            if proc is None:
                continue
            if ep_id:
                self._index[proc.id] = ep_id
            if user_id and proc.user_id != user_id:
                continue
            procs.append(proc)
        return procs


class PgVectorStore(BaseStore):
    """
    PostgreSQL + pgvector backed store for procedural memory.

    PgVector's own VectorDB implementation, inspired by MemMachine's semantic memory.
    Procedures are stored with embeddings for semantic search using cosine similarity.

    Embeddings are computed via OpenAI-compatible API with hash-based fallback.

    Index limitation:
        pgvector's ivfflat and hnsw indexes both support up to 2000 dimensions.
        For embeddings with dim > 2000 (e.g., Qwen3-Embedding-4B at 2560 dim),
        no index is created and sequential scan is used instead.
        To use index, set PGVECTOR_EMBEDDING_DIMENSIONS <= 2000 or use a lower-dim model.

    Schema:
        CREATE TABLE IF NOT EXISTS <table_name> (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL DEFAULT 'default',
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            tags JSONB NOT NULL DEFAULT '[]',
            kind TEXT NOT NULL DEFAULT 'skill',
            source_path TEXT,
            metadata JSONB NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            emb vector(2560)
        );
        CREATE INDEX IF NOT EXISTS idx_<table_name>_emb
            ON <table_name> USING <index_type> (emb vector_cosine_ops);  -- only if dim <= 2000

    Environment variables:
        PGVECTOR_BASE_URL              — PostgreSQL URL
        PGVECTOR_EMBEDDING_MODEL       — Embedding model
        PGVECTOR_EMBEDDING_API_BASE    — API base URL (required)
        PGVECTOR_EMBEDDING_API_KEY     — API key
        PGVECTOR_EMBEDDING_DIMENSIONS  — Embedding dimensions
        PGVECTOR_TABLE_NAME            — Table name (default: procedures)
        PGVECTOR_INDEX_TYPE            — Index type: ivfflat or hnsw (default: ivfflat)

    Note:
        PGVECTOR_EMBEDDING_API_BASE must be set via environment variable or
        passed explicitly. No hardcoded default - use .env file or set
        PGVECTOR_EMBEDDING_API_BASE before instantiating.
    """

    def __init__(
        self,
        base_url: str | None = None,
        emb_model: str | None = None,
        emb_api_base: str | None = None,
        emb_api_key: str | None = None,
        emb_dim: int | None = None,
        table_name: str | None = None,
        index_type: str | None = None,
    ) -> None:
        # Load from environment if not provided
        if base_url is None:
            base_url = os.getenv(
                "PGVECTOR_BASE_URL",
                "postgresql://pgvector:pgvector_password@localhost:5433/pgvector",
            )
        if emb_model is None:
            emb_model = os.getenv("PGVECTOR_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
        if emb_api_base is None:
            emb_api_base = os.getenv("PGVECTOR_EMBEDDING_API_BASE")
            if emb_api_base is None:
                raise ValueError(
                    "PGVECTOR_EMBEDDING_API_BASE must be set. "
                    "Add to .env file or set environment variable."
                )
        if emb_api_key is None:
            emb_api_key = os.getenv("PGVECTOR_EMBEDDING_API_KEY", "EMPTY")
        if emb_dim is None:
            emb_dim = int(os.getenv("PGVECTOR_EMBEDDING_DIMENSIONS", "2560"))
        if table_name is None:
            table_name = os.getenv("PGVECTOR_TABLE_NAME", "procedures")
        if index_type is None:
            index_type = os.getenv("PGVECTOR_INDEX_TYPE", "ivfflat")
        self._base_url = base_url
        self._emb_model = emb_model
        self._emb_api_base = emb_api_base
        self._emb_api_key = emb_api_key
        self._emb_dim = emb_dim
        self._table_name = table_name
        self._index_type = index_type

        self._engine: Any = None
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database connection and tables."""
        try:
            engine = create_engine(self._base_url)
            with engine.connect() as conn:
                # First, create the vector extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()

            # Register pgvector for proper vector type handling
            # This must be done AFTER extension creation and on a fresh connection
            with engine.connect() as conn:
                raw_conn = conn.connection.dbapi_connection
                if raw_conn is not None:
                    register_vector(raw_conn)

                # Create table with emb column
                conn.execute(
                    text(f"""
                    CREATE TABLE IF NOT EXISTS {self._table_name} (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL DEFAULT 'default',
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        category TEXT NOT NULL DEFAULT 'general',
                        tags JSONB NOT NULL DEFAULT '[]'::jsonb,
                        kind TEXT NOT NULL DEFAULT 'skill',
                        source_path TEXT,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        emb vector({self._emb_dim})
                    )
                """)
                )

                # Add missing columns if the table already exists (migration).
                conn.execute(
                    text(f"""
                    ALTER TABLE {self._table_name}
                    ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'skill'
                """)
                )
                conn.execute(
                    text(f"""
                    ALTER TABLE {self._table_name}
                    ADD COLUMN IF NOT EXISTS source_path TEXT
                """)
                )
                conn.execute(
                    text(f"""
                    ALTER TABLE {self._table_name}
                    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                """)
                )
                conn.execute(
                    text(f"""
                    ALTER TABLE {self._table_name}
                    ADD COLUMN IF NOT EXISTS updated_at TEXT NOT NULL DEFAULT ''
                """)
                )
                conn.execute(
                    text(f"""
                    ALTER TABLE {self._table_name}
                    ADD COLUMN IF NOT EXISTS emb vector({self._emb_dim})
                """)
                )

                conn.execute(
                    text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_kind
                    ON {self._table_name} (kind)
                """)
                )
                conn.execute(
                    text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_user_kind
                    ON {self._table_name} (user_id, kind)
                """)
                )
                conn.execute(
                    text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_source_path
                    ON {self._table_name} (source_path)
                """)
                )

                # Create index for semantic search
                # ivfflat: up to 2000 dimensions
                # hnsw: also limited to 2000 dimensions in current pgvector
                if self._emb_dim <= 2000:
                    conn.execute(
                        text(f"""
                        CREATE INDEX IF NOT EXISTS idx_{self._table_name}_emb
                        ON {self._table_name} USING {self._index_type} (emb vector_cosine_ops)
                    """)
                    )
                # Skip index creation for dimensions > 2000 (sequential scan will be used)
                conn.commit()

            self._engine = engine
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize PgVectorStore database: {e}"
            ) from e

    def _get_emb_config(self) -> dict:
        """Get embedding API configuration."""
        return {
            "api_base": self._emb_api_base,
            "api_key": self._emb_api_key,
            "model": self._emb_model,
            "dim": self._emb_dim,
        }

    def _get_max_tokens(self) -> int:
        """Get max tokens for the current embedding model.

        Priority:
        1. PGVECTOR_EMBEDDING_MAX_TOKENS environment variable
        2. DEFAULT_EMBEDDING_MAX_TOKENS (8192, matches Qwen3-Embedding-4B)
        """
        env_max = os.getenv("PGVECTOR_EMBEDDING_MAX_TOKENS")
        if env_max:
            try:
                return int(env_max)
            except ValueError:
                logger.warning(
                    "Invalid PGVECTOR_EMBEDDING_MAX_TOKENS=%r, using default", env_max
                )

        return DEFAULT_EMBEDDING_MAX_TOKENS

    def _count_tokens(self, text: str) -> int:
        """Estimate token count in text.

        Uses a character-based heuristic (~4 chars per token). This is an
        approximation — chunk boundaries affect embedding quality since each
        chunk is embedded independently then mean-pooled. For the default
        Qwen3-Embedding-4B model the estimate is conservative; set
        PGVECTOR_EMBEDDING_MAX_TOKENS explicitly to tune chunking frequency.
        """
        return len(text) // 4

    def _compute_emb(self, text: str, max_tokens: int | None = None) -> list[float]:
        """Compute embedding vector using OpenAI-compatible API.

        For long texts exceeding max_tokens, splits into chunks,
        embeds each chunk, and returns the mean embedding.
        """
        if max_tokens is None:
            max_tokens = self._get_max_tokens()

        config = self._get_emb_config()

        # Count tokens to check if chunking is needed
        text_tokens = self._count_tokens(text)

        # Split text into chunks if too long
        if text_tokens <= max_tokens:
            chunks = [text]
        else:
            logger.info(
                "Text has %d tokens (max %d), splitting into chunks",
                text_tokens,
                max_tokens,
            )
            chunks = self._split_text_by_tokens(text, max_tokens)

        if len(chunks) == 1:
            # Single chunk - direct embedding
            try:
                return self._embed_chunk(chunks[0], config)
            except Exception as exc:
                logger.warning(
                    "Embedding API failed (%s: %s); falling back to "
                    "hash-based pseudo-embedding — search results will not "
                    "be semantically meaningful until the endpoint is reachable.",
                    type(exc).__name__,
                    exc,
                )
                return self._hash_emb(chunks[0], config["dim"])

        # Multiple chunks - embed each and average
        chunk_embeddings = []
        for chunk in chunks:
            try:
                emb = self._embed_chunk(chunk, config)
                chunk_embeddings.append(emb)
            except Exception as exc:
                logger.warning(
                    "Chunk embedding failed (%s: %s); using zero vector for chunk.",
                    type(exc).__name__,
                    exc,
                )
                chunk_embeddings.append([0.0] * config["dim"])

        # Mean pooling
        return self._mean_pool(chunk_embeddings)

    def _split_text_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        """Split long text into chunks by estimated token count.

        Uses character-based token estimation (~4 chars/token) and splits at
        sentence boundaries when possible. Chunk boundaries are approximate;
        each chunk is embedded independently then mean-pooled, so boundary
        placement has bounded impact on final embedding quality.
        """
        # Split by sentence endings first
        sentences = re.split(r"(?<=[.!?।।\n])\s+", text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            # Estimate tokens for this sentence
            sentence_tokens = self._count_tokens(sentence)

            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Start new chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # If any chunk still exceeds max_tokens, split by character estimate
        # (fallback for very long sentences without punctuation)
        final_chunks = []
        chars_per_token = len(text) / max(1, self._count_tokens(text)) if text else 4
        max_chars = int(max_tokens * chars_per_token)

        for chunk in chunks:
            if len(chunk) > max_chars:
                # Hard split by characters
                for i in range(0, len(chunk), max_chars):
                    final_chunks.append(chunk[i : i + max_chars])
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _embed_chunk(self, chunk: str, config: dict) -> list[float]:
        """Embed a single chunk of text."""
        url = config["api_base"].rstrip("/") + "/embeddings"
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config["model"],
            "input": chunk,
            "encoding_format": "float",
        }

        response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

    def _hash_emb(self, text: str, dim: int) -> list[float]:
        """Generate a deterministic pseudo-embedding via hashing.

        Word-aware fallback when the embedding API is unavailable: each word
        contributes an MD5-derived vector so distinct vocabulary yields distinct
        vectors, and the result is L2-normalized to match real embeddings.
        """
        emb = [0.0] * dim
        words = text.lower().split()
        for word in words:
            word_hash = hashlib.md5(word.encode()).hexdigest()
            for i in range(min(len(word_hash), dim)):
                val = (int(word_hash[i % len(word_hash)], 16) - 8) / 8.0
                emb[i] += val / max(len(words), 1)
        norm = sum(x * x for x in emb) ** 0.5
        if norm > 0:
            emb = [x / norm for x in emb]
        return emb

    @staticmethod
    def _mean_pool(embeddings: list[list[float]]) -> list[float]:
        """Compute mean of multiple embedding vectors."""
        if not embeddings:
            return []
        dim = len(embeddings[0])
        result = [0.0] * dim
        for emb in embeddings:
            for i, val in enumerate(emb):
                result[i] += val
        for i in range(dim):
            result[i] /= len(embeddings)
        return result

    async def _compute_emb_async(
        self, text: str, max_tokens: int | None = None
    ) -> list[float]:
        """Compute embedding vector asynchronously using httpx.AsyncClient.

        For long texts exceeding max_tokens, splits into chunks,
        embeds each chunk, and returns the mean embedding.
        """

        if max_tokens is None:
            max_tokens = self._get_max_tokens()

        config = self._get_emb_config()

        # Count tokens to check if chunking is needed
        text_tokens = self._count_tokens(text)

        # Split text into chunks if too long
        if text_tokens <= max_tokens:
            chunks = [text]
        else:
            logger.info(
                "Text has %d tokens (max %d), splitting into chunks",
                text_tokens,
                max_tokens,
            )
            chunks = self._split_text_by_tokens(text, max_tokens)

        if len(chunks) == 1:
            try:
                return await self._embed_chunk_async(chunks[0], config)
            except Exception as exc:
                logger.warning(
                    "Async embedding API failed (%s: %s); falling back to "
                    "hash-based pseudo-embedding — search results will not "
                    "be semantically meaningful until the endpoint is reachable.",
                    type(exc).__name__,
                    exc,
                )
                return self._hash_emb(chunks[0], config["dim"])

        # Multiple chunks - embed each sequentially and average
        chunk_embeddings = []
        for chunk in chunks:
            try:
                emb = await self._embed_chunk_async(chunk, config)
                chunk_embeddings.append(emb)
            except Exception as exc:
                logger.warning(
                    "Async chunk embedding failed (%s: %s); using zero vector for chunk.",
                    type(exc).__name__,
                    exc,
                )
                chunk_embeddings.append([0.0] * config["dim"])

        return self._mean_pool(chunk_embeddings)

    async def _embed_chunk_async(self, chunk: str, config: dict) -> list[float]:
        """Embed a single chunk of text asynchronously."""
        url = config["api_base"].rstrip("/") + "/embeddings"
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config["model"],
            "input": chunk,
            "encoding_format": "float",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, headers=headers, json=payload, timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]

    async def _compute_embs_batch_async(
        self,
        texts: list[str],
        batch_size: int = 50,
        max_workers: int = 10,
    ) -> list[list[float]]:
        """Compute embeddings for multiple texts in parallel batches.

        Uses batch API for efficiency, with semaphore to limit concurrent requests.
        Reduced default max_workers from 50 to 10 to avoid overwhelming the embedding API.
        """
        import asyncio
        from asyncio import Semaphore

        config = self._get_emb_config()
        url = config["api_base"].rstrip("/") + "/embeddings"
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        }

        semaphore = Semaphore(max_workers)

        async def compute_batch(batch: list[str]) -> list[list[float]]:
            """Compute embeddings for a batch of texts using batch API."""
            try:
                async with semaphore:
                    payload = {
                        "model": config["model"],
                        "input": batch,
                        "encoding_format": "float",
                    }
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            url, headers=headers, json=payload, timeout=120.0
                        )
                    response.raise_for_status()
                    data = response.json()
                    return [item["embedding"] for item in data["data"]]
            except Exception as exc:
                logger.warning(
                    "Async batch embedding API failed (%s: %s); falling back to individual embedding.",
                    type(exc).__name__,
                    exc,
                )

                # Fallback: compute individually in parallel, bounded by semaphore
                async def _embed_one(text: str) -> list[float]:
                    async with semaphore:
                        return await self._compute_emb_async(text)

                return await asyncio.gather(*(_embed_one(text) for text in batch))

        # Process in batches for memory efficiency
        all_embs = []
        batch_tasks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_tasks.append(compute_batch(batch))

        results = await asyncio.gather(*batch_tasks)
        for batch_embs in results:
            all_embs.extend(batch_embs)
        return all_embs

    def _compute_embs_batch(
        self,
        texts: list[str],
        batch_size: int = 5,
    ) -> list[list[float]]:
        """Compute embeddings for multiple texts using batch API calls.

        Groups texts into batches and sends each batch in a single API request
        to reduce HTTP overhead. Falls back to per-text _compute_emb (which
        handles chunking for long texts) when the batch API fails.
        """
        config = self._get_emb_config()
        url = config["api_base"].rstrip("/") + "/embeddings"
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        }

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                payload = {
                    "model": config["model"],
                    "input": batch,
                    "encoding_format": "float",
                }
                response = httpx.post(url, headers=headers, json=payload, timeout=120.0)
                response.raise_for_status()
                data = response.json()
                all_embeddings.extend(item["embedding"] for item in data["data"])
            except Exception as exc:
                logger.warning(
                    "Batch embedding API failed (%s: %s); falling back to "
                    "individual embedding.",
                    type(exc).__name__,
                    exc,
                )
                for text in batch:
                    all_embeddings.append(self._compute_emb(text))
        return all_embeddings

    def _to_text(self, procedure: Procedure) -> str:
        """Convert procedure to text for embedding."""
        return procedure_search_text(procedure)

    def add(
        self,
        procedure: Procedure | list[Procedure],
        batch_size: int = 10,
    ) -> int:
        """Add a procedure or procedures.

        Args:
            procedure: Single Procedure or list of Procedures
            batch_size: Batch size for embedding API calls (default: 10)

        Returns:
            1 for single, number of inserted procedures for batch
        """
        if isinstance(procedure, list):
            if not procedure:
                return 0
            texts = [self._to_text(proc) for proc in procedure]
            embeddings = self._compute_embs_batch(texts, batch_size=batch_size)
            num_inserted = 0
            for proc, emb in zip(procedure, embeddings):
                try:
                    self._insert_procedure(proc, emb)
                    num_inserted += 1
                except Exception:
                    pass
            return num_inserted
        else:
            text_content = self._to_text(procedure)
            emb = self._compute_emb(text_content)
            self._insert_procedure(procedure, emb)
            return 1

    async def add_async(
        self,
        procedure: Procedure | list[Procedure],
        batch_size: int = 10,
        max_workers: int = 10,
    ) -> int:
        """Add a procedure or procedures asynchronously.

        Args:
            procedure: Single Procedure or list of Procedures
            batch_size: Batch size for embedding API calls (default: 10)
            max_workers: Max concurrent embedding requests (default: 10)

        Returns:
            1 for single, number of inserted procedures for batch
        """
        import asyncio
        from asyncio import Semaphore

        if isinstance(procedure, list):
            if not procedure:
                return 0
            texts = [self._to_text(proc) for proc in procedure]
            embeddings = await self._compute_embs_batch_async(
                texts, batch_size, max_workers
            )
            semaphore = Semaphore(max_workers)

            async def insert_single(proc: Procedure, emb: list[float]) -> int:
                async with semaphore:
                    try:
                        await asyncio.to_thread(self._insert_procedure, proc, emb)
                        return 1
                    except Exception:
                        return 0

            tasks = [
                insert_single(proc, emb) for proc, emb in zip(procedure, embeddings)
            ]
            results = await asyncio.gather(*tasks)
            return sum(results)
        else:
            text_content = self._to_text(procedure)
            emb = await self._compute_emb_async(text_content)
            self._insert_procedure(procedure, emb)
            return 1

    def _insert_procedure(self, procedure: Procedure, emb: list[float]) -> None:
        """Insert a procedure with pre-computed embedding."""
        emb_str = "[" + ",".join(str(v) for v in emb) + "]"

        with self._engine.connect() as conn:
            # Use CAST for the vector type - the ::vector syntax doesn't work with parameters
            conn.execute(
                text(f"""
                INSERT INTO {self._table_name} (
                    id, user_id, title, content, category, tags, kind, source_path,
                    metadata, created_at, updated_at, emb
                )
                VALUES (
                    :id, :user_id, :title, :content, :category, :tags, :kind,
                    :source_path, :metadata, :created_at, :updated_at,
                    CAST(:emb AS vector)
                )
                ON CONFLICT (id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    category = EXCLUDED.category,
                    tags = EXCLUDED.tags,
                    kind = EXCLUDED.kind,
                    source_path = EXCLUDED.source_path,
                    metadata = EXCLUDED.metadata,
                    created_at = EXCLUDED.created_at,
                    updated_at = EXCLUDED.updated_at,
                    emb = EXCLUDED.emb
            """),
                {
                    "id": procedure.id,
                    "user_id": procedure.user_id,
                    "title": procedure.title,
                    "content": procedure.content,
                    "category": procedure.category,
                    "tags": json.dumps(procedure.tags),
                    "kind": procedure.kind,
                    "source_path": procedure.source_path,
                    "metadata": json.dumps(procedure.metadata),
                    "created_at": procedure.created_at,
                    "updated_at": procedure.updated_at,
                    "emb": emb_str,
                },
            )
            conn.commit()

    @staticmethod
    def _procedure_from_row(row: Any) -> Procedure:
        try:
            tags = json.loads(row.tags) if isinstance(row.tags, str) else row.tags
        except Exception:
            tags = []

        return Procedure(
            id=row.id,
            user_id=row.user_id,
            title=row.title,
            content=row.content,
            category=row.category,
            tags=tags or [],
            kind=getattr(row, "kind", "skill") or "skill",
            source_path=getattr(row, "source_path", None) or None,
            metadata=_metadata_json(getattr(row, "metadata", {}) or {}),
            created_at=row.created_at,
            updated_at=getattr(row, "updated_at", "") or row.created_at,
        )

    def _search_with_emb(
        self,
        query_emb: list[float],
        top_k: int,
        user_id: str | None = None,
        kind: str | None = "skill",
    ) -> list[SearchResult]:
        """Search using pre-computed query embedding."""
        emb_str = "[" + ",".join(str(v) for v in query_emb) + "]"

        with self._engine.connect() as conn:
            filters = []
            params = {"emb": emb_str, "limit": top_k}
            if user_id:
                filters.append("user_id = :user_id")
                params["user_id"] = user_id
            if kind is not None:
                filters.append("kind = :kind")
                params["kind"] = kind
            filter_clause = "AND " + " AND ".join(filters) if filters else ""

            query_sql = text(f"""
                SELECT
                    id, user_id, title, content, category, tags, kind, source_path,
                    metadata, created_at, updated_at,
                       1 - (emb <=> CAST(:emb AS vector)) AS score
                FROM {self._table_name}
                WHERE TRUE {filter_clause}
                ORDER BY emb <=> CAST(:emb AS vector)
                LIMIT :limit
            """)

            result = conn.execute(query_sql, params)
            rows = result.fetchall()

        results = []
        for row in rows:
            proc = self._procedure_from_row(row)
            results.append(SearchResult(procedure=proc, score=float(row.score)))

        return results

    def search(
        self,
        query: str | list[str],
        top_k: int = 5,
        user_id: str | None = None,
        kind: str | None = "skill",
        batch_size: int = 10,
    ) -> list[SearchResult] | list[list[SearchResult]]:
        """Search for procedures by semantic similarity.

        Args:
            query: Single query string or list of queries
            top_k: Number of results per query
            user_id: User ID for filtering
            batch_size: Batch size for embedding API calls (default: 10) - PgVectorStore only

        Returns:
            Single list for single query, list of lists for batch
        """
        if isinstance(query, list):
            query_embs = self._compute_embs_batch(query, batch_size=batch_size)
            results = []
            for query_emb in query_embs:
                search_results = self._search_with_emb(query_emb, top_k, user_id, kind)
                results.append(search_results)
            return results
        else:
            query_emb = self._compute_emb(query)
            return self._search_with_emb(query_emb, top_k, user_id, kind)

    async def search_async(
        self,
        query: str | list[str],
        top_k: int = 5,
        user_id: str | None = None,
        kind: str | None = "skill",
        batch_size: int = 10,
        max_workers: int = 10,
    ) -> list[SearchResult] | list[list[SearchResult]]:
        """Search for procedures by semantic similarity asynchronously.

        Args:
            query: Single query string or list of queries
            top_k: Number of results per query
            user_id: User ID for filtering
            batch_size: Batch size for embedding API calls (default: 10) - PgVectorStore only
            max_workers: Max concurrent requests (default: 10)

        Returns:
            Single list for single query, list of lists for batch
        """
        import asyncio
        from asyncio import Semaphore

        if isinstance(query, list):
            query_embs = await self._compute_embs_batch_async(
                query, batch_size=batch_size, max_workers=max_workers
            )
            semaphore = Semaphore(max_workers)

            async def search_single(query_emb: list[float]) -> list[SearchResult]:
                async with semaphore:
                    return await asyncio.to_thread(
                        self._search_with_emb, query_emb, top_k, user_id, kind
                    )

            tasks = [search_single(qe) for qe in query_embs]
            return await asyncio.gather(*tasks)
        else:
            query_emb = await self._compute_emb_async(query)
            return await asyncio.to_thread(
                self._search_with_emb, query_emb, top_k, user_id, kind
            )

    async def delete_async(
        self,
        id: str | list[str],
        max_workers: int = 50,
    ) -> int:
        """Delete a procedure or procedures asynchronously.

        Args:
            id: Single ID or list of IDs
            max_workers: Max concurrent operations (default: 50)

        Returns:
            int: Number of procedures deleted
        """
        import asyncio
        from asyncio import Semaphore

        if isinstance(id, list):
            semaphore = Semaphore(max_workers)

            async def delete_single(i: str) -> int:
                async with semaphore:
                    try:
                        return await asyncio.to_thread(self.delete, i)
                    except Exception:
                        return 0

            tasks = [delete_single(i) for i in id]
            results = await asyncio.gather(*tasks)
            return sum(results)
        else:
            result = await asyncio.to_thread(self.delete, id)
            return result

    def get(self, id: str) -> Procedure | None:
        """Get a procedure by ID."""
        with self._engine.connect() as conn:
            result = conn.execute(
                text(f"""
                SELECT
                    id, user_id, title, content, category, tags, kind, source_path,
                    metadata, created_at, updated_at
                FROM {self._table_name} WHERE id = :id
            """),
                {"id": id},
            )
            row = result.fetchone()

        if row is None:
            return None

        return self._procedure_from_row(row)

    def delete(
        self,
        id: str | list[str],
    ) -> int:
        """Delete a procedure or procedures by ID.

        Returns:
            int: Number of procedures deleted
        """
        if isinstance(id, list):
            num_deleted = 0
            for i in id:
                if self.delete(i):
                    num_deleted += 1
            return num_deleted
        else:
            try:
                with self._engine.connect() as conn:
                    result = conn.execute(
                        text(f"""
                        DELETE FROM {self._table_name} WHERE id = :id
                    """),
                        {"id": id},
                    )
                    conn.commit()
                    return result.rowcount
            except Exception:
                return 0

    def list(self, user_id: str | None = None) -> list[Procedure]:
        """List all procedures, optionally filtered by user_id."""
        with self._engine.connect() as conn:
            if user_id:
                result = conn.execute(
                    text(f"""
                    SELECT
                        id, user_id, title, content, category, tags, kind, source_path,
                        metadata, created_at, updated_at
                    FROM {self._table_name} WHERE user_id = :user_id
                """),
                    {"user_id": user_id},
                )
            else:
                result = conn.execute(
                    text(f"""
                    SELECT
                        id, user_id, title, content, category, tags, kind, source_path,
                        metadata, created_at, updated_at
                    FROM {self._table_name}
                """),
                )
            rows = result.fetchall()

        return [self._procedure_from_row(row) for row in rows]
