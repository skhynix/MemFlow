# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Storage backends for MemFlow.

EmulatedStore     — in-memory dict, word-overlap search (testing / demos)
FileStore         — Markdown files on disk, word-overlap search (local dev)
MemMachineStore   — MemMachine VectorDB, semantic search (production)
VectorStore       — Abstract base for vector DB backends with embedding logic
QdrantStore       — Qdrant vector DB, cosine/dot/euclid distance (production)
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
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

# Default constants for batch operations
DEFAULT_MAX_BATCHES = 32
DEFAULT_MAX_WORKERS = 48

# Default token limit for the embedding model.
# Qwen3-Embedding-4B (.env.example default) supports 8192 tokens. Use char-based
# estimation only — see _count_tokens. Chunk boundaries are approximate; quality
# impact is bounded since chunks are mean-pooled before indexing.
DEFAULT_EMBEDDING_MAX_TOKENS = 8192

# Timeout (seconds) for a single embedding API request. Lowered from 300s so a
# wedged/unresponsive embedding server doesn't stall the whole gather for 5
# minutes — the per-text fallback (hash embedding) kicks in much sooner.
EMBEDDING_REQUEST_TIMEOUT = float(os.getenv("EMBEDDING_REQUEST_TIMEOUT", "60"))

# Max retry attempts for transient embedding API errors (connect/timeout/
# 5xx). A failed text falls back to a hash/zero vector rather than stalling.
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "2"))

# ---------------------------------------------------------------------------
# Hybrid search constants
# ---------------------------------------------------------------------------
# Named-vector names used inside hybrid-enabled Qdrant collections.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# FastEmbed BM25 model for client-side sparse vector generation. Self-hosted
# Qdrant cannot use the Document cloud-inference API, so sparse vectors are
# produced locally with Qdrant/bm25 (which expects Modifier.IDF on the server
# side to apply corpus-frequency weighting to the raw term frequencies).
# Read lazily (not at import time) so QDRANT_SPARSE_MODEL set in .env — which
# MemFlow loads in its constructor, after this module is imported — takes effect.
SPARSE_MODEL_DEFAULT = "Qdrant/bm25"


def env_flag(name: str, default: str = "off") -> bool:
    """Read a boolean env var (``1/on/true/yes`` are true)."""
    return os.getenv(name, default).strip().lower() in ("1", "on", "true", "yes")


from memflow.models import Procedure, SearchResult, procedure_search_text  # noqa: E402

logger = logging.getLogger(__name__)


def _id_to_uuid(id_str: str) -> str:
    """Convert any string ID to a deterministic UUID for Qdrant point IDs."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, id_str))


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

    #: Whether this backend can serve ``search_hybrid`` (sparse+dense RRF).
    #: Backends that implement it flip this to True; callers probe it instead
    #: of try/except'ing the call.
    supports_hybrid: bool = False

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

    # Async methods - only QdrantStore implements these
    async def add_async(
        self,
        procedure: Procedure | list[Procedure],
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> int:
        raise NotImplementedError("Async operations are only supported by QdrantStore")

    async def search_async(
        self,
        query: str | list[str],
        top_k: int = 5,
        user_id: str | None = None,
        kind: str | None = "skill",
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> list[SearchResult] | list[list[SearchResult]]:
        raise NotImplementedError("Async operations are only supported by QdrantStore")

    async def delete_async(
        self,
        id: str | list[str],
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> int:
        raise NotImplementedError("Async operations are only supported by QdrantStore")


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


class VectorStore(BaseStore):
    """Abstract base for vector DB backends with embedding logic.

    Owns embedding configuration (model, API base/key, dimensions, query
    instruction, max tokens) and all embedding computation methods (chunking,
    batch, async, hash fallback, mean pool, truncate dim).

    Subclasses (QdrantStore) pass embedding config
    via ``super().__init__()``. CRUD operations remain abstract — subclasses
    implement ``add``, ``search``, ``get``, ``delete``, ``list`` (and async
    variants where supported) using the shared embedding helpers here.
    """

    def __init__(
        self,
        emb_model: str,
        emb_api_base: str,
        emb_api_key: str,
        emb_dim: int,
        query_instruction: str = "",
        emb_max_tokens: int | None = None,
    ) -> None:
        self._emb_model = emb_model
        self._emb_api_base = emb_api_base
        self._emb_api_key = emb_api_key
        self._emb_dim = emb_dim
        self._query_instruction = query_instruction
        self._emb_max_tokens = emb_max_tokens

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _get_emb_config(self) -> dict:
        return {
            "api_base": self._emb_api_base,
            "api_key": self._emb_api_key,
            "model": self._emb_model,
            "dim": self._emb_dim,
        }

    def _get_max_tokens(self) -> int:
        """Get max tokens for the current embedding model.

        Priority:
        1. ``self._emb_max_tokens`` (set from ``VECTOR_EMBEDDING_MAX_TOKENS``)
        2. ``DEFAULT_EMBEDDING_MAX_TOKENS`` (8192, matches Qwen3-Embedding-4B)
        """
        emb_max_tokens = getattr(self, "_emb_max_tokens", None)
        if emb_max_tokens is not None:
            return emb_max_tokens
        return DEFAULT_EMBEDDING_MAX_TOKENS

    def _count_tokens(self, text: str) -> int:
        """Estimate token count in text.

        Uses a character-based heuristic (~4 chars per token). This is an
        approximation — chunk boundaries affect embedding quality since each
        chunk is embedded independently then mean-pooled. For the default
        Qwen3-Embedding-4B model the estimate is conservative; set
        ``VECTOR_EMBEDDING_MAX_TOKENS`` explicitly to tune chunking frequency.
        """
        return len(text) // 4

    def _compute_emb(
        self, text: str, max_tokens: int | None = None, is_query: bool = False
    ) -> list[float]:
        """Compute embedding vector using OpenAI-compatible API.

        For long texts exceeding max_tokens, splits into chunks,
        embeds each chunk, and returns the mean embedding.
        """
        if is_query and self._query_instruction:
            text = f"Instruct: {self._query_instruction}\nQuery: {text}"

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
        headers = {"Content-Type": "application/json"}
        if config["api_key"]:
            headers["Authorization"] = f"Bearer {config['api_key']}"
        payload = {
            "model": config["model"],
            "input": chunk,
            "encoding_format": "float",
        }

        response = httpx.post(
            url, headers=headers, json=payload, timeout=EMBEDDING_REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        emb = data["data"][0]["embedding"]
        return self._truncate_dim(emb, config.get("dim"))

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

    # ------------------------------------------------------------------
    # Sparse (BM25) embedding helpers
    # ------------------------------------------------------------------

    # Model cache keyed by model name, shared across instances (loading a
    # FastEmbed model is expensive). Keyed so stores configured with
    # different QDRANT_SPARSE_MODEL values don't silently share one model.
    _sparse_models: dict = {}
    _sparse_lock = threading.Lock()
    # fastembed's embed()/query_embed() wrap shared per-model state that is
    # not documented as thread-safe; since the model instance is cached
    # class-wide, concurrent add_async/search threads must serialize around
    # inference. Loading and inference use separate locks so a slow model
    # load never blocks searches on other stores' models.
    _sparse_infer_lock = threading.Lock()

    def _get_sparse_model(self) -> Any:
        """Lazily load the FastEmbed BM25 sparse model (cached by name, thread-safe)."""
        model_name = os.getenv("QDRANT_SPARSE_MODEL", SPARSE_MODEL_DEFAULT)
        cached = VectorStore._sparse_models.get(model_name)
        if cached is not None:
            return cached
        with VectorStore._sparse_lock:
            cached = VectorStore._sparse_models.get(model_name)
            if cached is None:
                try:
                    from fastembed import SparseTextEmbedding
                except ImportError as exc:
                    raise ImportError(
                        "fastembed is required for BM25 sparse vectors on this "
                        "collection (it has a sparse vector schema). Install "
                        "with: uv sync --extra hybrid"
                    ) from exc

                cached = SparseTextEmbedding(model_name=model_name)
                VectorStore._sparse_models[model_name] = cached
        return cached

    def _compute_sparse(self, text: str, is_query: bool = False) -> Any:
        """Compute a BM25 sparse vector (indices + values) for one text.

        Returns a ``qdrant_client.models.SparseVector``, or ``None`` when the
        text yields no BM25 tokens (e.g. stopword-only) — callers skip the
        sparse vector rather than sending an empty one, which Qdrant rejects.
        For documents the FastEmbed BM25 model yields term-frequency-style
        values; the IDF re-weighting is applied server-side by Qdrant's
        ``Modifier.IDF``. For queries, ``query_embed`` emits binary presence
        (values of 1).
        """
        from qdrant_client import models

        model = self._get_sparse_model()
        # Materialize the generator inside the lock: fastembed yields lazily
        # and its wrapper is not thread-safe under concurrent calls.
        with VectorStore._sparse_infer_lock:
            vectors = list(model.query_embed(text) if is_query else model.embed(text))
        for sparse in vectors:
            if len(sparse.indices) == 0:
                return None
            return models.SparseVector(
                indices=list(sparse.indices), values=[float(v) for v in sparse.values]
            )
        # Empty text — no tokens, no sparse vector.
        return None

    def _compute_sparse_batch(self, texts: list[str]) -> list[Any]:
        """Compute BM25 sparse vectors for a batch of documents.

        Entries whose text yields no BM25 tokens are returned as ``None`` so
        the upsert path skips the sparse vector instead of sending an empty
        one; the dense vector is still written for such documents.
        """
        from qdrant_client import models

        model = self._get_sparse_model()
        results: list[Any] = []
        num_empty = 0
        # Serialize inference — see _sparse_infer_lock comment.
        with VectorStore._sparse_infer_lock:
            sparse_vectors = list(model.embed(texts))
        for sparse in sparse_vectors:
            if len(sparse.indices) == 0:
                results.append(None)
                num_empty += 1
                continue
            results.append(
                models.SparseVector(
                    indices=list(sparse.indices),
                    values=[float(v) for v in sparse.values],
                )
            )
        if num_empty:
            logger.warning(
                "Sparse embedding produced no tokens for %d/%d documents; "
                "they will be indexed without a sparse vector (invisible to "
                "BM25 search).",
                num_empty,
                len(texts),
            )
        return results

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

    @staticmethod
    def _truncate_dim(emb: list[float], target_dim: int | None) -> list[float]:
        """Truncate embedding to target_dim and L2-normalize (Matryoshka-style).

        When the server returns a vector larger than target_dim, keep the first
        target_dim components and renormalize. This avoids sending the
        ``dimensions`` parameter (which can cause server-side load/queue issues)
        while still producing a reduced-dimension embedding suitable for
        cosine similarity.
        """
        if not target_dim or target_dim <= 0 or len(emb) <= target_dim:
            return emb
        truncated = emb[:target_dim]
        norm = sum(x * x for x in truncated) ** 0.5
        if norm > 0:
            return [x / norm for x in truncated]
        return truncated

    async def _compute_emb_async(
        self, text: str, max_tokens: int | None = None, is_query: bool = False
    ) -> list[float]:
        """Compute embedding vector asynchronously using httpx.AsyncClient.

        For long texts exceeding max_tokens, splits into chunks,
        embeds each chunk, and returns the mean embedding.
        """
        if is_query and self._query_instruction:
            text = f"Instruct: {self._query_instruction}\nQuery: {text}"

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
        """Embed a single chunk of text asynchronously.

        Retries transient failures (connect/timeout/5xx) up to
        ``EMBEDDING_MAX_RETRIES`` times. Non-transient errors (4xx) surface
        immediately so the caller can fall back to a hash/zero vector.
        """
        url = config["api_base"].rstrip("/") + "/embeddings"
        headers = {"Content-Type": "application/json"}
        if config["api_key"]:
            headers["Authorization"] = f"Bearer {config['api_key']}"
        payload = {
            "model": config["model"],
            "input": chunk,
            "encoding_format": "float",
        }

        last_exc: Exception | None = None
        for attempt in range(EMBEDDING_MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=EMBEDDING_REQUEST_TIMEOUT,
                    )
                response.raise_for_status()
                data = response.json()
                emb = data["data"][0]["embedding"]
                return self._truncate_dim(emb, config.get("dim"))
            except httpx.HTTPStatusError as exc:
                # 4xx is not transient — don't retry.
                if exc.response.status_code < 500:
                    raise
                last_exc = exc
            except (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.RemoteProtocolError,
            ) as exc:
                last_exc = exc

        raise last_exc if last_exc else RuntimeError("embedding request failed")

    async def _compute_embs_batch_async(
        self,
        texts: list[str],
        batch_size: int = 50,
        max_workers: int = 10,
        is_query: bool = False,
    ) -> list[list[float]]:
        """Compute embeddings for multiple texts in parallel batches.

        Uses batch API for efficiency, with semaphore to limit concurrent requests.
        Reduced default max_workers from 50 to 10 to avoid overwhelming the embedding API.
        """
        if is_query and self._query_instruction:
            texts = [f"Instruct: {self._query_instruction}\nQuery: {t}" for t in texts]

        import asyncio
        from asyncio import Semaphore

        config = self._get_emb_config()
        url = config["api_base"].rstrip("/") + "/embeddings"
        headers = {"Content-Type": "application/json"}
        if config["api_key"]:
            headers["Authorization"] = f"Bearer {config['api_key']}"

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
                            url,
                            headers=headers,
                            json=payload,
                            timeout=EMBEDDING_REQUEST_TIMEOUT,
                        )
                    response.raise_for_status()
                    data = response.json()
                    return [
                        self._truncate_dim(item["embedding"], config.get("dim"))
                        for item in data["data"]
                    ]
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
        is_query: bool = False,
    ) -> list[list[float]]:
        """Compute embeddings for multiple texts using batch API calls.

        Groups texts into batches and sends each batch in a single API request
        to reduce HTTP overhead. Falls back to per-text _compute_emb (which
        handles chunking for long texts) when the batch API fails.
        """
        if is_query and self._query_instruction:
            texts = [f"Instruct: {self._query_instruction}\nQuery: {t}" for t in texts]

        config = self._get_emb_config()
        url = config["api_base"].rstrip("/") + "/embeddings"
        headers = {"Content-Type": "application/json"}
        if config["api_key"]:
            headers["Authorization"] = f"Bearer {config['api_key']}"

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                payload = {
                    "model": config["model"],
                    "input": batch,
                    "encoding_format": "float",
                }
                response = httpx.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=EMBEDDING_REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                all_embeddings.extend(
                    self._truncate_dim(item["embedding"], config.get("dim"))
                    for item in data["data"]
                )
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

    @staticmethod
    def _sanitize_content(procedure: Procedure) -> Procedure:
        """Strip NUL bytes from content so backend payload JSON accepts it.

        NUL (0x00) characters break JSON serialization. Some upstream corpora
        embed control bytes (e.g. in directory-tree blocks) that must be
        removed before storage and embedding so both paths see the same
        cleaned text.
        """
        if "\x00" in procedure.content:
            return replace(procedure, content=procedure.content.replace("\x00", ""))
        return procedure


class QdrantStore(VectorStore):
    """
    Qdrant vector database backed store for procedural memory.

    Qdrant vector DB implementation for procedural memory. Procedures are
    stored as points with embeddings for semantic search using cosine
    similarity.

    Embeddings are computed via OpenAI-compatible API with hash-based fallback
    (shared logic inherited from ``VectorStore``).

    Collection schema (payload fields map to Procedure attributes):
        id: point id (UUID string)
        user_id: keyword (filterable)
        title: text
        content: text
        category: text
        tags: keyword array
        kind: keyword (filterable)
        source_path: text
        metadata: json
        created_at: text
        updated_at: text
        emb: dense vector of size emb_dim

    Qdrant-specific environment variables:
        QDRANT_BASE_URL              — Qdrant server URL
        QDRANT_API_KEY               — Optional API key for secured clusters
        QDRANT_COLLECTION_NAME       — Collection name (default: procedures)
        QDRANT_INDEX_TYPE            — Index type: hnsw or flat (default: hnsw)
        QDRANT_DISTANCE              — Distance metric: Cosine, Dot, or Euclid (default: Cosine)
        QDRANT_INDEX_M               — HNSW max connections per layer (default: 16)
        QDRANT_INDEX_EF_CONSTRUCT    — HNSW search depth during build (default: 100)

    Embedding configuration (shared, read from VECTOR_* env):
        VECTOR_EMBEDDING_MODEL       — Embedding model
        VECTOR_EMBEDDING_API_BASE    — API base URL (required)
        VECTOR_EMBEDDING_API_KEY     — API key for embedding endpoint
        VECTOR_EMBEDDING_DIMENSIONS  — Embedding dimensions
        VECTOR_EMBEDDING_MAX_TOKENS  — Max tokens per chunk (optional, default 8192)
        VECTOR_EMBEDDING_QUERY_INSTRUCTION — Query instruction prefix (optional)

    Note:
        VECTOR_EMBEDDING_API_BASE must be set via environment variable or
        passed explicitly. No hardcoded default - use .env file or set
        VECTOR_EMBEDDING_API_BASE before instantiating.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        index_type: str | None = None,
        distance: str | None = None,
        hnsw_m: int | None = None,
        hnsw_ef_construct: int | None = None,
        emb_model: str | None = None,
        emb_api_base: str | None = None,
        emb_api_key: str | None = None,
        emb_dim: int | None = None,
        query_instruction: str | None = None,
        emb_max_tokens: int | None = None,
    ) -> None:
        # Load Qdrant-specific config from QDRANT_* env when not provided
        if base_url is None:
            base_url = os.getenv("QDRANT_BASE_URL", "http://localhost:6333")
        if api_key is None:
            api_key = os.getenv("QDRANT_API_KEY")
        api_key = api_key or None
        if collection_name is None:
            collection_name = os.getenv("QDRANT_COLLECTION_NAME", "procedures")
        if index_type is None:
            index_type = os.getenv("QDRANT_INDEX_TYPE", "hnsw")
        if distance is None:
            distance = os.getenv("QDRANT_DISTANCE", "Cosine")
        if hnsw_m is None:
            hnsw_m = int(os.getenv("QDRANT_INDEX_M", "16"))
        if hnsw_ef_construct is None:
            hnsw_ef_construct = int(os.getenv("QDRANT_INDEX_EF_CONSTRUCT", "100"))

        # Load embedding config from VECTOR_* env when not provided
        if emb_model is None:
            emb_model = os.getenv("VECTOR_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
        if emb_api_base is None:
            emb_api_base = os.getenv("VECTOR_EMBEDDING_API_BASE")
            if emb_api_base is None:
                raise ValueError(
                    "VECTOR_EMBEDDING_API_BASE must be set. "
                    "Add to .env file or set environment variable."
                )
        if emb_api_key is None:
            emb_api_key = os.getenv("VECTOR_EMBEDDING_API_KEY", "EMPTY")
        if emb_dim is None:
            emb_dim = int(os.getenv("VECTOR_EMBEDDING_DIMENSIONS", "2560"))
        if query_instruction is None:
            query_instruction = os.getenv("VECTOR_EMBEDDING_QUERY_INSTRUCTION", "")
        if emb_max_tokens is None:
            env_max = os.getenv("VECTOR_EMBEDDING_MAX_TOKENS")
            if env_max:
                try:
                    emb_max_tokens = int(env_max)
                except ValueError:
                    logger.warning(
                        "Invalid VECTOR_EMBEDDING_MAX_TOKENS=%r, using default",
                        env_max,
                    )

        # Hybrid search config. The collection's actual schema (named
        # dense+sparse vs single unnamed dense) is DETECTED from Qdrant in
        # _init_collection, never configured by the user — the only env var
        # read here is MEMFLOW_HYBRID_SEARCH_ENABLED, and only to decide the
        # schema for a brand-new collection (sparse included so hybrid can be
        # enabled later without re-seeding). Switching hybrid on/off against an
        # existing collection just follows whatever schema it already has.
        self._create_with_sparse = env_flag("MEMFLOW_HYBRID_SEARCH_ENABLED")

        # Initialize VectorStore with embedding config
        super().__init__(
            emb_model=emb_model,
            emb_api_base=emb_api_base,
            emb_api_key=emb_api_key,
            emb_dim=emb_dim,
            query_instruction=query_instruction,
            emb_max_tokens=emb_max_tokens,
        )

        # Qdrant-specific attributes
        self._base_url = base_url
        self._api_key = api_key
        self._collection_name = collection_name
        self._index_type = index_type
        self._distance = distance
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construct = hnsw_ef_construct
        # Schema flags — authoritative values set by _init_collection from the
        # actual collection (or its creation). Defaults avoid attribute gaps if
        # init fails partway.
        self._named_schema = False
        self._sparse_enabled = False

        self._client: Any = None
        self._lock = threading.Lock()
        self._init_collection()

    @property
    def supports_hybrid(self) -> bool:
        """True when the (auto-detected) collection schema has the sparse
        vector — i.e. hybrid/sparse search can actually be served. Callers
        should probe this instead of try/except'ing ``search_hybrid``."""
        return self._sparse_enabled

    # ------------------------------------------------------------------
    # Collection initialization
    # ------------------------------------------------------------------

    def _init_collection(self) -> None:
        """Initialize Qdrant client and collection."""
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required for QdrantStore. Install with: uv sync"
            ) from exc

        self._client = QdrantClient(url=self._base_url, api_key=self._api_key)

        try:
            from qdrant_client import models

            # Map distance string to Qdrant Distance enum
            distance_map = {
                "Cosine": models.Distance.COSINE,
                "Dot": models.Distance.DOT,
                "Euclid": models.Distance.EUCLID,
            }
            distance_enum = distance_map.get(self._distance, models.Distance.COSINE)

            # Create collection if it doesn't exist
            if not self._client.collection_exists(self._collection_name):
                if self._create_with_sparse:
                    # New collection, hybrid enabled — create with named
                    # vectors (dense + sparse BM25) from the start so hybrid
                    # can be turned on later without re-seeding. The sparse
                    # vector uses Modifier.IDF so Qdrant applies
                    # corpus-frequency weighting to the raw BM25 term
                    # frequencies emitted by the client-side fastembed model
                    # (self-hosted Qdrant cannot use the cloud Document
                    # inference API).
                    vectors_config = {
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=self._emb_dim, distance=distance_enum
                        ),
                    }
                    sparse_vectors_config = {
                        SPARSE_VECTOR_NAME: models.SparseVectorParams(
                            index=models.SparseIndexParams(),
                            modifier=models.Modifier.IDF,
                        ),
                    }
                    self._named_schema = True
                    self._sparse_enabled = True
                else:
                    vectors_config = models.VectorParams(
                        size=self._emb_dim, distance=distance_enum
                    )
                    sparse_vectors_config = None
                    self._named_schema = False
                    self._sparse_enabled = False

                if self._index_type == "hnsw":
                    hnsw_config = models.HnswConfigDiff(
                        m=self._hnsw_m, ef_construct=self._hnsw_ef_construct
                    )
                    self._client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=vectors_config,
                        hnsw_config=hnsw_config,
                        sparse_vectors_config=sparse_vectors_config,
                    )
                else:
                    # flat index — Qdrant default
                    self._client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=vectors_config,
                        sparse_vectors_config=sparse_vectors_config,
                    )
            else:
                # Collection exists — adopt its actual schema regardless of
                # env config. upsert/search then use the same vector layout
                # the collection was created with, so a schema/env mismatch
                # can never corrupt or crash them. Hybrid (mode != "dense")
                # additionally requires the sparse vector; when missing it is
                # disabled here (supports_hybrid stays False) and callers
                # route to dense instead of probing an exception.
                info = self._client.get_collection(self._collection_name)
                # For a single unnamed vector Qdrant returns VectorParams
                # (a model, not a dict); named collections return a dict of
                # name -> VectorParams. Only the dict form is a named schema.
                vectors_config = info.config.params.vectors
                self._named_schema = (
                    isinstance(vectors_config, dict)
                    and DENSE_VECTOR_NAME in vectors_config
                )
                sparse_names = set((info.config.params.sparse_vectors or {}).keys())
                self._sparse_enabled = SPARSE_VECTOR_NAME in sparse_names
                if self._sparse_enabled and not self._named_schema:
                    logger.warning(
                        "Collection %r has a sparse vector but no named %r "
                        "dense vector — unexpected schema; hybrid will be "
                        "skipped.",
                        self._collection_name,
                        DENSE_VECTOR_NAME,
                    )
                    self._sparse_enabled = False

            # Create payload field indexes for efficient filtering
            try:
                self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name="user_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # Index may already exist
            try:
                self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name="kind",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # Index may already exist

        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize Qdrant collection: {exc}"
            ) from exc

    def _to_text(self, procedure: Procedure) -> str:
        """Convert procedure to text for embedding."""
        return procedure_search_text(procedure)

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def _point_to_procedure(self, point: Any) -> Procedure:
        """Convert a Qdrant point to a Procedure object."""
        payload = point.payload or {}

        tags = payload.get("tags", [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []
        if not isinstance(tags, list):
            tags = []

        metadata = payload.get("metadata", {})
        if isinstance(metadata, str):
            metadata = _metadata_json(metadata)
        if not isinstance(metadata, dict):
            metadata = {}

        return Procedure(
            id=payload.get("id", str(point.id)),
            user_id=payload.get("user_id", "default"),
            title=payload.get("title", ""),
            content=payload.get("content", ""),
            category=payload.get("category", "general"),
            tags=tags or [],
            kind=payload.get("kind", "skill"),
            source_path=payload.get("source_path"),
            metadata=metadata,
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", payload.get("created_at", "")),
        )

    def _upsert_point(
        self, procedure: Procedure, emb: list[float], sparse_vec: Any = None
    ) -> None:
        """Upsert a procedure as a Qdrant point with pre-computed embedding.

        The vector layout follows the collection's detected schema: named
        collections always receive the named ``dense`` vector (plus ``sparse``
        when a ``sparse_vec`` was computed); unnamed collections receive the
        bare dense vector.
        """
        from qdrant_client import models

        payload = {
            "id": procedure.id,
            "user_id": procedure.user_id,
            "title": procedure.title,
            "content": procedure.content,
            "category": procedure.category,
            "tags": procedure.tags,
            "kind": procedure.kind,
            "source_path": procedure.source_path,
            "metadata": procedure.metadata,
            "created_at": procedure.created_at,
            "updated_at": procedure.updated_at,
        }

        if self._named_schema:
            vector = {DENSE_VECTOR_NAME: emb}
            if sparse_vec is not None:
                vector[SPARSE_VECTOR_NAME] = sparse_vec
        else:
            vector = emb

        point = models.PointStruct(
            id=_id_to_uuid(procedure.id), vector=vector, payload=payload
        )

        self._client.upsert(collection_name=self._collection_name, points=[point])

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
            procedure = [self._sanitize_content(proc) for proc in procedure]
            texts = [self._to_text(proc) for proc in procedure]
            embeddings = self._compute_embs_batch(texts, batch_size=batch_size)
            sparse_vecs = (
                self._compute_sparse_batch(texts) if self._sparse_enabled else None
            )
            num_inserted = 0
            for i, (proc, emb) in enumerate(zip(procedure, embeddings)):
                try:
                    sv = sparse_vecs[i] if sparse_vecs is not None else None
                    self._upsert_point(proc, emb, sparse_vec=sv)
                    num_inserted += 1
                except Exception:
                    pass
            return num_inserted
        else:
            procedure = self._sanitize_content(procedure)
            text_content = self._to_text(procedure)
            emb = self._compute_emb(text_content)
            sparse_vec = (
                self._compute_sparse(text_content) if self._sparse_enabled else None
            )
            self._upsert_point(procedure, emb, sparse_vec=sparse_vec)
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
            procedure = [self._sanitize_content(proc) for proc in procedure]
            texts = [self._to_text(proc) for proc in procedure]
            embeddings = await self._compute_embs_batch_async(
                texts, batch_size, max_workers
            )
            # Sparse BM25 is computed locally (CPU-bound) — offload to a thread
            # to avoid blocking the event loop. One batched call per upsert.
            if self._sparse_enabled:
                sparse_vecs = await asyncio.to_thread(self._compute_sparse_batch, texts)
            else:
                sparse_vecs = None
            semaphore = Semaphore(max_workers)

            async def insert_single(proc: Procedure, emb: list[float], sv: Any) -> int:
                async with semaphore:
                    try:
                        await asyncio.to_thread(self._upsert_point, proc, emb, sv)
                        return 1
                    except Exception:
                        return 0

            tasks = [
                insert_single(proc, emb, sparse_vecs[i] if sparse_vecs else None)
                for i, (proc, emb) in enumerate(zip(procedure, embeddings))
            ]
            results = await asyncio.gather(*tasks)
            return sum(results)
        else:
            procedure = self._sanitize_content(procedure)
            text_content = self._to_text(procedure)
            emb = await self._compute_emb_async(text_content)
            if self._sparse_enabled:
                sparse_vec = await asyncio.to_thread(self._compute_sparse, text_content)
            else:
                sparse_vec = None
            self._upsert_point(procedure, emb, sparse_vec=sparse_vec)
            return 1

    def _search_with_emb(
        self,
        query_emb: list[float],
        top_k: int,
        user_id: str | None = None,
        kind: str | None = "skill",
    ) -> list[SearchResult]:
        """Search using pre-computed query embedding."""
        query_filter = self._build_query_filter(user_id, kind)

        # Hybrid collections store the dense vector under a named slot ("dense");
        # dense-only collections use the unnamed default. Pass `using=` only for
        # named-vector collections so Qdrant resolves the right vector.
        dense_using = DENSE_VECTOR_NAME if self._named_schema else None

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_emb,
            using=dense_using,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return [
            SearchResult(procedure=self._point_to_procedure(p), score=float(p.score))
            for p in response.points
        ]

    # ------------------------------------------------------------------
    # Hybrid search (dense + sparse RRF fusion via query_points)
    # ------------------------------------------------------------------

    def _build_query_filter(self, user_id: str | None, kind: str | None) -> Any:
        from qdrant_client import models

        must = []
        if user_id:
            must.append(
                models.FieldCondition(
                    key="user_id", match=models.MatchValue(value=user_id)
                )
            )
        if kind is not None:
            must.append(
                models.FieldCondition(key="kind", match=models.MatchValue(value=kind))
            )
        return models.Filter(must=must) if must else None

    def search_hybrid(
        self,
        query: str,
        rrf_top_k: int = 10,
        user_id: str | None = None,
        kind: str | None = "skill",
        mode: str = "hybrid",
        rrf_weights: list[float] | None = None,
        sparse_top_k: int = 200,
        dense_top_k: int = 200,
        hnsw_ef: int | None = None,
        rrf_k: int = 60,
        dense_emb: list[float] | None = None,
        sparse_vec: Any = None,
    ) -> list[SearchResult]:
        """Hybrid sparse+dense retrieval with weighted RRF fusion.

        Single ``query_points`` call: dense + sparse prefetches are fused
        server-side via ``RrfQuery(Rrf(weights=[dense_w, sparse_w], k))``. When
        ``mode`` is ``"dense"`` or ``"sparse"`` that channel's vector is
        queried directly — no prefetch, no fusion (used for isolated Recall@K
        measurement in experiment 4.1).

        Requires a collection whose (auto-detected) schema includes the named
        sparse vector — probe ``supports_hybrid`` before calling. For
        ``mode="dense"`` any collection works — legacy unnamed vectors are
        queried by position rather than name.

        The three Top-K parameters in the hybrid-search pipeline:

        - ``sparse_top_k`` — SPARSE_SEARCH_TOP_K: sparse (BM25) prefetch depth.
        - ``dense_top_k`` — DENSE_SEARCH_TOP_K: dense (embedding) prefetch depth.
        - ``rrf_top_k`` — RRF_TOP_K: number of candidates the RRF fusion returns.

        Args:
            query: query string
            rrf_top_k: RRF_TOP_K — number of fused candidates to return. In
                dense/sparse-only mode this is the single-channel result count.
            mode: ``"hybrid"`` | ``"dense"`` | ``"sparse"``
            rrf_weights: ``(dense_weight, sparse_weight)`` per-prefetch weights
                for the RRF fusion (Qdrant v1.17+ extension; multiplies each
                channel's ``1/(k+rank)`` score). ``None`` → equal ``(0.5, 0.5)``.
            sparse_top_k: SPARSE_SEARCH_TOP_K — sparse prefetch limit.
            dense_top_k: DENSE_SEARCH_TOP_K — dense prefetch limit.
            hnsw_ef: optional HNSW ``ef`` search param (higher = more thorough
                graph traversal).
            rrf_k: RRF constant ``k`` in ``score = 1/(k + rank)`` (default 60 —
                the standard IR value; lower k rewards top ranks more
                aggressively).
            dense_emb: pre-computed dense query embedding. When ``None`` the
                embedding is computed on the fly (one API call). Experiment
                harnesses pass a pre-computed vector to amortize the embedding
                API cost across many configs over the same query.
            sparse_vec: pre-computed sparse query vector (same purpose as
                ``dense_emb``).

        Returns:
            List of ``SearchResult`` ranked by fused score. In hybrid mode the
            RRF scores are rescaled to 0~1 relative to the top hit (top hit =
            1.0) so cosine-calibrated threshold consumers don't filter
            everything out; the scores are top-relative, not absolute
            relevance, and not comparable across queries. Operational
            consequence: an absolute-similarity gate (e.g. the hook's
            ``min_score``) is effectively disabled on the fused path — the top
            hit is 1.0 by construction, so even a query unrelated to every
            skill still yields at least one candidate, and RRF's slow
            rank decay lets deep results through too (with the default
            weights/k, roughly anything within the prefetch depth passes a
            0.2 gate). Exception: a token-less query (no BM25 terms) runs the
            dense leg alone and keeps raw cosine scores.
        """
        if not self._sparse_enabled and mode != "dense":
            raise RuntimeError(
                f"search_hybrid(mode={mode!r}) requires a collection with a "
                f"sparse vector; collection {self._collection_name!r} was "
                f"created without one. Recreate the collection with "
                f"MEMFLOW_HYBRID_SEARCH_ENABLED=on (the existing one keeps "
                f"its schema regardless of the switch) or use mode='dense'. "
                f"(Probe store.supports_hybrid before calling.)"
            )

        from qdrant_client import models

        query_filter = self._build_query_filter(user_id, kind)
        params = models.SearchParams(hnsw_ef=hnsw_ef) if hnsw_ef is not None else None

        use_named = self._named_schema
        dense_using = DENSE_VECTOR_NAME if use_named else None

        # Single-channel modes query the vector directly (no fusion needed).
        # Hybrid mode builds a prefetch list and fuses via RrfQuery below.
        if mode == "dense":
            # Single channel — no fusion needed; query with the dense vector.
            if dense_emb is None:
                dense_emb = self._compute_emb(query, is_query=True)
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=dense_emb,
                using=dense_using,
                query_filter=query_filter,
                search_params=params,
                limit=rrf_top_k,
                with_payload=True,
                with_vectors=False,
            )
        elif mode == "sparse":
            if sparse_vec is None:
                sparse_vec = self._compute_sparse(query, is_query=True)
            if sparse_vec is None:
                # Token-less query — BM25 has nothing to match on.
                return []
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=sparse_vec,
                using=SPARSE_VECTOR_NAME,
                query_filter=query_filter,
                # hnsw_ef only tunes the dense HNSW traversal; the sparse
                # index ignores it (same reasoning as the sparse prefetch in
                # hybrid mode below).
                limit=rrf_top_k,
                with_payload=True,
                with_vectors=False,
            )
        elif mode == "hybrid":
            if dense_emb is None:
                dense_emb = self._compute_emb(query, is_query=True)
            if sparse_vec is None:
                sparse_vec = self._compute_sparse(query, is_query=True)
            if sparse_vec is None:
                # Token-less query (e.g. stopword-only): BM25 contributes
                # nothing and Qdrant rejects an empty sparse query, so run
                # the dense leg alone. Returns RAW cosine scores — deliberately
                # NOT passed through the RRF rescale below, which would turn a
                # weak top hit into a guaranteed 1.0. Consumers therefore see
                # cosine semantics on this path and top-relative semantics on
                # the fused path (see test_degraded_hybrid_keeps_raw_scores).
                response = self._client.query_points(
                    collection_name=self._collection_name,
                    query=dense_emb,
                    using=dense_using,
                    query_filter=query_filter,
                    search_params=params,
                    limit=rrf_top_k,
                    with_payload=True,
                    with_vectors=False,
                )
                return [
                    SearchResult(
                        procedure=self._point_to_procedure(p), score=float(p.score)
                    )
                    for p in response.points
                ]

            # Single dense vector + sparse (2-prefetch RRF fusion).
            if rrf_weights is None:
                weights = [0.5, 0.5]
            else:
                weights = [float(w) for w in rrf_weights]
                if len(weights) != 2:
                    raise ValueError(
                        f"rrf_weights must have 2 entries (dense, sparse), "
                        f"got {len(weights)}"
                    )
            prefetch = [
                models.Prefetch(
                    query=dense_emb,
                    using=dense_using,
                    filter=query_filter,
                    params=params,
                    limit=dense_top_k,
                ),
                models.Prefetch(
                    query=sparse_vec,
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    # hnsw_ef only tunes the dense HNSW traversal; the
                    # sparse index ignores it, so don't pass it here.
                    limit=sparse_top_k,
                ),
            ]
            rrf = models.RrfQuery(rrf=models.Rrf(weights=weights, k=rrf_k))
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=rrf,
                prefetch=prefetch,
                query_filter=query_filter,
                limit=rrf_top_k,
                with_payload=True,
                with_vectors=False,
            )
        else:
            raise ValueError(
                f"Unknown search mode: {mode!r}. "
                f"Expected one of: 'dense', 'sparse', 'hybrid'."
            )

        scores = [float(p.score) for p in response.points]
        if mode == "hybrid" and scores:
            # Qdrant's weighted RRF score is a rank-sum (sum of w/(k+rank)),
            # topping out around 0.016 — raw scores would be filtered out
            # entirely by cosine-calibrated thresholds (e.g.
            # SkillContextSelector's min_score=0.2). Rescale so the top hit
            # scores 1.0 and the rest fall proportionally. NOTE: this makes
            # scores *relative to the top hit of this query*, not absolute
            # relevance — score thresholds under hybrid mean "fraction of
            # the best hit", and scores are not comparable across queries.
            # Accepted tradeoff: RRF discards absolute similarity, so min_score
            # is no longer an absolute-similarity gate here. The top hit passes
            # by construction (irrelevant queries still inject ≥1 candidate)
            # and the slow 1/(k+rank) decay means the gate filters almost
            # nothing within the prefetch depth. Mitigation if this becomes a
            # problem in production: issue a parallel dense query (the query
            # embedding is already computed) and gate the fused hits on their
            # raw cosine.
            top = max(scores)
            if top > 0:
                scores = [s / top for s in scores]

        return [
            SearchResult(procedure=self._point_to_procedure(p), score=score)
            for p, score in zip(response.points, scores)
        ]

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
            batch_size: Batch size for embedding API calls (default: 10) - QdrantStore only

        Returns:
            Single list for single query, list of lists for batch
        """
        if isinstance(query, list):
            query_embs = self._compute_embs_batch(
                query, batch_size=batch_size, is_query=True
            )
            results = []
            for query_emb in query_embs:
                search_results = self._search_with_emb(query_emb, top_k, user_id, kind)
                results.append(search_results)
            return results
        else:
            query_emb = self._compute_emb(query, is_query=True)
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
            batch_size: Batch size for embedding API calls (default: 10) - QdrantStore only
            max_workers: Max concurrent requests (default: 10)

        Returns:
            Single list for single query, list of lists for batch
        """
        import asyncio
        from asyncio import Semaphore

        if isinstance(query, list):
            query_embs = await self._compute_embs_batch_async(
                query, batch_size=batch_size, max_workers=max_workers, is_query=True
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
            query_emb = await self._compute_emb_async(query, is_query=True)
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
        try:
            points = self._client.retrieve(
                collection_name=self._collection_name,
                ids=[_id_to_uuid(id)],
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return None

        if not points:
            return None

        return self._point_to_procedure(points[0])

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
                from qdrant_client import models

                # Check existence first to return accurate count
                existing = self.get(id)
                if existing is None:
                    return 0
                self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=models.PointIdsList(points=[_id_to_uuid(id)]),
                )
                return 1
            except Exception:
                return 0

    def list(self, user_id: str | None = None) -> list[Procedure]:
        """List all procedures, optionally filtered by user_id."""
        from qdrant_client import models

        must = []
        if user_id:
            must.append(
                models.FieldCondition(
                    key="user_id", match=models.MatchValue(value=user_id)
                )
            )
        query_filter = models.Filter(must=must) if must else None

        all_points = []
        offset = None
        limit = 256

        while True:
            results, next_offset = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=query_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(results)
            if next_offset is None:
                break
            offset = next_offset

        return [self._point_to_procedure(p) for p in all_points]
