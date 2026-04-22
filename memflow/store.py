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
import os
import threading
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

from memflow.models import Procedure, SearchResult

from sqlalchemy import create_engine, text

# Register pgvector for proper vector type handling
import psycopg2
from pgvector.psycopg2 import register_vector


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
        CREATE TABLE IF NOT EXISTS procedures (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL DEFAULT 'default',
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            tags JSONB NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            emb vector(2560)
        );
        CREATE INDEX IF NOT EXISTS idx_procedures_emb
            ON procedures USING ivfflat (emb vector_cosine_ops);  -- only if dim <= 2000

    Environment variables:
        PGVECTOR_BASE_URL              — PostgreSQL URL
        PGVECTOR_EMBEDDING_MODEL       — Embedding model
        PGVECTOR_EMBEDDING_API_BASE    — API base URL (required)
        PGVECTOR_EMBEDDING_API_KEY     — API key
        PGVECTOR_EMBEDDING_DIMENSIONS  — Embedding dimensions

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
    ) -> None:
        # Load from environment if not provided
        if base_url is None:
            base_url = os.getenv("PGVECTOR_BASE_URL", "postgresql://pgvector:pgvector_password@localhost:5433/pgvector")
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
        self._base_url = base_url
        self._emb_model = emb_model
        self._emb_api_base = emb_api_base
        self._emb_api_key = emb_api_key
        self._emb_dim = emb_dim

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
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS procedures (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL DEFAULT 'default',
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        category TEXT NOT NULL DEFAULT 'general',
                        tags JSONB NOT NULL DEFAULT '[]'::jsonb,
                        created_at TEXT NOT NULL,
                        emb vector({self._emb_dim})
                    )
                """))

                # Add emb column if table exists but column is missing (migration)
                conn.execute(text(f"""
                    ALTER TABLE procedures
                    ADD COLUMN IF NOT EXISTS emb vector({self._emb_dim})
                """))

                # Create index for semantic search
                # ivfflat: up to 2000 dimensions
                # hnsw: also limited to 2000 dimensions in current pgvector
                if self._emb_dim <= 2000:
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_procedures_emb
                        ON procedures USING ivfflat (emb vector_cosine_ops)
                    """))
                # Skip index creation for dimensions > 2000 (sequential scan will be used)
                conn.commit()

            self._engine = engine
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PgVectorStore database: {e}") from e

    def _get_emb_config(self) -> dict:
        """Get embedding API configuration."""
        return {
            "api_base": self._emb_api_base,
            "api_key": self._emb_api_key,
            "model": self._emb_model,
            "dim": self._emb_dim,
        }

    def _compute_emb(self, text: str) -> list[float]:
        """Compute embedding vector using OpenAI-compatible API."""
        config = self._get_emb_config()

        try:
            url = config["api_base"].rstrip("/") + "/embeddings"
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": config["model"],
                "input": text,
                "encoding_format": "float",
            }

            response = httpx.post(url, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            emb = data["data"][0]["embedding"]
            return emb
        except Exception:
            # Fallback: simple hash-based embedding
            emb = [0.0] * config["dim"]
            words = text.lower().split()
            for word in words:
                word_hash = hashlib.md5(word.encode()).hexdigest()
                for i in range(min(len(word_hash), config["dim"])):
                    val = (int(word_hash[i % len(word_hash)], 16) - 8) / 8.0
                    emb[i] += val / len(words)
            norm = sum(x * x for x in emb) ** 0.5
            if norm > 0:
                emb = [x / norm for x in emb]
            return emb

    def _to_text(self, procedure: Procedure) -> str:
        """Convert procedure to text for embedding."""
        return f"# {procedure.title}\n\n{procedure.content}"

    def add(self, procedure: Procedure) -> None:
        """Add a procedure to the store."""

        text_content = self._to_text(procedure)
        emb = self._compute_emb(text_content)

        # Convert embedding to PostgreSQL vector literal format: "[0.1,0.2,...]"
        emb_str = "[" + ",".join(str(v) for v in emb) + "]"

        with self._engine.connect() as conn:
            # Use CAST for the vector type - the ::vector syntax doesn't work with parameters
            conn.execute(text("""
                INSERT INTO procedures (id, user_id, title, content, category, tags, created_at, emb)
                VALUES (:id, :user_id, :title, :content, :category, :tags, :created_at, CAST(:emb AS vector))
                ON CONFLICT (id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    category = EXCLUDED.category,
                    tags = EXCLUDED.tags,
                    created_at = EXCLUDED.created_at,
                    emb = EXCLUDED.emb
            """), {
                "id": procedure.id,
                "user_id": procedure.user_id,
                "title": procedure.title,
                "content": procedure.content,
                "category": procedure.category,
                "tags": json.dumps(procedure.tags),
                "created_at": procedure.created_at,
                "emb": emb_str,
            })
            conn.commit()

    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[SearchResult]:
        """Search for procedures by semantic similarity."""
        query_emb = self._compute_emb(query)
        # Convert embedding to PostgreSQL vector literal format: "[0.1,0.2,...]"
        emb_str = "[" + ",".join(str(v) for v in query_emb) + "]"

        with self._engine.connect() as conn:
            if user_id:
                filter_clause = "AND user_id = :user_id"
                params = {"emb": emb_str, "user_id": user_id, "limit": top_k}
            else:
                filter_clause = ""
                params = {"emb": emb_str, "limit": top_k}

            # Use CAST for the vector type
            query_sql = text(f"""
                SELECT id, user_id, title, content, category, tags, created_at,
                       1 - (emb <=> CAST(:emb AS vector)) AS score
                FROM procedures
                WHERE TRUE {filter_clause}
                ORDER BY emb <=> CAST(:emb AS vector)
                LIMIT :limit
            """)

            result = conn.execute(query_sql, params)
            rows = result.fetchall()

        results = []
        for row in rows:
            try:
                tags = json.loads(row.tags) if isinstance(row.tags, str) else row.tags
            except Exception:
                tags = []

            proc = Procedure(
                id=row.id,
                user_id=row.user_id,
                title=row.title,
                content=row.content,
                category=row.category,
                tags=tags,
                created_at=row.created_at,
            )
            results.append(SearchResult(procedure=proc, score=float(row.score)))

        return results

    def get(self, id: str) -> Procedure | None:
        """Get a procedure by ID."""
        with self._engine.connect() as conn:
            result = conn.execute(text("""
                SELECT id, user_id, title, content, category, tags, created_at
                FROM procedures WHERE id = :id
            """), {"id": id})
            row = result.fetchone()

        if row is None:
            return None

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
            tags=tags,
            created_at=row.created_at,
        )

    def delete(self, id: str) -> bool:
        """Delete a procedure by ID."""
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text("""
                    DELETE FROM procedures WHERE id = :id
                """), {"id": id})
                conn.commit()
                return result.rowcount > 0
        except Exception:
            return False

    def list_all(self, user_id: str | None = None) -> list[Procedure]:
        """List all procedures."""
        with self._engine.connect() as conn:
            if user_id:
                result = conn.execute(text("""
                    SELECT id, user_id, title, content, category, tags, created_at
                    FROM procedures WHERE user_id = :user_id
                """), {"user_id": user_id})
            else:
                result = conn.execute(text("""
                    SELECT id, user_id, title, content, category, tags, created_at
                    FROM procedures
                """))
            rows = result.fetchall()

        procs = []
        for row in rows:
            try:
                tags = json.loads(row.tags) if isinstance(row.tags, str) else row.tags
            except Exception:
                tags = []

            procs.append(Procedure(
                id=row.id,
                user_id=row.user_id,
                title=row.title,
                content=row.content,
                category=row.category,
                tags=tags,
                created_at=row.created_at,
            ))
        return procs
