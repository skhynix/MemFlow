# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qdrant skill storage, retrieval, and SKILL.md synchronization."""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from memflow.models import SearchResult, SkillRecord
from memflow.skills import load_skill
from memflow.store import (
    BaseStore,
    QdrantStore,
    _validate_skill_input,
    _validate_skill_kind,
    env_flag,
)

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    """Read an int env var, warning and falling back to ``default`` on bad input.

    Mirrors ``_parse_rrf_weights``'s leniency: a malformed value should degrade
    to the built-in default rather than crash ``MemFlow()`` construction.
    ``minimum`` rejects nonsensical values (e.g. a non-positive Top-K that
    would only fail later as a server-side 400).
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, falling back to %d", name, raw, default)
        return default
    if minimum is not None and value < minimum:
        logger.warning(
            "Invalid %s=%r (must be >= %d), falling back to %d",
            name,
            raw,
            minimum,
            default,
        )
        return default
    return value


def _skill_api_payload(record: SkillRecord, event: str, **extra: object) -> dict:
    skill = record.metadata.get("skill", {})
    if not isinstance(skill, dict):
        skill = {}
    governance = record.metadata.get("governance", {})
    if not isinstance(governance, dict):
        governance = {}
    warnings = governance.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = [str(warnings)] if warnings else []

    payload = {
        "id": record.id,
        "title": record.title,
        "kind": record.kind,
        "sha256": skill.get("sha256"),
        "governance": {
            "trust_state": governance.get("trust_state"),
            "mode": governance.get("mode"),
            "warnings": warnings,
        },
        "source_path": record.source_path,
        "root_path": skill.get("root_path"),
        "event": event,
    }
    payload.update(extra)
    return payload


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mark_stale_skill(
    record: SkillRecord,
    warning: str,
) -> tuple[SkillRecord, bool]:
    metadata = (
        copy.deepcopy(record.metadata) if isinstance(record.metadata, dict) else {}
    )
    governance = metadata.setdefault("governance", {})
    if not isinstance(governance, dict):
        governance = {}
        metadata["governance"] = governance

    warnings = governance.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    changed = warning not in warnings
    if changed:
        warnings.append(warning)
    governance["warnings"] = warnings

    skill = metadata.setdefault("skill", {})
    if isinstance(skill, dict) and skill.get("stale") is not True:
        skill["stale"] = True
        changed = True

    if not changed:
        return record, False
    return replace(record, metadata=metadata), True


def _skill_source_candidate(path: Path) -> str:
    if path.name == "SKILL.md":
        return str(path.resolve(strict=False))
    return str((path / "SKILL.md").resolve(strict=False))


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


class MemFlow:
    """Manage skills with an explicit store or environment-configured Qdrant."""

    def __init__(self, *, store: BaseStore | None = None, use_env: bool = True) -> None:
        if use_env:
            _load_env_file()
        if store is None:
            if not use_env:
                raise ValueError("An explicit store is required when use_env=False")
            backend = os.getenv("MEMFLOW_BACKEND", "qdrant").strip().lower()
            if backend != "qdrant":
                raise ValueError(
                    f"unsupported backend {backend!r}; set MEMFLOW_BACKEND to qdrant"
                )
            store = QdrantStore()
        self.store = store
        # Hybrid search config (read from env).
        # Top-K parameters govern the pipeline depth:
        #   HYBRID_SPARSE_SEARCH_TOP_K / HYBRID_DENSE_SEARCH_TOP_K — Qdrant prefetch limits
        #   HYBRID_RRF_TOP_K — RRF fusion output. 100 is plenty online: the
        #   fusion ranking is limit-independent, so this only bounds how many
        #   results (payloads) come back before truncation to top_k.
        self._hybrid_search_enabled = env_flag("MEMFLOW_HYBRID_SEARCH_ENABLED")
        self._hybrid_sparse_search_top_k = _env_int(
            "MEMFLOW_HYBRID_SPARSE_SEARCH_TOP_K", 200, minimum=1
        )
        self._hybrid_dense_search_top_k = _env_int(
            "MEMFLOW_HYBRID_DENSE_SEARCH_TOP_K", 200, minimum=1
        )
        self._hybrid_rrf_top_k = _env_int("MEMFLOW_HYBRID_RRF_TOP_K", 100, minimum=1)
        self._hybrid_rrf_weights = os.getenv("MEMFLOW_HYBRID_RRF_WEIGHTS", "0.6,0.4")
        self._hybrid_rrf_k = _env_int("MEMFLOW_HYBRID_RRF_K", 60, minimum=1)
        # Parsed once; falls back with a warning on malformed input.
        self._hybrid_rrf_weights_parsed = self._parse_rrf_weights()
        # Reranker config (search-method agnostic; reranks candidates from any path).
        # Order mirrors .env / .env.example: PIPELINE → FETCH_TOP_K → SCOPE → DOC_CAP
        # (SCOPE is optional — code defaults per stage; DOC_CAP and TIMEOUT live
        # in reranker.py, read lazily) → CE_TOP_K → LLM_TOP_K.
        self._reranker_pipeline = os.getenv("MEMFLOW_RERANKER_PIPELINE", "none")
        self._reranker_fetch_top_k = _env_int(
            "MEMFLOW_RERANKER_FETCH_TOP_K", 100, minimum=1
        )
        # "" = no global override; each rerank stage falls back to its own
        # default scope (CE=full, LLM=meta — the per-stage SkillRet optima).
        self._reranker_scope = os.getenv("MEMFLOW_RERANKER_SCOPE", "")
        # 0 is valid and means "no stage cap" (passed through as None below).
        self._reranker_ce_top_k = _env_int("MEMFLOW_RERANKER_CE_TOP_K", 10, minimum=0)
        self._reranker_llm_top_k = _env_int("MEMFLOW_RERANKER_LLM_TOP_K", 10, minimum=0)
        self._rerank_pipeline_obj: Any = None
        # Guards lazy construction in _get_rerank_pipeline (search is called
        # from multiple threads under concurrent serving/benchmark load).
        self._rerank_pipeline_lock = threading.Lock()
        # Warn once per instance when hybrid degrades to dense (the store's
        # capability is fixed at init — per-query warnings would just spam).
        self._hybrid_fallback_warned = False
        # Warn once when FETCH_TOP_K shrinks the hybrid candidate pool below
        # RRF_TOP_K (config fixed at init — per-query warnings would spam).
        self._fetch_cap_warned = False

    def add(self, skill: SkillRecord | list[SkillRecord]) -> dict:
        """Store skill records, returning event information or batch counts."""
        _validate_skill_input(skill)
        if isinstance(skill, list):
            total = len(skill)
            num_seeded = self.store.add(skill) if skill else 0
            return {
                "num_seeded": num_seeded,
                "num_skipped": total - num_seeded,
                "total": total,
            }
        self.store.add(skill)
        return {"id": skill.id, "title": skill.title, "event": "ADD"}

    def add_skill(
        self,
        path: str | Path,
        user_id: str = "default",
        source: str = "local",
        trust_state: str | None = None,
    ) -> dict:
        """Index a skill directory or direct SKILL.md path."""
        record = load_skill(
            path, user_id=user_id, source=source, trust_state=trust_state
        )
        self.store.add(record)
        return _skill_api_payload(record, "ADD_SKILL")

    def sync_skill(self, path_or_id: str | Path) -> dict:
        """Refresh a skill record from its canonical SKILL.md source."""
        existing = self.store.get(str(path_or_id))
        if existing is not None and existing.kind != "skill":
            raise ValueError("sync_skill only supports skill records")
        path = Path(path_or_id).expanduser()
        if existing is None:
            resolved = _skill_source_candidate(path)
            for record in self.store.list():
                if record.kind == "skill" and record.source_path == resolved:
                    existing = record
                    break
        if existing is not None:
            if not existing.source_path:
                warning = (
                    f"Skill source path is missing for stored skill {existing.id}."
                )
                stale_record, metadata_updated = _mark_stale_skill(existing, warning)
                if metadata_updated:
                    self.store.add(stale_record)
                return _skill_api_payload(
                    stale_record,
                    "STALE_SYNC_SKILL",
                    status="stale",
                    warning=warning,
                    metadata_updated=metadata_updated,
                )
            path = Path(existing.source_path)
            if not path.exists():
                warning = f"SKILL.md source file is missing: {path}"
                stale_record, metadata_updated = _mark_stale_skill(existing, warning)
                if metadata_updated:
                    self.store.add(stale_record)
                return _skill_api_payload(
                    stale_record,
                    "STALE_SYNC_SKILL",
                    status="stale",
                    warning=warning,
                    metadata_updated=metadata_updated,
                )

            stored_skill = existing.metadata.get("skill", {})
            stored_sha256 = (
                stored_skill.get("sha256") if isinstance(stored_skill, dict) else None
            )
            current_sha256 = _sha256_path(path)
            is_stale = (
                isinstance(stored_skill, dict) and stored_skill.get("stale") is True
            )
            if stored_sha256 == current_sha256 and not is_stale:
                return _skill_api_payload(
                    existing,
                    "NOOP_SYNC_SKILL",
                    status="noop",
                    reason="sha256_unchanged",
                )

            governance = existing.metadata.get("governance", {})
            if not isinstance(governance, dict):
                governance = {}
            source = str(governance.get("source") or "local")
            trust_state = governance.get("trust_state")
            record = load_skill(
                path,
                user_id=existing.user_id,
                source=source,
                trust_state=trust_state,
            )
            record.created_at = existing.created_at
        else:
            record = load_skill(path)

        self.store.add(record)
        return _skill_api_payload(record, "SYNC_SKILL")

    def list_skills(
        self,
        user_id: str | None = None,
        trust_state: str | None = None,
    ) -> list[SkillRecord]:
        """List indexed skills, optionally filtered by trust state."""
        skills = [p for p in self.store.list() if p.kind == "skill"]
        if user_id:
            skills = [p for p in skills if p.user_id == user_id]
        if trust_state is None:
            return skills
        return [
            record
            for record in skills
            if record.metadata.get("governance", {}).get("trust_state") == trust_state
        ]

    def search_skills(
        self,
        query: str,
        user_id: str | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search indexed skills."""
        return self.search(query, user_id=user_id, top_k=top_k, kind="skill")

    def get_skill(
        self,
        id_or_name: str,
        include_content: bool = True,
    ) -> SkillRecord | None:
        """Return a skill by id, title, or frontmatter name."""
        for record in self.store.list():
            if record.kind != "skill":
                continue
            skill = record.metadata.get("skill", {})
            if id_or_name in (record.id, record.title, skill.get("name")):
                return record if include_content else replace(record, content="")
        return None

    def _parse_rrf_weights(self) -> tuple[float, float]:
        """Parse the MEMFLOW_HYBRID_RRF_WEIGHTS env value into a (dense, sparse) tuple."""
        raw = self._hybrid_rrf_weights
        try:
            parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            logger.warning(
                "Invalid MEMFLOW_HYBRID_RRF_WEIGHTS=%r, falling back to (0.6, 0.4)",
                raw,
            )
            return (0.6, 0.4)
        if len(parts) != 2:
            logger.warning(
                "MEMFLOW_HYBRID_RRF_WEIGHTS=%r needs 2 values (dense, sparse), "
                "got %d; falling back to (0.6, 0.4)",
                raw,
                len(parts),
            )
            return (0.6, 0.4)
        if any(w < 0 for w in parts):
            logger.warning(
                "MEMFLOW_HYBRID_RRF_WEIGHTS=%r must be non-negative, "
                "falling back to (0.6, 0.4)",
                raw,
            )
            return (0.6, 0.4)
        return (parts[0], parts[1])

    def _search_hybrid(
        self,
        query: str,
        user_id: str | None,
        kind: str,
        rrf_top_k: int,
    ) -> list[SearchResult]:
        """Hybrid retrieval (sparse+dense RRF) for a single query.

        ``rrf_top_k`` is the fusion depth requested by the caller — passed
        through to Qdrant as-is so the caller's cap (e.g. the reranker
        FETCH_TOP_K bound computed in ``search``) actually reduces the
        server-side limit instead of being re-inflated here.

        The store must be hybrid-capable (``store.supports_hybrid``); routing
        is decided by ``search`` before calling, so errors from
        ``search_hybrid`` here are real failures and propagate.
        """
        return self.store.search_hybrid(
            query=query,
            rrf_top_k=rrf_top_k,
            user_id=user_id,
            kind=kind,
            mode="hybrid",
            rrf_weights=self._hybrid_rrf_weights_parsed,
            sparse_top_k=self._hybrid_sparse_search_top_k,
            dense_top_k=self._hybrid_dense_search_top_k,
            rrf_k=self._hybrid_rrf_k,
        )

    def _get_rerank_pipeline(self) -> Any:
        """Lazily build the RerankPipeline (stage clients are only
        constructed when reranking actually runs).

        A misconfigured reranker (e.g. PIPELINE=ce without
        RERANKER_CE_API_BASE) must not take down every search — runtime
        endpoint failures already degrade to retrieval order, so config
        errors degrade the same way: warn once and fall back to a
        truncate-only pipeline. The failed build is cached (both outcomes),
        and the double-checked lock keeps concurrent callers from
        constructing duplicate pipelines/clients.
        """
        if self._rerank_pipeline_obj is not None:
            return self._rerank_pipeline_obj
        from memflow.reranker import NoopReranker, RerankPipeline

        with self._rerank_pipeline_lock:
            if self._rerank_pipeline_obj is not None:
                return self._rerank_pipeline_obj
            try:
                pipeline = RerankPipeline(
                    pipeline=self._reranker_pipeline,
                    ce_top_k=self._reranker_ce_top_k or None,
                    llm_top_k=self._reranker_llm_top_k or None,
                    doc_scope=self._reranker_scope,
                )
                # Force stage construction now so config errors surface here,
                # not on the first rerank() call.
                pipeline.stages()
            except Exception as exc:
                logger.warning(
                    "Reranker pipeline %r failed to build (%s: %s); "
                    "reranking disabled, keeping retrieval order.",
                    self._reranker_pipeline,
                    type(exc).__name__,
                    exc,
                )
                # NoopReranker.rerank has the same call signature the
                # consumer uses (truncate only) — cache it so the failed
                # build is not retried on every query.
                pipeline = NoopReranker()
            self._rerank_pipeline_obj = pipeline
        return self._rerank_pipeline_obj

    def _reranker_enabled(self) -> bool:
        """Single definition of "reranking is configured on" — shared by
        search()'s candidate-depth decision and _rerank_if_enabled's routing
        so the two can never drift."""
        return bool(self._reranker_pipeline) and self._reranker_pipeline.lower() != (
            "none"
        )

    def _rerank_if_enabled(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
        rerank: bool = True,
    ) -> list[SearchResult]:
        """Rerank candidates when MEMFLOW_RERANKER_PIPELINE is set; otherwise truncate.

        ``rerank=False`` skips reranking even when it is globally enabled —
        callers can use this to request retrieval order without candidate
        over-fetch or reranker endpoint latency.
        """
        if not candidates:
            return []
        if rerank and self._reranker_enabled():
            pipeline = self._get_rerank_pipeline()
            return pipeline.rerank(query, candidates, top_k=top_k)
        return candidates[:top_k]

    def search(
        self,
        query: str | list[str],
        user_id: str | None = None,
        top_k: int = 5,
        kind: str = "skill",
        rerank: bool = True,
    ) -> list[SearchResult] | list[list[SearchResult]]:
        """Retrieve relevant skills by similarity.

        Pipeline (single-string queries):
        1. **Retrieval** — hybrid (sparse+dense RRF) when
           ``MEMFLOW_HYBRID_SEARCH_ENABLED=on`` and the store's
           ``supports_hybrid`` probe passes, otherwise dense-only. An
           enabled-but-unsupported store (e.g. a dense-only collection)
           degrades to dense with a one-time warning. Candidate depth is
           ``max(configured pool, top_k)`` — FETCH_TOP_K/RRF_TOP_K size the
           pool, but a caller asking for more than the pool always gets its
           full request (the pool is a floor for retrieval, not a ceiling on
           the caller's top_k); deep-top_k callers therefore feed more than
           FETCH_TOP_K candidates to the reranker.
        2. **Rerank** — when ``MEMFLOW_RERANKER_PIPELINE`` is set
           (``ce`` | ``llm`` | ``ce_llm`` | ``llm_ce``), candidates are
           reranked and truncated to ``top_k``. Without reranking the
           retrieved candidates are truncated to ``top_k``.

        Batch (list) queries always use the plain dense path and bypass
        the reranker.

        Args:
            query: Single query string or list of queries
            user_id: User ID for filtering
            top_k: Number of results per query (final truncation)
            kind: Only the skill record kind is supported.
            rerank: Set False to bypass the reranker (and its candidate
                over-fetch) even when ``MEMFLOW_RERANKER_PIPELINE`` is on —
                for callers that only need retrieval order.

        Returns:
            Single list for single query, list of lists for batch
        """
        _validate_skill_kind(kind)
        if isinstance(query, list):
            return self.store.search(query, top_k=top_k, user_id=user_id, kind=kind)
        rerank_on = rerank and self._reranker_enabled()
        # Hybrid path: RRF_TOP_K sets the candidate depth; reranker truncates.
        if self._hybrid_search_enabled and getattr(
            self.store, "supports_hybrid", False
        ):
            if rerank_on:
                # Shrink the pool to RERANKER_FETCH_TOP_K when it is the
                # smaller of the two configured depths — not a hard ceiling
                # on reranker input: max(..., top_k) below always honors a
                # caller asking deeper than the pool (e.g. R@200 bench runs
                # feed 200 candidates; the prompt-budget guard covers the
                # context risk). Warn once when the shrink is active, since
                # the pool silently evaluates fewer candidates than
                # RRF_TOP_K configured (.env.example only documents this as
                # a comment; every other silent degradation here warns).
                pool_k = min(self._hybrid_rrf_top_k, self._reranker_fetch_top_k)
                if (
                    self._reranker_fetch_top_k < self._hybrid_rrf_top_k
                    and not self._fetch_cap_warned
                ):
                    self._fetch_cap_warned = True
                    logger.warning(
                        "Reranker FETCH_TOP_K=%d < HYBRID_RRF_TOP_K=%d: the "
                        "candidate pool is capped at FETCH_TOP_K (queries "
                        "requesting more still get their full depth). Keep "
                        "the two equal unless the shrink is intended.",
                        self._reranker_fetch_top_k,
                        self._hybrid_rrf_top_k,
                    )
                rrf_k = max(pool_k, top_k)
            else:
                rrf_k = max(self._hybrid_rrf_top_k, top_k)
            candidates = self._search_hybrid(query, user_id, kind, rrf_k)
            return self._rerank_if_enabled(query, candidates, top_k, rerank=rerank)
        # Enabled but the store/collection can't serve hybrid (e.g. a
        # dense-only collection). Degrade to dense — warn once, since the
        # capability is fixed at store init and per-query warnings spam.
        if self._hybrid_search_enabled and not self._hybrid_fallback_warned:
            self._hybrid_fallback_warned = True
            logger.warning(
                "Hybrid search enabled but %s is attached to a collection "
                "without sparse vectors (created before hybrid support); "
                "using dense search. Existing collections keep their "
                "schema regardless of the switch or re-seeding. To enable "
                "hybrid, use a separate new collection created with "
                "MEMFLOW_HYBRID_SEARCH_ENABLED=on.",
                type(self.store).__name__,
            )
        # Dense-only path: over-fetch to reranker candidate depth when reranking.
        fetch_k = max(top_k, self._reranker_fetch_top_k) if rerank_on else top_k
        candidates = self.store.search(query, top_k=fetch_k, user_id=user_id, kind=kind)
        return self._rerank_if_enabled(query, candidates, top_k, rerank=rerank)

    async def search_async(
        self,
        query: str | list[str],
        user_id: str | None = None,
        top_k: int = 5,
        kind: str = "skill",
        max_workers: int = 50,
    ) -> list[SearchResult] | list[list[SearchResult]]:
        """Retrieve relevant skills by similarity asynchronously.

        Args:
            query: Single query string or list of queries
            user_id: User ID for filtering
            top_k: Number of results per query
            kind: Only the skill record kind is supported.
            max_workers: Max concurrent requests (default: 50)

        Returns:
            Single list for single query, list of lists for batch
        """
        _validate_skill_kind(kind)
        if isinstance(query, list):
            return await self.store.search_async(
                query,
                top_k=top_k,
                user_id=user_id,
                kind=kind,
                max_workers=max_workers,
            )
        else:
            import asyncio

            return await asyncio.to_thread(self.search, query, user_id, top_k, kind)

    def delete(
        self,
        id: str | list[str],
        user_id: str | None = None,
    ) -> dict:
        """Delete stored records by exact ID.

        Args:
            id: Single ID or list of IDs
            user_id: User ID for filtering

        Returns:
            dict with event info for single, dict with counts for batch
        """
        if isinstance(id, list):
            # Batch delete
            total = len(id)
            num_deleted = self.store.delete(id)
            return {
                "num_deleted": num_deleted,
                "num_failed": total - num_deleted,
                "total": total,
            }
        else:
            # Single delete
            record = None
            candidates = self.store.list(user_id=user_id)
            for candidate in candidates:
                if candidate.id == id:
                    record = candidate
                    break

            if record is None:
                return {
                    "id": id,
                    "event": "DELETE",
                    "deleted": False,
                    "error": "not_found",
                }

            try:
                num_deleted = self.store.delete(record.id)
                deleted = num_deleted > 0
            except Exception:
                deleted = False

            if deleted:
                return {
                    "id": record.id,
                    "title": record.title,
                    "event": "DELETE",
                    "deleted": True,
                }

            return {
                "id": record.id,
                "title": record.title,
                "event": "DELETE",
                "deleted": False,
                "error": "delete_failed",
            }

    async def delete_async(
        self,
        id: str | list[str],
        user_id: str | None = None,
        max_workers: int = 50,
    ) -> dict:
        """Delete stored records asynchronously.

        Args:
            id: Single ID or list of IDs
            user_id: User ID for filtering
            max_workers: Max concurrent operations (default: 50)

        Returns:
            dict with event info for single, dict with counts for batch
        """
        # Note: user_id is only used for pre-deletion lookup, not passed to store.delete_async

        if isinstance(id, list):
            # Batch async delete
            total = len(id)
            num_deleted = await self.store.delete_async(id, max_workers=max_workers)
            return {
                "num_deleted": num_deleted,
                "num_failed": total - num_deleted,
                "total": total,
            }
        else:
            # Single async delete - need to look up the record for its title
            record = None
            candidates = self.store.list(user_id=user_id)
            for candidate in candidates:
                if candidate.id == id:
                    record = candidate
                    break

            if record is None:
                return {
                    "id": id,
                    "event": "DELETE",
                    "deleted": False,
                    "error": "not_found",
                }

            num_deleted = await self.store.delete_async(id, max_workers=max_workers)
            deleted = num_deleted > 0

            if deleted:
                return {
                    "id": record.id,
                    "title": record.title,
                    "event": "DELETE",
                    "deleted": True,
                }

            return {
                "id": record.id,
                "title": record.title,
                "event": "DELETE",
                "deleted": False,
                "error": "delete_failed",
            }

    async def add_async(
        self,
        skill: SkillRecord | list[SkillRecord],
        *,
        max_workers: int = 50,
    ) -> dict:
        """Add a skill or skills asynchronously.

        Args:
            skill: Single SkillRecord or list of SkillRecords
            max_workers: Max concurrent operations (default: 50)

        Returns:
            dict with event info for single, dict with counts for batch
        """

        _validate_skill_input(skill)
        if isinstance(skill, list):
            # Batch async add
            total = len(skill)
            num_seeded = (
                await self.store.add_async(skill, max_workers=max_workers)
                if skill
                else 0
            )
            return {
                "num_seeded": num_seeded,
                "num_skipped": total - num_seeded,
                "total": total,
            }
        else:
            # Single async add
            await self.store.add_async(skill, max_workers=max_workers)
            return {"id": skill.id, "title": skill.title, "event": "ADD"}
