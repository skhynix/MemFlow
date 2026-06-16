# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from memflow import MemFlow, Procedure

WIKIHOW_CORPUS_SOURCE = "paolop/human-instructions-dataset-updated-json-files"


@dataclass
class WikiHowProcedureRecord:
    id: str
    title: str
    content: str
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorpusSeedStats:
    source_repo: str
    corpus_path: str
    num_records: int = 0
    num_seeded: int = 0
    num_reused: int = 0
    num_skipped: int = 0
    num_deleted: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)

    @property
    def active_corpus_size(self) -> int:
        return self.num_seeded + self.num_reused

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["active_corpus_size"] = self.active_corpus_size
        return payload


@dataclass
class RetrievedWikiHowProcedure:
    procedure_id: str
    title: str
    category: str
    score: float
    tags: list[str] = field(default_factory=list)


def iter_jsonl_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream a JSONL file as dictionaries."""
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {jsonl_path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected object at {jsonl_path}:{line_number}, "
                    f"got {type(record).__name__}"
                )
            yield record


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def normalize_wikihow_record(raw: dict[str, Any]) -> WikiHowProcedureRecord:
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    return WikiHowProcedureRecord(
        id=str(raw.get("id", "")).strip(),
        title=str(raw.get("title", "")).strip(),
        content=str(raw.get("content", "")).strip(),
        category=str(raw.get("category", "general")).strip() or "general",
        tags=_string_list(raw.get("tags")),
        metadata=dict(metadata),
    )


def iter_wikihow_procedures(path: str | Path) -> Iterator[WikiHowProcedureRecord]:
    for raw in iter_jsonl_records(path):
        yield normalize_wikihow_record(raw)


def wikihow_record_to_procedure(
    record: WikiHowProcedureRecord | dict[str, Any], user_id: str
) -> Procedure:
    """Convert a WikiHow corpus record into a direct MemFlow Procedure."""
    normalized = (
        normalize_wikihow_record(record) if isinstance(record, dict) else record
    )
    return Procedure(
        id=normalized.id,
        title=normalized.title,
        content=normalized.content,
        user_id=user_id,
        category=normalized.category,
        tags=list(normalized.tags),
        kind="procedure",
    )


def _collect_corpus_ids(corpus_path: str | Path) -> set[str]:
    return {record.id for record in iter_wikihow_procedures(corpus_path) if record.id}


def _collect_existing_procedure_ids(memflow: MemFlow, user_id: str) -> set[str]:
    try:
        return {proc.id for proc in memflow.store.list_all(user_id=user_id) if proc.id}
    except Exception as exc:
        raise RuntimeError(
            "Failed to list existing procedures before WikiHow corpus reuse for "
            f"user_id={user_id!r}; refusing to seed because duplicate corpus IDs "
            "could be created. Resolve the list_all error before rerunning; use "
            "--clear-existing only for an intentional fresh reseed."
        ) from exc


def seed_wikihow_corpus(
    memflow: MemFlow,
    user_id: str,
    corpus_path: str | Path,
    clear_existing: bool = False,
    batch_size: int = 100,
) -> CorpusSeedStats:
    """Seed WikiHow procedures into MemFlow using batch embedding.

    Uses memflow.add() with list for efficient batch processing.

    Args:
        memflow: MemFlow instance
        user_id: User ID for procedures
        corpus_path: Path to JSONL corpus file
        clear_existing: Whether to clear existing procedures first
        batch_size: Number of procedures to add in each batch
    """
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus path not found: {corpus_path}")

    stats = CorpusSeedStats(
        source_repo=WIKIHOW_CORPUS_SOURCE,
        corpus_path=str(corpus_path),
    )

    print("\n=== Seeding WikiHow Procedure Silver Corpus ===")

    existing_ids: set[str] = set()
    if clear_existing:
        print("Clearing existing WikiHow procedures...")
        corpus_ids = _collect_corpus_ids(corpus_path)
        for proc in memflow.store.list_all(user_id=user_id):
            if proc.id in corpus_ids:
                memflow.store.delete(proc.id)
                stats.num_deleted += 1
        print(f"  Deleted {stats.num_deleted} existing procedures")
    else:
        existing_ids = _collect_existing_procedure_ids(memflow, user_id=user_id)
        print(f"Reusing existing procedures by ID ({len(existing_ids)} available)")

    categories: Counter[str] = Counter()
    print("Streaming procedures from JSONL...")

    # Collect procedures to seed
    procedures_to_seed: list[Procedure] = []
    for record in iter_wikihow_procedures(corpus_path):
        stats.num_records += 1
        if not record.id:
            stats.num_skipped += 1
            continue

        categories[record.category] += 1
        if not clear_existing and record.id in existing_ids:
            stats.num_reused += 1
            continue

        procedure = wikihow_record_to_procedure(record, user_id=user_id)
        procedures_to_seed.append(procedure)

        # Add in batches
        if len(procedures_to_seed) >= batch_size:
            result = memflow.add(procedure=procedures_to_seed, user_id=user_id)
            stats.num_seeded += result.get("num_seeded", 0)
            stats.num_skipped += result.get("num_skipped", 0)
            print(
                f"\r  Seeded {stats.num_seeded} procedures ({stats.num_reused} reused)",
                end="",
                flush=True,
            )
            procedures_to_seed = []

    # Add remaining procedures
    if procedures_to_seed:
        result = memflow.add(procedure=procedures_to_seed, user_id=user_id)
        stats.num_seeded += result.get("num_seeded", 0)
        stats.num_skipped += result.get("num_skipped", 0)

    stats.category_counts = dict(sorted(categories.items()))
    print()
    print(
        f"Corpus seeding complete: {stats.num_seeded} seeded, "
        f"{stats.num_reused} reused "
        f"({stats.num_skipped} skipped)\n"
    )
    return stats


class MemFlowWikiHowAdapter:
    """Thin retrieval adapter around MemFlow.search()."""

    def __init__(
        self,
        memflow: MemFlow,
        user_id: str,
        corpus_size: int,
        backend: str,
        llm_provider: str,
        llm_model: str,
    ) -> None:
        self.memflow = memflow
        self.user_id = user_id
        self.corpus_size = corpus_size
        self.backend = backend
        self.llm_provider = llm_provider
        self.llm_model = llm_model

    def get_system_name(self) -> str:
        return "memflow_search_wikihow_procedure_silver"

    def get_system_info(self) -> dict[str, Any]:
        return {
            "method": "memflow.search",
            "backend": self.backend,
            "user_id": self.user_id,
            "corpus_size": self.corpus_size,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
        }

    def retrieve(
        self,
        query: str,
        k: int = 5,
        exclude_procedure_ids: set[str] | None = None,
    ) -> list[RetrievedWikiHowProcedure]:
        """Retrieve results for a single query (sync version).

        For async processing, use retrieve_async() or retrieve_batch_async().
        """
        excluded = exclude_procedure_ids or set()
        fetch_k = k + len(excluded)
        search_results = self.memflow.search(
            query, user_id=self.user_id, top_k=fetch_k, kind="procedure"
        )
        retrieved: list[RetrievedWikiHowProcedure] = []
        for result in search_results:
            procedure = result.procedure
            if procedure.id in excluded:
                continue
            score = float(result.score) if result.score is not None else 0.0
            retrieved.append(
                RetrievedWikiHowProcedure(
                    procedure_id=procedure.id,
                    title=procedure.title,
                    category=procedure.category,
                    tags=list(procedure.tags),
                    score=score,
                )
            )
        return retrieved[:k]

    def retrieve_batch(
        self,
        queries: list[tuple[str, set[str]]],
        k: int = 5,
    ) -> list[list[RetrievedWikiHowProcedure]]:
        """Retrieve results for multiple queries using batch embedding (sync version).

        Uses memflow.search() with list for efficient batch processing.
        For async processing, use retrieve_batch_async().
        """
        query_strings = [q for q, _ in queries]
        exclude_map = {i: excl for i, (_, excl) in enumerate(queries)}

        # Use search with list (auto-detects batch)
        all_results = self.memflow.search(
            query=query_strings,
            user_id=self.user_id,
            top_k=k + max(len(excl) for excl in exclude_map.values())
            if exclude_map
            else k,
            kind="procedure",
        )

        # Filter excluded results
        final_results = []
        for i, results in enumerate(all_results):
            excluded = exclude_map.get(i, set())
            filtered = []
            for result in results:
                if result.procedure.id not in excluded:
                    score = float(result.score) if result.score is not None else 0.0
                    filtered.append(
                        RetrievedWikiHowProcedure(
                            procedure_id=result.procedure.id,
                            title=result.procedure.title,
                            category=result.procedure.category,
                            tags=list(result.procedure.tags),
                            score=score,
                        )
                    )
            final_results.append(filtered[:k])

        return final_results

    async def retrieve_async(
        self,
        query: str,
        k: int = 5,
        exclude_procedure_ids: set[str] | None = None,
    ) -> list[RetrievedWikiHowProcedure]:
        """Retrieve results for a single query asynchronously."""
        import asyncio

        return await asyncio.to_thread(
            self.retrieve, query, k=k, exclude_procedure_ids=exclude_procedure_ids
        )

    async def retrieve_batch_async(
        self,
        queries: list[tuple[str, set[str]]],
        k: int = 5,
        max_concurrency: int = 64,
    ) -> list[list[RetrievedWikiHowProcedure]]:
        """Retrieve results for multiple queries in parallel.

        Args:
            queries: List of (query_string, exclude_procedure_ids) tuples
            k: Number of results to return per query
            max_concurrency: Maximum concurrent requests

        Returns:
            List of retrieved results for each query
        """
        import asyncio
        from asyncio import Semaphore

        semaphore = Semaphore(max_concurrency)

        async def retrieve_single(
            query: str, exclude_procedure_ids: set[str]
        ) -> list[RetrievedWikiHowProcedure]:
            async with semaphore:
                return await asyncio.to_thread(
                    self.retrieve,
                    query,
                    k=k,
                    exclude_procedure_ids=exclude_procedure_ids,
                )

        tasks = [retrieve_single(query, exclude_ids) for query, exclude_ids in queries]
        return await asyncio.gather(*tasks)
