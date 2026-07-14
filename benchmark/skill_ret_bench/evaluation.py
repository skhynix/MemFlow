# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation module for SkillRet benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .adapter import iter_jsonl_records


@dataclass
class SkillRetBenchmarkQuery:
    """A single query from the SkillRet query bank."""

    query_id: str
    query: str
    source_skill_id: str
    relevant_skill_ids: list[str]
    source_metadata: dict[str, Any] = None
    relevance_notes: Any = None
    rejected_close_candidates: list[Any] = None

    def __post_init__(self):
        if self.source_metadata is None:
            self.source_metadata = {}
        if self.rejected_close_candidates is None:
            self.rejected_close_candidates = []


@dataclass
class SkillRetEvaluationResult:
    """Results from evaluating SkillRet benchmark queries."""

    overall_metrics: dict[str, Any]
    category_stratified_metrics: dict[str, Any]
    query_results: list[dict[str, Any]]


def count_query_bank_records(query_bank_path: str | Path) -> int:
    """Count the number of records in a JSONL query bank file."""
    return sum(1 for _ in iter_jsonl_records(query_bank_path))


def _string_list(value: Any) -> list[str]:
    """Convert value to list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def _normalize_query(raw: dict[str, Any], idx: int) -> SkillRetBenchmarkQuery:
    """Normalize a raw query record from anonymous-ed-benchmark/SKILLRET.

    Expected schema (HuggingFace: anonymous-ed-benchmark/SKILLRET):
    - queries: id, query, skill_ids (relevant skill IDs), k (count)
    - qrels: query_id, skill_id, relevance (binary: 1)
    """
    # Support both schemas:
    # 1. Direct queries file: id, query, skill_ids, k
    # 2. Pre-combined format: query_id, query, relevant_skill_ids, source_skill_id
    source_metadata = (
        raw.get("source_metadata")
        if isinstance(raw.get("source_metadata"), dict)
        else {}
    )
    rejected = raw.get("rejected_close_candidates")

    # Try direct schema first (anonymous-ed-benchmark)
    relevant_ids = _string_list(raw.get("skill_ids", []))
    if not relevant_ids:
        # Fallback to combined format
        relevant_ids = _string_list(raw.get("relevant_skill_ids", []))

    # Source skill ID for holdout: only use if explicitly present in the data.
    # Do NOT fall back to relevant_ids[0] — that would exclude a ground-truth
    # relevant skill from retrieval, zeroing out metrics for single-relevant
    # queries.
    source_skill_id = str(raw.get("source_skill_id", "")).strip()

    return SkillRetBenchmarkQuery(
        query_id=str(raw.get("query_id", raw.get("id", f"query_{idx}"))),
        query=str(raw.get("query", "")).strip(),
        source_skill_id=source_skill_id,
        relevant_skill_ids=relevant_ids,
        source_metadata=dict(source_metadata),
        relevance_notes=raw.get("relevance_notes"),
        rejected_close_candidates=(
            list(rejected) if isinstance(rejected, list) else _string_list(rejected)
        ),
    )


def load_skill_ret_query_bank(
    query_bank_path: str | Path,
    max_queries: int | None = None,
) -> list[SkillRetBenchmarkQuery]:
    """Load the SkillRet query bank from JSONL records.

    Args:
        query_bank_path: Path to JSONL query bank file
        max_queries: Maximum number of queries to load (for testing)

    Returns:
        List of SkillRetBenchmarkQuery objects
    """
    queries: list[SkillRetBenchmarkQuery] = []
    for idx, raw in enumerate(iter_jsonl_records(query_bank_path), start=1):
        query = _normalize_query(raw, idx)
        if not query.query:
            continue
        queries.append(query)
        if max_queries is not None and len(queries) >= max_queries:
            break
    return queries


def _discount(rank: int) -> float:
    """Compute DCG discount for a given rank."""
    return 1.0 / math.log2(rank + 1)


def _binary_relevance_flags(
    retrieved_ids: list[str], relevant_ids: set[str]
) -> list[int]:
    """Convert retrieved IDs to binary relevance flags."""
    seen_relevant: set[str] = set()
    flags: list[int] = []
    for proc_id in retrieved_ids:
        if proc_id in relevant_ids and proc_id not in seen_relevant:
            flags.append(1)
            seen_relevant.add(proc_id)
        else:
            flags.append(0)
    return flags


def compute_binary_ir_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str] | set[str],
    k_values: list[int],
) -> dict[str, Any]:
    """Compute binary IR metrics over the explicit relevant ID set.

    Args:
        retrieved_ids: List of retrieved procedure IDs in rank order
        relevant_ids: Set or list of relevant procedure IDs (ground truth)
        k_values: List of k values to compute metrics at

    Returns:
        Dictionary with metrics: hit_at_k, precision_at_k, recall_at_k,
        f1_at_k, ndcg_at_k, reciprocal_rank, average_precision
    """
    relevant_set = {str(proc_id) for proc_id in relevant_ids if str(proc_id)}
    relevance_flags = _binary_relevance_flags(retrieved_ids, relevant_set)
    num_relevant = len(relevant_set)

    hit_at_k: dict[str, float] = {}
    precision_at_k: dict[str, float] = {}
    recall_at_k: dict[str, float] = {}
    f1_at_k: dict[str, float] = {}
    ndcg_at_k: dict[str, float] = {}

    for k in k_values:
        top_flags = relevance_flags[:k]
        rel_in_top_k = sum(top_flags)
        precision = rel_in_top_k / k if k else 0.0
        recall = rel_in_top_k / num_relevant if num_relevant else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )

        dcg = sum(flag * _discount(rank) for rank, flag in enumerate(top_flags, 1))
        ideal_relevant = min(num_relevant, k)
        idcg = sum(_discount(rank) for rank in range(1, ideal_relevant + 1))

        sk = str(k)
        hit_at_k[sk] = 1.0 if rel_in_top_k > 0 else 0.0
        precision_at_k[sk] = precision
        recall_at_k[sk] = recall
        f1_at_k[sk] = f1
        ndcg_at_k[sk] = dcg / idcg if idcg else 0.0

    first_relevant_rank = 0
    precision_sum = 0.0
    running_rel = 0
    for rank, is_relevant in enumerate(relevance_flags, start=1):
        if not is_relevant:
            continue
        running_rel += 1
        precision_sum += running_rel / rank
        if not first_relevant_rank:
            first_relevant_rank = rank

    average_precision = precision_sum / num_relevant if num_relevant else 0.0
    reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    return {
        "hit_at_k": hit_at_k,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "f1_at_k": f1_at_k,
        "ndcg_at_k": ndcg_at_k,
        "reciprocal_rank": reciprocal_rank,
        "average_precision": average_precision,
        "num_relevant": num_relevant,
        "num_relevant_retrieved": running_rel,
    }


def _empty_aggregate(k_values: list[int]) -> dict[str, Any]:
    """Return empty aggregated metrics structure."""
    return {
        "num_queries": 0,
        "mrr": 0.0,
        "map": 0.0,
        "hit_at_k": {str(k): 0.0 for k in k_values},
        "precision_at_k": {str(k): 0.0 for k in k_values},
        "recall_at_k": {str(k): 0.0 for k in k_values},
        "f1_at_k": {str(k): 0.0 for k in k_values},
        "ndcg_at_k": {str(k): 0.0 for k in k_values},
    }


def aggregate_query_metrics(
    query_metrics: list[dict[str, Any]], k_values: list[int]
) -> dict[str, Any]:
    """Aggregate metrics across multiple queries.

    Args:
        query_metrics: List of per-query metric dictionaries
        k_values: List of k values used in metrics

    Returns:
        Aggregated metrics dictionary
    """
    if not query_metrics:
        return _empty_aggregate(k_values)

    return {
        "num_queries": len(query_metrics),
        "mrr": mean(m["reciprocal_rank"] for m in query_metrics),
        "map": mean(m["average_precision"] for m in query_metrics),
        "hit_at_k": {
            str(k): mean(m["hit_at_k"][str(k)] for m in query_metrics) for k in k_values
        },
        "precision_at_k": {
            str(k): mean(m["precision_at_k"][str(k)] for m in query_metrics)
            for k in k_values
        },
        "recall_at_k": {
            str(k): mean(m["recall_at_k"][str(k)] for m in query_metrics)
            for k in k_values
        },
        "f1_at_k": {
            str(k): mean(m["f1_at_k"][str(k)] for m in query_metrics) for k in k_values
        },
        "ndcg_at_k": {
            str(k): mean(m["ndcg_at_k"][str(k)] for m in query_metrics)
            for k in k_values
        },
    }


def _source_category(query: SkillRetBenchmarkQuery) -> str:
    """Extract source category from query metadata."""
    for key in (
        "source_normalized_root_category",
        "source_normalized_category",
        "category",
        "source_category",
        "skillret_category",
    ):
        value = query.source_metadata.get(key)
        if value:
            return str(value)
    return "unknown"


def _safe_float(value: Any) -> float:
    """Safely convert value to float."""
    try:
        return float(value)
    except Exception:
        return 0.0


def _source_holdout_ids(query: SkillRetBenchmarkQuery) -> set[str]:
    """Get source skill ID to hold out from retrieval."""
    source_id = str(query.source_skill_id).strip()
    return {source_id} if source_id else set()


async def evaluate_skill_ret_queries_async(
    retrieval_system: Any,
    queries: list[SkillRetBenchmarkQuery],
    k_values: list[int],
    top_k: int,
    max_concurrency: int = 64,
) -> SkillRetEvaluationResult:
    """Evaluate retrieval performance using async batch processing.

    Args:
        retrieval_system: System with retrieve_batch_async() method
        queries: List of benchmark queries
        k_values: List of k values for metrics
        top_k: Maximum k value for retrieval
        max_concurrency: Maximum concurrent requests

    Returns:
        SkillRetEvaluationResult with all metrics
    """
    query_results: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = {}

    total = len(queries)
    print(f"Starting SkillRet evaluation of {total} queries (async)...")

    # Prepare batch queries: (query_string, exclude_procedure_ids)
    batch_queries = [(query.query, _source_holdout_ids(query)) for query in queries]

    # Execute batch retrieval asynchronously
    all_retrieved = await retrieval_system.retrieve_batch_async(
        queries=batch_queries,
        k=top_k,
        max_concurrency=max_concurrency,
    )

    # Process results
    for index, (query, retrieved) in enumerate(zip(queries, all_retrieved), start=1):
        retrieved_ids = [str(item.procedure_id) for item in retrieved]

        metrics = compute_binary_ir_metrics(
            retrieved_ids=retrieved_ids,
            relevant_ids=query.relevant_skill_ids,
            k_values=k_values,
        )
        all_metrics.append(metrics)
        by_category.setdefault(_source_category(query), []).append(metrics)

        retrieved_payload = [
            {
                "rank": rank,
                "procedure_id": item.procedure_id,
                "title": item.title,
                "category": item.category,
                "tags": item.tags,
                "score": _safe_float(item.score),
            }
            for rank, item in enumerate(retrieved, start=1)
        ]

        query_results.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "source_skill_id": query.source_skill_id,
                "relevant_skill_ids": query.relevant_skill_ids,
                "source_metadata": query.source_metadata,
                "heldout_procedure_ids": sorted(_source_holdout_ids(query)),
                "relevance_notes": query.relevance_notes,
                "rejected_close_candidates": query.rejected_close_candidates,
                "retrieved": retrieved_payload,
                "metrics": {
                    **metrics,
                    "num_retrieved": len(retrieved_payload),
                },
            }
        )

        if index % max(1, total // 10) == 0 or index == total:
            pct = (index / total) * 100 if total else 100.0
            print(f"\rProgress: {index}/{total} ({pct:.1f}%)", end="", flush=True)

    print()

    return SkillRetEvaluationResult(
        overall_metrics=aggregate_query_metrics(all_metrics, k_values),
        category_stratified_metrics={
            category: aggregate_query_metrics(metrics, k_values)
            for category, metrics in sorted(by_category.items())
        },
        query_results=query_results,
    )


def evaluate_skill_ret_queries(
    retrieval_system: Any,
    queries: list[SkillRetBenchmarkQuery],
    k_values: list[int],
    top_k: int,
) -> SkillRetEvaluationResult:
    """Evaluate retrieval performance using batch processing (sync version).

    Args:
        retrieval_system: System with retrieve_batch() method
        queries: List of benchmark queries
        k_values: List of k values for metrics
        top_k: Maximum k value for retrieval

    Returns:
        SkillRetEvaluationResult with all metrics
    """
    query_results: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = {}

    total = len(queries)
    print(f"Starting SkillRet evaluation of {total} queries (sync batch)...")

    # Prepare batch queries: (query_string, exclude_procedure_ids)
    batch_queries = [(query.query, _source_holdout_ids(query)) for query in queries]

    # Execute batch retrieval
    all_retrieved = retrieval_system.retrieve_batch(
        queries=batch_queries,
        k=top_k,
    )

    # Process results
    for index, (query, retrieved) in enumerate(zip(queries, all_retrieved), start=1):
        retrieved_ids = [str(item.procedure_id) for item in retrieved]

        metrics = compute_binary_ir_metrics(
            retrieved_ids=retrieved_ids,
            relevant_ids=query.relevant_skill_ids,
            k_values=k_values,
        )
        all_metrics.append(metrics)
        by_category.setdefault(_source_category(query), []).append(metrics)

        retrieved_payload = [
            {
                "rank": rank,
                "procedure_id": item.procedure_id,
                "title": item.title,
                "category": item.category,
                "tags": item.tags,
                "score": _safe_float(item.score),
            }
            for rank, item in enumerate(retrieved, start=1)
        ]

        query_results.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "source_skill_id": query.source_skill_id,
                "relevant_skill_ids": query.relevant_skill_ids,
                "source_metadata": query.source_metadata,
                "heldout_procedure_ids": sorted(_source_holdout_ids(query)),
                "relevance_notes": query.relevance_notes,
                "rejected_close_candidates": query.rejected_close_candidates,
                "retrieved": retrieved_payload,
                "metrics": {
                    **metrics,
                    "num_retrieved": len(retrieved_payload),
                },
            }
        )

        if index % max(1, total // 10) == 0 or index == total:
            pct = (index / total) * 100 if total else 100.0
            print(f"\rProgress: {index}/{total} ({pct:.1f}%)", end="", flush=True)

    print()

    return SkillRetEvaluationResult(
        overall_metrics=aggregate_query_metrics(all_metrics, k_values),
        category_stratified_metrics={
            category: aggregate_query_metrics(metrics, k_values)
            for category, metrics in sorted(by_category.items())
        },
        query_results=query_results,
    )
