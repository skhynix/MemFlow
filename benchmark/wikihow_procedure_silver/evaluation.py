# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from benchmark.wikihow_procedure_silver.adapter import iter_jsonl_records


@dataclass
class WikiHowBenchmarkQuery:
    query_id: str
    query: str
    source_procedure_id: str
    relevant_procedure_ids: list[str]
    source_metadata: dict[str, Any] = field(default_factory=dict)
    relevance_notes: Any = None
    rejected_close_candidates: list[Any] = field(default_factory=list)


@dataclass
class WikiHowEvaluationResult:
    overall_metrics: dict[str, Any]
    source_category_stratified_metrics: dict[str, Any]
    query_results: list[dict[str, Any]]


def count_query_bank_records(query_bank_path: str | Path) -> int:
    return sum(1 for _ in iter_jsonl_records(query_bank_path))


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def _normalize_query(raw: dict[str, Any], idx: int) -> WikiHowBenchmarkQuery:
    source_metadata = (
        raw.get("source_metadata")
        if isinstance(raw.get("source_metadata"), dict)
        else {}
    )
    rejected = raw.get("rejected_close_candidates")
    return WikiHowBenchmarkQuery(
        query_id=str(raw.get("query_id", raw.get("id", f"query_{idx}"))),
        query=str(raw.get("query", "")).strip(),
        source_procedure_id=str(raw.get("source_procedure_id", "")).strip(),
        relevant_procedure_ids=_string_list(raw.get("relevant_procedure_ids")),
        source_metadata=dict(source_metadata),
        relevance_notes=raw.get("relevance_notes"),
        rejected_close_candidates=(
            list(rejected) if isinstance(rejected, list) else _string_list(rejected)
        ),
    )


def load_wikihow_query_bank(
    query_bank_path: str | Path,
    max_queries: int | None = None,
) -> list[WikiHowBenchmarkQuery]:
    """Load the WikiHow query bank from JSONL records."""
    queries: list[WikiHowBenchmarkQuery] = []
    for idx, raw in enumerate(iter_jsonl_records(query_bank_path), start=1):
        query = _normalize_query(raw, idx)
        if not query.query:
            continue
        queries.append(query)
        if max_queries is not None and len(queries) >= max_queries:
            break
    return queries


def _discount(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def _binary_relevance_flags(
    retrieved_ids: list[str], relevant_ids: set[str]
) -> list[int]:
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
    """Compute binary IR metrics over the explicit relevant ID set."""
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


def _source_category(query: WikiHowBenchmarkQuery) -> str:
    for key in (
        "source_normalized_root_category",
        "source_normalized_category",
        "category",
        "source_category",
        "wikihow_category",
    ):
        value = query.source_metadata.get(key)
        if value:
            return str(value)
    return "unknown"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _source_holdout_ids(query: WikiHowBenchmarkQuery) -> set[str]:
    source_id = str(query.source_procedure_id).strip()
    return {source_id} if source_id else set()


def evaluate_wikihow_queries(
    retrieval_system: Any,
    queries: list[WikiHowBenchmarkQuery],
    k_values: list[int],
    top_k: int,
) -> WikiHowEvaluationResult:
    query_results: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    by_source_category: dict[str, list[dict[str, Any]]] = {}

    total = len(queries)
    print(f"Starting WikiHow Procedure Silver evaluation of {total} queries...")

    for index, query in enumerate(queries, start=1):
        heldout_ids = _source_holdout_ids(query)
        retrieved = retrieval_system.retrieve(
            query.query,
            k=top_k,
            exclude_procedure_ids=heldout_ids,
        )
        retrieved_ids = [str(item.procedure_id) for item in retrieved]

        metrics = compute_binary_ir_metrics(
            retrieved_ids=retrieved_ids,
            relevant_ids=query.relevant_procedure_ids,
            k_values=k_values,
        )
        all_metrics.append(metrics)
        by_source_category.setdefault(_source_category(query), []).append(metrics)

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
                "source_procedure_id": query.source_procedure_id,
                "relevant_procedure_ids": query.relevant_procedure_ids,
                "source_metadata": query.source_metadata,
                "heldout_procedure_ids": sorted(heldout_ids),
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

    return WikiHowEvaluationResult(
        overall_metrics=aggregate_query_metrics(all_metrics, k_values),
        source_category_stratified_metrics={
            category: aggregate_query_metrics(metrics, k_values)
            for category, metrics in sorted(by_source_category.items())
        },
        query_results=query_results,
    )
