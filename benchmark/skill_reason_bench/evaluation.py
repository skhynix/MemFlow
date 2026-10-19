# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation module for SkillReason-Bench.

SkillReason-Bench differs from SkillRet/SkillRouter in one structural way that
shapes this module: **every query has exactly one gold skill** (recorded in
``positive_ids``), so relevance is binary and NDCG/MRR take their single-gold
forms. Queries carry a ``domain`` label (ten domains) used for stratified
reporting.

Two metric views are reported:

1. A general binary IR view (hit/precision/recall/F1/nDCG per k, MRR, MAP) for
   comparability with the sibling benchmarks.
2. The **official protocol view** — ``compute_official_metrics`` replicates the
   dataset's own ``evaluate.py`` exactly (Recall@{1,5,10,20}, NDCG@10, MRR@20),
   and ``write_official_predictions_jsonl`` emits predictions in the official
   format so the corpus can also be scored with the upstream script. The
   official metrics need rank positions up to 20, so retrieval must fetch at
   least top-20 (the runners clamp the fetch depth accordingly).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from .adapter import iter_jsonl_records

# Windows fixed by the official SkillReason-Bench protocol.
OFFICIAL_RECALL_KS = (1, 5, 10, 20)
OFFICIAL_NDCG_K = 10
OFFICIAL_MRR_K = 20


@dataclass
class SkillReasonQuery:
    """A single evaluation request from queries.jsonl."""

    query_id: str
    query: str
    domain: str
    category: str
    relevant_skill_ids: list[str]
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillReasonEvaluationResult:
    """Results from evaluating SkillReason-Bench queries."""

    overall_metrics: dict[str, Any]
    official_metrics: dict[str, Any]
    domain_stratified_metrics: dict[str, Any]
    query_results: list[dict[str, Any]]


def count_query_records(queries_path: str | Path) -> int:
    """Count the number of records in the queries JSONL file."""
    return sum(1 for _ in iter_jsonl_records(queries_path))


def _string_list(value: Any) -> list[str]:
    """Convert value to list of strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def load_skill_reason_queries(
    queries_path: str | Path,
    max_queries: int | None = None,
) -> list[SkillReasonQuery]:
    """Load SkillReason-Bench queries from ``data/queries.jsonl``.

    Expected schema:
    - query_id, query, positive_ids (gold skill IDs; single element in the
      official split), category, domain

    Records with a blank query or an empty ``positive_ids`` are dropped with a
    warning: the official split ships exactly one gold per query, and a
    gold-less record would crash ``compute_official_metrics`` (division by
    ``len(gold) == 0``) and silently dilute the binary averages.
    """
    queries: list[SkillReasonQuery] = []
    dropped_no_gold = 0
    for idx, raw in enumerate(iter_jsonl_records(queries_path), start=1):
        relevant = _string_list(raw.get("positive_ids", []))
        query = str(raw.get("query", "")).strip()
        if not query:
            continue
        if not relevant:
            dropped_no_gold += 1
            continue
        queries.append(
            SkillReasonQuery(
                query_id=str(raw.get("query_id", f"query_{idx}")),
                query=query,
                domain=str(raw.get("domain", "unknown")).strip() or "unknown",
                category=str(raw.get("category", "")).strip(),
                relevant_skill_ids=relevant,
                source_metadata={"category": str(raw.get("category", "")).strip()},
            )
        )
        if max_queries is not None and len(queries) >= max_queries:
            break
    if dropped_no_gold:
        print(
            f"Warning: dropped {dropped_no_gold} query record(s) without a gold "
            f"in positive_ids from {queries_path} — the official split always "
            "ships exactly one; a truncated or corrupted queries.jsonl scores 0 "
            "for the missing gold and would divide by zero in the official block."
        )
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
    for skill_id in retrieved_ids:
        if skill_id in relevant_ids and skill_id not in seen_relevant:
            flags.append(1)
            seen_relevant.add(skill_id)
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
        retrieved_ids: List of retrieved skill IDs in rank order
        relevant_ids: Set or list of relevant skill IDs (ground truth)
        k_values: List of k values to compute metrics at

    Returns:
        Dictionary with metrics: hit_at_k, precision_at_k, recall_at_k,
        f1_at_k, ndcg_at_k (binary), reciprocal_rank, average_precision
    """
    relevant_set = {str(skill_id) for skill_id in relevant_ids if str(skill_id)}
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


def compute_official_metrics(
    ranked_ids_by_query: dict[str, list[str]],
    gold_by_query: dict[str, set[str]],
) -> dict[str, Any]:
    """Compute the official SkillReason-Bench metric block.

    This mirrors the scoring loop of the dataset's own ``evaluate.py``
    (donghongjiang/skillreason-bench) exactly, so results are directly
    comparable with the upstream leaderboard:

      - Recall@k for k in (1, 5, 10, 20): |gold ∩ top-k| / |gold| averaged
        over ALL queries in ``gold_by_query`` (missing predictions score 0);
      - NDCG@10 with the single-gold ideal (IDCG = 1): 1/log2(rank+1) when the
        first hit is within rank 10, else 0;
      - MRR@20: 1/rank of the first hit within rank 20, else 0.

    Args:
        ranked_ids_by_query: ``{query_id -> ranked skill IDs}`` (only queries
            present in ``gold_by_query`` are scored)
        gold_by_query: ``{query_id -> gold skill IDs}``

    Returns:
        Official-style metrics dict (same keys as upstream evaluate.py).
    """
    recall = {k: 0.0 for k in OFFICIAL_RECALL_KS}
    ndcg = 0.0
    mrr = 0.0

    for query_id, gold in gold_by_query.items():
        ranked = ranked_ids_by_query.get(query_id, [])
        for k in OFFICIAL_RECALL_KS:
            recall[k] += len(gold.intersection(ranked[:k])) / len(gold)
        first_rank = next(
            (
                index + 1
                for index, skill_id in enumerate(ranked[:OFFICIAL_MRR_K])
                if skill_id in gold
            ),
            None,
        )
        if first_rank is not None:
            mrr += 1.0 / first_rank
            if first_rank <= OFFICIAL_NDCG_K:
                ndcg += 1.0 / math.log2(first_rank + 1)

    total = len(gold_by_query)
    if total == 0:
        # A gold-less query set scores nothing meaningful (upstream would
        # divide by zero here); fail loudly instead of emitting all-zero
        # metrics that read like a legitimate bottom-of-the-leaderboard run.
        raise ValueError(
            "compute_official_metrics received no gold queries; every query "
            "record was dropped (empty positive_ids) or the file is empty"
        )
    metrics: dict[str, Any] = {
        f"Recall@{k}": recall[k] / total for k in OFFICIAL_RECALL_KS
    }
    metrics.update(
        {
            f"NDCG@{OFFICIAL_NDCG_K}": ndcg / total,
            f"MRR@{OFFICIAL_MRR_K}": mrr / total,
            "num_queries": total,
            "prediction_coverage": (
                total - len(set(gold_by_query) - set(ranked_ids_by_query))
            )
            / total,
        }
    )
    return metrics


def write_official_predictions_jsonl(
    path: str | Path,
    ranked_ids_by_query: dict[str, list[str]],
) -> None:
    """Write predictions in the official ``evaluate.py`` input format.

    One JSON object per line: ``{"query_id": ..., "retrieved_skill_ids": [...]}``.
    Re-score with the dataset's own script via::

        python evaluate.py --queries data/queries.jsonl --predictions <path>
    """
    out = Path(path)
    with out.open("w", encoding="utf-8") as handle:
        for query_id, ranked in ranked_ids_by_query.items():
            handle.write(
                json.dumps({"query_id": query_id, "retrieved_skill_ids": list(ranked)})
                + "\n"
            )


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
    """Aggregate per-query binary IR metrics across queries."""
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


def _safe_float(value: Any) -> float:
    """Safely convert value to float."""
    try:
        return float(value)
    except Exception:
        return 0.0


def _assemble_results(
    queries: list[SkillReasonQuery],
    all_retrieved: list[list[Any]],
    k_values: list[int],
) -> SkillReasonEvaluationResult:
    """Score retrieved rankings against the queries (shared sync/async path)."""
    query_results: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    by_domain: dict[str, list[dict[str, Any]]] = {}
    ranked_ids_by_query: dict[str, list[str]] = {}
    gold_by_query: dict[str, set[str]] = {}

    for query, retrieved in zip(queries, all_retrieved):
        retrieved_ids = [str(item.skill_id) for item in retrieved]
        ranked_ids_by_query[query.query_id] = retrieved_ids
        gold_by_query[query.query_id] = set(query.relevant_skill_ids)

        metrics = compute_binary_ir_metrics(
            retrieved_ids=retrieved_ids,
            relevant_ids=query.relevant_skill_ids,
            k_values=k_values,
        )
        all_metrics.append(metrics)
        by_domain.setdefault(query.domain, []).append(metrics)

        retrieved_payload = [
            {
                "rank": rank,
                "procedure_id": item.skill_id,
                "title": item.title,
                "category": item.category,
                "tags": item.tags,
                "score": _safe_float(item.score),
                "relevant": item.skill_id in set(query.relevant_skill_ids),
            }
            for rank, item in enumerate(retrieved, start=1)
        ]

        query_results.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "domain": query.domain,
                "category": query.category,
                "relevant_skill_ids": query.relevant_skill_ids,
                "retrieved": retrieved_payload,
                "metrics": {
                    **metrics,
                    "num_retrieved": len(retrieved_payload),
                },
            }
        )

    return SkillReasonEvaluationResult(
        overall_metrics=aggregate_query_metrics(all_metrics, k_values),
        official_metrics=compute_official_metrics(ranked_ids_by_query, gold_by_query),
        domain_stratified_metrics={
            domain: aggregate_query_metrics(metrics, k_values)
            for domain, metrics in sorted(by_domain.items())
        },
        query_results=query_results,
    )


def evaluate_skill_reason_queries(
    retrieval_system: Any,
    queries: list[SkillReasonQuery],
    k_values: list[int],
    top_k: int,
) -> SkillReasonEvaluationResult:
    """Evaluate retrieval performance using batch processing (sync version)."""
    total = len(queries)
    print(f"Starting SkillReason-Bench evaluation of {total} queries (sync batch)...")

    batch_queries = [(query.query, set()) for query in queries]
    all_retrieved = retrieval_system.retrieve_batch(queries=batch_queries, k=top_k)

    return _assemble_results(queries, all_retrieved, k_values)


async def evaluate_skill_reason_queries_async(
    retrieval_system: Any,
    queries: list[SkillReasonQuery],
    k_values: list[int],
    top_k: int,
    max_concurrency: int = 64,
) -> SkillReasonEvaluationResult:
    """Evaluate retrieval performance using async batch processing."""
    total = len(queries)
    print(f"Starting SkillReason-Bench evaluation of {total} queries (async)...")

    batch_queries = [(query.query, set()) for query in queries]
    all_retrieved = await retrieval_system.retrieve_batch_async(
        queries=batch_queries,
        k=top_k,
        max_concurrency=max_concurrency,
    )

    return _assemble_results(queries, all_retrieved, k_values)
