# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from procedural_memory_benchmark.utils.paths import get_query_bank_path


# Relevance threshold for binary relevance classification.
# Queries with relevance_score >= this threshold are considered relevant.
DEFAULT_RELEVANCE_THRESHOLD = 6.0


@dataclass
class GoldRelevantTrajectory:
    trajectory_id: str
    relevance_score: float

    @property
    def is_relevant(self) -> bool:
        return self.relevance_score >= DEFAULT_RELEVANCE_THRESHOLD


@dataclass
class GoldBenchmarkQuery:
    query_id: str
    task_description: str
    complexity_tier: str
    query_type: str | None
    source: str | None
    relevant_trajectories: list[GoldRelevantTrajectory]


@dataclass
class GoldBenchmarkResult:
    overall_metrics: dict[str, Any]
    complexity_stratified_metrics: dict[str, Any]
    query_results: list[dict[str, Any]]


def _safe_float(value: Any, field_name: str = "unknown") -> float:
    try:
        return float(value)
    except Exception:
        # Log warning for data quality tracking
        import warnings
        warnings.warn(f"Invalid float value for {field_name}: {value!r}", RuntimeWarning, stacklevel=2)
        return 0.0


def _discount(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def _quantize_relevance_score(score: float) -> int:
    """Match Proced_mem_bench MetricsCalculator's 0-10 -> 0-3 gain mapping."""
    return min(3, max(0, int(score / 3.33)))


def load_gold_query_bank(query_bank_path: str | None = None) -> list[GoldBenchmarkQuery]:
    path = Path(query_bank_path or get_query_bank_path())
    raw_data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw_data, dict):
        records = raw_data.get("queries", [])
    else:
        records = raw_data

    queries: list[GoldBenchmarkQuery] = []
    for idx, raw in enumerate(records):
        # Support both schema variants used by query banks:
        # - task_description / complexity_tier
        # - query_text / tier
        query_id = str(raw.get("query_id", raw.get("id", f"query_{idx}")))
        task_description = str(raw.get("task_description", raw.get("query_text", "")))
        complexity_tier = str(raw.get("complexity_tier", raw.get("tier", "UNKNOWN"))).upper()
        rel_items = []
        for item in raw.get("relevant_trajectories", []) or []:
            if isinstance(item, dict):
                trajectory_id = str(item.get("trajectory_id", item.get("task_instance_id", "")))
                relevance_score = _safe_float(item.get("relevance_score", 0.0), field_name=f"query_{query_id}.relevance_score")
            else:
                trajectory_id = str(getattr(item, "trajectory_id", getattr(item, "task_instance_id", "")))
                relevance_score = _safe_float(getattr(item, "relevance_score", 0.0), field_name=f"query_{query_id}.relevance_score")
            if trajectory_id:
                rel_items.append(
                    GoldRelevantTrajectory(
                        trajectory_id=trajectory_id,
                        relevance_score=relevance_score,
                    )
                )

        queries.append(
            GoldBenchmarkQuery(
                query_id=query_id,
                task_description=task_description,
                complexity_tier=complexity_tier,
                query_type=raw.get("query_type"),
                source=raw.get("source"),
                relevant_trajectories=rel_items,
            )
        )
    return queries


def _metrics_for_query(
    retrieved_ids: list[str],
    gold_map: dict[str, float],
    k_values: list[int],
    query_id: str = "unknown",
) -> dict[str, Any]:
    # Match Proced_mem_bench's current runner behavior:
    # only the retrieved pool is judged for binary relevance/graded gain.
    retrieved_scores = {tid: gold_map.get(tid, 0.0) for tid in retrieved_ids}
    relevance_judgments = {tid: (score >= DEFAULT_RELEVANCE_THRESHOLD) for tid, score in retrieved_scores.items()}
    num_relevant = sum(1 for is_rel in relevance_judgments.values() if is_rel)

    precision_at_k: dict[str, float] = {}
    recall_at_k: dict[str, float] = {}
    f1_at_k: dict[str, float] = {}
    ndcg_at_k: dict[str, float] = {}

    for k in k_values:
        top_k_ids = retrieved_ids[:k]
        rel_in_top_k = sum(1 for tid in top_k_ids if relevance_judgments.get(tid, False))
        precision = rel_in_top_k / k if k else 0.0
        recall = (rel_in_top_k / num_relevant) if num_relevant else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        # NDCG@k uses the same quantized gains and retrieved-pool IDCG as Proced_mem_bench.
        relevance_values = [_quantize_relevance_score(retrieved_scores.get(tid, 0.0)) for tid in top_k_ids]
        dcg = 0.0
        for rank, relevance in enumerate(relevance_values, start=1):
            if relevance > 0:
                dcg += relevance * _discount(rank)

        ideal_relevance = sorted(relevance_values, reverse=True)
        idcg = 0.0
        for rank, relevance in enumerate(ideal_relevance, start=1):
            if relevance > 0:
                idcg += relevance * _discount(rank)
        ndcg = dcg / idcg if idcg > 0 else 0.0

        precision_at_k[str(k)] = precision
        recall_at_k[str(k)] = recall
        f1_at_k[str(k)] = f1
        ndcg_at_k[str(k)] = ndcg

    # AP denominator matches the retrieved-pool relevant count used upstream.
    running_rel = 0
    precision_sum = 0.0
    for rank, tid in enumerate(retrieved_ids, start=1):
        if relevance_judgments.get(tid, False):
            running_rel += 1
            precision_sum += running_rel / rank
    average_precision = precision_sum / num_relevant if num_relevant else 0.0

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "f1_at_k": f1_at_k,
        "ndcg_at_k": ndcg_at_k,
        "average_precision": average_precision,
        "num_relevant": num_relevant,
        "num_gold_relevant": sum(1 for score in gold_map.values() if score >= DEFAULT_RELEVANCE_THRESHOLD),
    }


def _aggregate_query_metrics(query_metrics: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    if not query_metrics:
        return {
            "num_queries": 0,
            "map": 0.0,
            "precision_at_k": {str(k): 0.0 for k in k_values},
            "recall_at_k": {str(k): 0.0 for k in k_values},
            "f1_at_k": {str(k): 0.0 for k in k_values},
            "ndcg_at_k": {str(k): 0.0 for k in k_values},
        }

    return {
        "num_queries": len(query_metrics),
        "map": mean(m["average_precision"] for m in query_metrics),
        "precision_at_k": {
            str(k): mean(m["precision_at_k"][str(k)] for m in query_metrics)
            for k in k_values
        },
        "recall_at_k": {
            str(k): mean(m["recall_at_k"][str(k)] for m in query_metrics)
            for k in k_values
        },
        "f1_at_k": {
            str(k): mean(m["f1_at_k"][str(k)] for m in query_metrics)
            for k in k_values
        },
        "ndcg_at_k": {
            str(k): mean(m["ndcg_at_k"][str(k)] for m in query_metrics)
            for k in k_values
        },
    }


def evaluate_gold_queries(
    retrieval_system: Any,
    queries: list[GoldBenchmarkQuery],
    k_values: list[int],
    top_k: int,
) -> GoldBenchmarkResult:
    query_results: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    by_tier: dict[str, list[dict[str, Any]]] = {}

    total = len(queries)
    print(f"Starting evaluation of {total} queries...")

    for i, query in enumerate(queries, 1):
        # Retrieve once at top_k=max(k_values) and slice per-k inside metric function.
        retrieved = retrieval_system.retrieve(query.task_description, k=top_k)
        retrieved_ids = [
            str(getattr(item, "trajectory_id", getattr(item, "task_instance_id", "")))
            for item in retrieved
        ]

        # Keep raw graded scores for NDCG and derive binary relevance from them.
        gold_map = {item.trajectory_id: item.relevance_score for item in query.relevant_trajectories}
        metrics = _metrics_for_query(retrieved_ids=retrieved_ids, gold_map=gold_map, k_values=k_values, query_id=query.query_id)

        all_metrics.append(metrics)
        by_tier.setdefault(query.complexity_tier, []).append(metrics)

        retrieved_payload = []
        for rank, item in enumerate(retrieved, start=1):
            retrieved_payload.append(
                {
                    "rank": rank,
                    "trajectory_id": str(getattr(item, "trajectory_id", getattr(item, "task_instance_id", ""))),
                    "task_description": str(getattr(item, "task_description", "")),
                    "similarity_score": _safe_float(getattr(item, "similarity_score", 0.0), field_name=f"query_{query.query_id}.similarity_score"),
                    "total_steps": int(getattr(item, "total_steps", 0) or 0),
                }
            )

        query_results.append(
            {
                "query_id": query.query_id,
                "task_description": query.task_description,
                "complexity_tier": query.complexity_tier,
                "query_type": query.query_type,
                "source": query.source,
                "retrieved": retrieved_payload,
                "gold_relevance": [
                    {
                        "trajectory_id": rel.trajectory_id,
                        "relevance_score": rel.relevance_score,
                        "is_relevant": rel.is_relevant,
                    }
                    for rel in query.relevant_trajectories
                ],
                "metrics": {
                    **metrics,
                    "num_retrieved": len(retrieved_payload),
                },
            }
        )

        # Progress display (every 10%)
        if i % max(1, total // 10) == 0 or i == total:
            pct = (i / total) * 100
            print(f"\rProgress: {i}/{total} ({pct:.1f}%)", end="", flush=True)

    print()  # newline after loop

    overall = _aggregate_query_metrics(all_metrics, k_values=k_values)
    stratified = {
        tier: _aggregate_query_metrics(metrics, k_values=k_values)
        for tier, metrics in sorted(by_tier.items())
    }

    return GoldBenchmarkResult(
        overall_metrics=overall,
        complexity_stratified_metrics=stratified,
        query_results=query_results,
    )

