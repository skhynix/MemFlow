#!/usr/bin/env python3
"""Profile dense/sparse/hybrid retrieval on SkillRet with CLI args.

Uses the real adapter retrieve() path so exclude/holdout logic matches
run_retrieval.py exactly. The search mode is forced by monkey-patching
MemFlow.search to call store.search_hybrid with the desired mode.

Hybrid search params (rrf-weights, prefetch, rrf-k) can be overridden via
CLI flags. --rrf-top-k defaults to max(--recall-k) — fusion output caps
Recall@K, so the benchmark never clips its own metric (the online default
of 100 intentionally serves fewer).

Usage:
    uv run benchmark/skill_ret_bench/run_profiling.py --mode hybrid --recall-k 10 20 30 50 --max-queries 4392
    uv run benchmark/skill_ret_bench/run_profiling.py --mode dense sparse hybrid --recall-k 10 --max-queries 100
    uv run benchmark/skill_ret_bench/run_profiling.py --rrf-weights 0.6,0.4 --prefetch-top-k 300 --rrf-top-k 100
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env", override=True)

from benchmark.skill_ret_bench.adapter import MemFlowSkillRetAdapter  # noqa: E402
from benchmark.skill_ret_bench.evaluation import (  # noqa: E402
    _source_holdout_ids,
    compute_binary_ir_metrics,
    load_skill_ret_query_bank,
)
from memflow import MemFlow  # noqa: E402

DEFAULT_QUERY_BANK = (
    Path(__file__).resolve().parent
    / "data"
    / "SKILLRET"
    / "data"
    / "queries"
    / "test.jsonl"
)

ALL_MODES = ("dense", "sparse", "hybrid")


def _parse_weights(raw: str) -> list[float]:
    parts = [w.strip() for w in raw.split(",") if w.strip()]
    if len(parts) != 2:
        raise ValueError(f"--rrf-weights expects 2 values, got {len(parts)}: {raw!r}")
    weights = [float(w) for w in parts]
    if any(w < 0 for w in weights):
        raise ValueError(f"--rrf-weights must be non-negative, got {raw!r}")
    return weights


def patch_search_mode(
    memflow: MemFlow,
    mode: str,
    *,
    rrf_weights: list[float],
    sparse_top_k: int,
    dense_top_k: int,
    rrf_top_k: int,
    rrf_k: int,
) -> None:
    """Force memflow.search() to route through store.search_hybrid(mode=...)."""

    def search(query, user_id=None, top_k=5, kind="skill"):
        if isinstance(query, list):
            return memflow.store.search(query, top_k=top_k, user_id=user_id, kind=kind)
        effective_rrf_top_k = max(rrf_top_k, top_k)
        candidates = memflow.store.search_hybrid(
            query=query,
            rrf_top_k=effective_rrf_top_k,
            user_id=user_id,
            kind=kind,
            mode=mode,
            rrf_weights=rrf_weights,
            sparse_top_k=sparse_top_k,
            dense_top_k=dense_top_k,
            rrf_k=rrf_k,
        )
        return candidates[:top_k]

    memflow.search = search


async def run_mode(adapter, queries, k_values, max_concurrency=64):
    top_k = max(k_values)
    batch_queries = [(q.query, _source_holdout_ids(q)) for q in queries]
    all_retrieved = await adapter.retrieve_batch_async(
        queries=batch_queries, k=top_k, max_concurrency=max_concurrency
    )
    per_k = {k: [] for k in k_values}
    for q, retrieved in zip(queries, all_retrieved):
        relevant = {str(sid) for sid in q.relevant_skill_ids if str(sid)}
        if not relevant:
            continue
        retrieved_ids = [r.procedure_id for r in retrieved]
        metrics = compute_binary_ir_metrics(
            retrieved_ids=retrieved_ids,
            relevant_ids=q.relevant_skill_ids,
            k_values=k_values,
        )
        for k in k_values:
            per_k[k].append(metrics["recall_at_k"][str(k)])
    return {k: sum(v) / len(v) if v else 0.0 for k, v in per_k.items()}


def _parse_args():
    p = argparse.ArgumentParser(
        description="Profile dense/sparse/hybrid retrieval on SkillRet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- Pipeline params (in search order: prefetch → fusion → truncation) ---
    # 1) Prefetch depth: each channel retrieves this many candidates
    p.add_argument(
        "--prefetch-top-k",
        type=int,
        default=200,
        help="Sparse AND dense prefetch depth (default: 200). Overridden by "
        "--sparse-top-k/--dense-top-k when either is given.",
    )
    p.add_argument(
        "--sparse-top-k",
        type=int,
        default=None,
        help="Sparse prefetch depth (default: --prefetch-top-k)",
    )
    p.add_argument(
        "--dense-top-k",
        type=int,
        default=None,
        help="Dense prefetch depth (default: --prefetch-top-k)",
    )
    # 2) RRF fusion: combine the two channels
    p.add_argument(
        "--rrf-weights",
        default="0.6,0.4",
        help='RRF weights "dense_w,sparse_w" (default: 0.6,0.4). '
        "Parsed after argparse so the default is parsed too.",
    )
    p.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant k in score = 1/(k+rank) (default: 60)",
    )
    p.add_argument(
        "--rrf-top-k",
        type=int,
        default=None,
        help="RRF fusion output depth (default: max --recall-k — fusion output "
        "caps Recall@K, so it must never be below the largest evaluated K)",
    )
    # 3) Final truncation: evaluation K values
    p.add_argument(
        "--recall-k",
        nargs="+",
        type=int,
        default=[10, 20, 30],
        help="K values for Recall@K (default: 10 20 30)",
    )
    # --- Run config ---
    p.add_argument(
        "--mode",
        nargs="+",
        default=["hybrid"],
        choices=ALL_MODES,
        help="Search mode(s) to evaluate (default: hybrid)",
    )
    p.add_argument(
        "--max-queries",
        type=int,
        default=4392,
        help="Limit number of queries (default: 4392)",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=64,
        help="Max concurrent queries (default: 64)",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    k_values = sorted(set(k for k in args.recall_k if k > 0))
    if not k_values:
        raise ValueError("--recall-k must contain at least one positive integer")

    # Resolve hybrid params (pipeline order: prefetch → fusion → truncation)
    sparse_top_k = args.sparse_top_k or args.prefetch_top_k
    dense_top_k = args.dense_top_k or args.prefetch_top_k
    rrf_weights = _parse_weights(args.rrf_weights)
    rrf_k = args.rrf_k
    # Fusion output caps Recall@K; default to the largest evaluated K so
    # every measured K is reachable (the online default 100 is intentional
    # for serving, but a benchmark must not silently clip its own metric).
    rrf_top_k = args.rrf_top_k or max(k_values)

    # Print effective config (pipeline order)
    print(f"prefetch: sparse_top_k={sparse_top_k}  dense_top_k={dense_top_k}")
    print(f"fusion:   rrf_weights={rrf_weights}  rrf_k={rrf_k}  rrf_top_k={rrf_top_k}")
    print(f"trunc:    recall_k={k_values}\n")

    queries = load_skill_ret_query_bank(
        DEFAULT_QUERY_BANK, max_queries=args.max_queries
    )
    print(f"Loaded {len(queries)} queries\n")

    for mode in args.mode:
        memflow = MemFlow()
        patch_search_mode(
            memflow,
            mode,
            rrf_weights=rrf_weights,
            sparse_top_k=sparse_top_k,
            dense_top_k=dense_top_k,
            rrf_top_k=rrf_top_k,
            rrf_k=rrf_k,
        )
        adapter = MemFlowSkillRetAdapter(
            memflow=memflow,
            user_id="benchmark",
            corpus_size=6006,
            backend="qdrant",
            llm_provider="openai-compatible",
            llm_model="zai-org/GLM-4.7-Flash",
        )
        start = time.perf_counter()
        recalls = asyncio.run(
            run_mode(adapter, queries, k_values, max_concurrency=args.max_concurrency)
        )
        elapsed = time.perf_counter() - start
        parts = "  ".join(f"R@{k}={recalls[k]:.4f}" for k in k_values)
        print(f"  {mode:8s}  {parts}  time={elapsed:.2f}s")


if __name__ == "__main__":
    main()
