#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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

try:
    from memflow import MemFlowManager
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from memflow import MemFlowManager

from benchmark.proced_mem_bench.adapter import MemFlowRetrievalAdapter, seed_memflow_corpus
from benchmark.proced_mem_bench.evaluation import evaluate_gold_queries, load_gold_query_bank


DEFAULT_TIERS = ["HARD", "MEDIUM", "EASY"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MemFlow gold-label retrieval benchmark.")

    parser.add_argument("--user-id", default="benchmark")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--tiers", nargs="+", choices=DEFAULT_TIERS, default=DEFAULT_TIERS)
    parser.add_argument("--max-queries-per-tier", type=int)
    parser.add_argument("--query-bank-path")
    parser.add_argument("--corpus-path")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--results-filename")
    parser.add_argument("--clear-existing", action="store_true", help="Clear existing procedures before seeding (default: False)")

    return parser.parse_args()


def _normalized_k_values(raw_values: list[int]) -> list[int]:
    return sorted({k for k in raw_values if k > 0})


def _select_queries(queries: list[Any], tiers: list[str], max_per_tier: int | None) -> list[Any]:
    selected: list[Any] = []
    for tier in tiers:
        tier_queries = [q for q in queries if q.complexity_tier == tier]
        if max_per_tier is not None:
            tier_queries = tier_queries[:max_per_tier]
        selected.extend(tier_queries)

    if not selected:
        raise ValueError(f"No queries found for specified tiers: {tiers}")

    return selected


def _results_path(results_dir: Path, filename: str | None) -> Path:
    if filename:
        final_name = filename if filename.endswith(".json") else f"{filename}.json"
        return results_dir / final_name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return results_dir / f"memflow_gold_benchmark_{stamp}.json"


def _print_summary(
    system_name: str,
    system_info: dict[str, Any],
    corpus_size: int,
    num_queries: int,
    execution_time: float,
    overall: dict[str, Any],
    stratified: dict[str, Any],
    k_values: list[int],
) -> None:
    print("\n=== MemFlow Gold Benchmark Summary ===")
    print(f"System: {system_name}")
    print(f"Backend: {system_info.get('backend')}")
    print(f"Corpus size: {corpus_size}")
    print(f"Queries evaluated: {num_queries}")
    print(f"Execution time (s): {execution_time:.3f}")
    print(f"MAP: {overall.get('map', 0.0):.4f}")

    for k in k_values:
        sk = str(k)
        print(
            f"k={k}: "
            f"P@k={overall.get('precision_at_k', {}).get(sk, 0.0):.4f}  "
            f"R@k={overall.get('recall_at_k', {}).get(sk, 0.0):.4f}  "
            f"F1@k={overall.get('f1_at_k', {}).get(sk, 0.0):.4f}  "
            f"NDCG@k={overall.get('ndcg_at_k', {}).get(sk, 0.0):.4f}"
        )

    print("\nComplexity tiers:")
    for tier, metrics in stratified.items():
        print(f"- {tier}: MAP={metrics.get('map', 0.0):.4f}, queries={metrics.get('num_queries', 0)}")


def main() -> None:
    # Load .env file before parsing args.
    # Priority: CLI (execution params only) > env > .env > hardcoded default
    # CLI options are for benchmark execution parameters only.
    # Infrastructure settings (backend, LLM, etc.) are loaded from env/.env.
    _load_env_file()
    args = _parse_args()
    k_values = _normalized_k_values(args.k_values)
    if not k_values:
        raise ValueError("--k-values must contain at least one positive integer")
    top_k = max(k_values)

    backend = os.environ.get("MEMFLOW_BACKEND", "emulated")
    llm_provider = os.environ.get("LLM_PROVIDER", "ollama")
    llm_model = os.environ.get("LLM_MODEL", "llama3.2")

    manager = MemFlowManager()

    start = time.perf_counter()

    trajectory_map = seed_memflow_corpus(
        manager=manager,
        user_id=args.user_id,
        corpus_path=args.corpus_path,
        clear_existing=args.clear_existing,
    )

    retrieval_system = MemFlowRetrievalAdapter(
        manager=manager,
        user_id=args.user_id,
        trajectory_map=trajectory_map,
        backend=backend,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    all_queries = load_gold_query_bank(query_bank_path=args.query_bank_path)
    selected_queries = _select_queries(
        queries=all_queries,
        tiers=args.tiers,
        max_per_tier=args.max_queries_per_tier,
    )

    eval_result = evaluate_gold_queries(
        retrieval_system=retrieval_system,
        queries=selected_queries,
        k_values=k_values,
        top_k=top_k,
    )

    elapsed = time.perf_counter() - start

    system_name = retrieval_system.get_system_name()
    system_info = retrieval_system.get_system_info()

    results_payload = {
        "benchmark_mode": "gold_labels",
        "system_name": system_name,
        "system_info": system_info,
        "settings": {
            "user_id": args.user_id,
            "k_values": k_values,
            "top_k": top_k,
            "tiers": args.tiers,
            "max_queries_per_tier": args.max_queries_per_tier,
            "query_bank_path": args.query_bank_path,
            "corpus_path": args.corpus_path,
            "results_dir": args.results_dir,
            "results_filename": args.results_filename,
            "clear_existing": args.clear_existing,
            "backend": backend,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "llm_api_base": os.environ.get("LLM_API_BASE"),
            "data_dir": os.environ.get("MEMFLOW_DATA_DIR"),
            "memmachine_base_url": os.environ.get("MEMMACHINE_BASE_URL"),
            "memmachine_org_id": os.environ.get("MEMMACHINE_ORG_ID"),
            "memmachine_project": os.environ.get("MEMMACHINE_PROJECT"),
            "memmachine_api_key": os.environ.get("MEMMACHINE_API_KEY"),
        },
        "corpus_stats": {
            "num_trajectories": len(trajectory_map),
            "source": "agentinstruct",
        },
        "query_bank_stats": {
            "num_queries_total": len(all_queries),
            "num_queries_evaluated": len(selected_queries),
            "tiers": args.tiers,
        },
        "overall_metrics": eval_result.overall_metrics,
        "complexity_stratified_metrics": eval_result.complexity_stratified_metrics,
        "execution_time_seconds": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_results": eval_result.query_results,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = _results_path(results_dir, args.results_filename)
    output_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    _print_summary(
        system_name=system_name,
        system_info=system_info,
        corpus_size=len(trajectory_map),
        num_queries=len(selected_queries),
        execution_time=elapsed,
        overall=eval_result.overall_metrics,
        stratified=eval_result.complexity_stratified_metrics,
        k_values=k_values,
    )
    print(f"\nSaved results: {output_path}")


if __name__ == "__main__":
    main()
