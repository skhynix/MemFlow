#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate retrieval performance on WikiHow Procedure Silver query bank.

This script loads a query bank and evaluates retrieval performance
using MemFlow.search(). The corpus must be seeded beforehand.

Usage:
    # Basic usage (uses all defaults)
    uv run benchmark/wikihow_procedure_silver/run_retrieval.py

    # Custom query bank path
    uv run benchmark/wikihow_procedure_silver/run_retrieval.py \\
        --query-bank-path benchmark/wikihow_procedure_silver/benchmark_data/query_bank.jsonl

    # Limit queries for quick test
    uv run benchmark/wikihow_procedure_silver/run_retrieval.py --max-queries 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from memflow import MemFlow
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from memflow import MemFlow

from benchmark.wikihow_procedure_silver.adapter import (  # noqa: E402
    MemFlowWikiHowAdapter,
)
from benchmark.wikihow_procedure_silver.evaluation import (  # noqa: E402
    count_query_bank_records,
    evaluate_wikihow_queries,
    evaluate_wikihow_queries_async,
    load_wikihow_query_bank,
)

# Default paths relative to this script location
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_QUERY_BANK_PATH = SCRIPT_DIR / "benchmark_data" / "query_bank.jsonl"
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent.parent / "results"


def _load_env_file(env_path: str | None = None) -> None:
    """Load environment variables from .env file."""
    from dotenv import load_dotenv

    if env_path is None:
        env_path = ".env"

    path = Path(env_path)
    if not path.exists():
        return

    load_dotenv(dotenv_path=path, override=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval on WikiHow Procedure Silver query bank."
    )
    parser.add_argument(
        "--query-bank-path",
        type=Path,
        default=DEFAULT_QUERY_BANK_PATH,
        help=f"Path to query bank JSONL (default: {DEFAULT_QUERY_BANK_PATH})",
    )
    parser.add_argument(
        "--user-id",
        default="benchmark",
        help="User ID for seeded procedures (default: benchmark)",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="K values for evaluation metrics (default: 1 3 5 10)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Evaluate only first N queries (default: all)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory for results JSON (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--results-filename",
        default=None,
        help="Results filename (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous sequential mode (default: False)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=64,
        help="Max concurrent queries (default: 64)",
    )
    return parser.parse_args()


def _normalized_k_values(raw_values: list[int]) -> list[int]:
    return sorted({k for k in raw_values if k > 0})


def _results_path(results_dir: Path, filename: str | None) -> Path:
    if filename:
        final_name = filename if filename.endswith(".json") else f"{filename}.json"
        return results_dir / final_name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return results_dir / f"wikihow_retrieval_{stamp}.json"


def _print_summary(
    system_name: str,
    system_info: dict[str, Any],
    corpus_size: int,
    num_queries: int,
    execution_time: float,
    overall: dict[str, Any],
    k_values: list[int],
) -> None:
    print("\n=== WikiHow Retrieval Evaluation Summary ===")
    print(f"System: {system_name}")
    print(f"Backend: {system_info.get('backend')}")
    print(f"Corpus size: {corpus_size}")
    print(f"Queries evaluated: {num_queries}")
    print(f"Execution time: {execution_time:.3f}s")
    print(f"MRR: {overall.get('mrr', 0.0):.4f}")
    print(f"MAP: {overall.get('map', 0.0):.4f}")

    for k in k_values:
        sk = str(k)
        print(
            f"k={k}: "
            f"Hit@k={overall.get('hit_at_k', {}).get(sk, 0.0):.4f}  "
            f"P@k={overall.get('precision_at_k', {}).get(sk, 0.0):.4f}  "
            f"R@k={overall.get('recall_at_k', {}).get(sk, 0.0):.4f}  "
            f"F1@k={overall.get('f1_at_k', {}).get(sk, 0.0):.4f}  "
            f"NDCG@k={overall.get('ndcg_at_k', {}).get(sk, 0.0):.4f}"
        )


def main() -> None:
    _load_env_file()
    args = _parse_args()

    k_values = _normalized_k_values(args.k_values)
    if not k_values:
        raise ValueError("--k-values must contain at least one positive integer")
    top_k = max(k_values)

    backend = os.environ.get("MEMFLOW_BACKEND", "emulated")
    llm_provider = os.environ.get("LLM_PROVIDER", "ollama")
    llm_model = os.environ.get("LLM_MODEL", "llama3.2")

    print("\n=== Starting WikiHow Retrieval Evaluation ===")
    print(f"Query bank: {args.query_bank_path}")
    print(f"User ID: {args.user_id}")
    print(f"K values: {k_values}")
    print(f"Max queries: {args.max_queries or 'all'}")
    if args.sync:
        print("Mode: synchronous (sequential) processing")
    else:
        print("Mode: async parallel processing")

    memflow = MemFlow(sync_mode=args.sync)

    # Get corpus size from existing procedures
    existing = memflow.store.list_all(user_id=args.user_id)
    corpus_size = len(list(existing))

    if corpus_size == 0:
        print("\n[WARNING] No procedures found in corpus. Run run_seeding.py first.")
        sys.exit(1)

    print(f"Corpus size: {corpus_size}")

    start = time.perf_counter()

    retrieval_system = MemFlowWikiHowAdapter(
        memflow=memflow,
        user_id=args.user_id,
        corpus_size=corpus_size,
        backend=backend,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    query_bank_total = count_query_bank_records(args.query_bank_path)
    queries = load_wikihow_query_bank(
        query_bank_path=args.query_bank_path,
        max_queries=args.max_queries,
    )

    if args.sync:
        # Sync mode: batch embedding
        eval_result = evaluate_wikihow_queries(
            retrieval_system=retrieval_system,
            queries=queries,
            k_values=k_values,
            top_k=top_k,
        )
    else:
        # Async mode: parallel processing
        import asyncio

        eval_result = asyncio.run(
            evaluate_wikihow_queries_async(
                retrieval_system=retrieval_system,
                queries=queries,
                k_values=k_values,
                top_k=top_k,
                max_concurrency=args.max_concurrency,
            )
        )

    elapsed = time.perf_counter() - start
    system_name = retrieval_system.get_system_name()
    system_info = retrieval_system.get_system_info()

    results_payload = {
        "benchmark_name": "wikihow_procedure_silver_v1",
        "run_mode": "retrieval_evaluation",
        "system_name": system_name,
        "system_info": system_info,
        "settings": {
            "user_id": args.user_id,
            "k_values": k_values,
            "top_k": top_k,
            "query_bank_path": str(args.query_bank_path),
            "max_queries": args.max_queries,
            "backend": backend,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "sync_mode": args.sync,
        },
        "corpus_stats": {
            "corpus_size": corpus_size,
        },
        "query_bank_stats": {
            "num_queries_total": query_bank_total,
            "num_queries_evaluated": len(queries),
            "max_queries": args.max_queries,
        },
        "overall_metrics": eval_result.overall_metrics,
        "source_category_stratified_metrics": (
            eval_result.source_category_stratified_metrics
        ),
        "execution_time_seconds": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_results": eval_result.query_results,
    }

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = _results_path(results_dir, args.results_filename)
    output_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    _print_summary(
        system_name=system_name,
        system_info=system_info,
        corpus_size=corpus_size,
        num_queries=len(queries),
        execution_time=elapsed,
        overall=eval_result.overall_metrics,
        k_values=k_values,
    )
    print(f"\nSaved results: {output_path}")


if __name__ == "__main__":
    main()
