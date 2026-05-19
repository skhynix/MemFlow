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
    from memflow import MemFlow
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from memflow import MemFlow

from benchmark.wikihow_procedure_silver.adapter import (  # noqa: E402
    MemFlowWikiHowAdapter,
    seed_wikihow_corpus,
)
from benchmark.wikihow_procedure_silver.evaluation import (  # noqa: E402
    count_query_bank_records,
    evaluate_wikihow_queries,
    load_wikihow_query_bank,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MemFlow on WikiHow Procedure Silver v1."
    )
    parser.add_argument("--user-id", default="benchmark")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--query-bank-path")
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--results-filename")
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Seed or reuse the WikiHow corpus and skip query evaluation.",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help=(
            "Delete matching existing WikiHow procedures for the user before "
            "seeding every valid corpus record. By default, existing IDs are reused."
        ),
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Evaluate only the first N query-bank records for smoke tests.",
    )
    args = parser.parse_args()
    if not args.seed_only and not args.query_bank_path:
        parser.error("--query-bank-path is required unless --seed-only is set")
    return args


def _normalized_k_values(raw_values: list[int]) -> list[int]:
    return sorted({k for k in raw_values if k > 0})


def _results_path(results_dir: Path, filename: str | None) -> Path:
    if filename:
        final_name = filename if filename.endswith(".json") else f"{filename}.json"
        return results_dir / final_name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return results_dir / f"memflow_wikihow_procedure_silver_{stamp}.json"


def _print_summary(
    system_name: str,
    system_info: dict[str, Any],
    corpus_size: int,
    num_queries: int,
    execution_time: float,
    overall: dict[str, Any],
    k_values: list[int],
) -> None:
    print("\n=== WikiHow Procedure Silver Summary ===")
    print(f"System: {system_name}")
    print(f"Backend: {system_info.get('backend')}")
    print(f"Corpus size: {corpus_size}")
    print(f"Queries evaluated: {num_queries}")
    print(f"Execution time (s): {execution_time:.3f}")
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


def _print_seed_summary(
    backend: str,
    corpus_size: int,
    execution_time: float,
    seed_stats: dict[str, Any],
) -> None:
    print("\n=== WikiHow Procedure Silver Seed Summary ===")
    print("Mode: seed-only")
    print(f"Backend: {backend}")
    print(f"Corpus size: {corpus_size}")
    print(f"Seeded: {seed_stats.get('num_seeded', 0)}")
    print(f"Reused: {seed_stats.get('num_reused', 0)}")
    print(f"Skipped: {seed_stats.get('num_skipped', 0)}")
    print(f"Deleted: {seed_stats.get('num_deleted', 0)}")
    print(f"Execution time (s): {execution_time:.3f}")


def main() -> None:
    # Load .env file before parsing args.
    # Priority: CLI (execution params only) > env > .env > hardcoded default
    # CLI options are for benchmark execution parameters only.
    # Infrastructure settings (backend, LLM, etc.) are loaded from env/.env.
    _load_env_file()
    args = _parse_args()

    k_values = _normalized_k_values(args.k_values)
    if not args.seed_only and not k_values:
        raise ValueError("--k-values must contain at least one positive integer")
    top_k = max(k_values) if k_values else None

    backend = os.environ.get("MEMFLOW_BACKEND", "emulated")
    llm_provider = os.environ.get("LLM_PROVIDER", "ollama")
    llm_model = os.environ.get("LLM_MODEL", "llama3.2")

    memflow = MemFlow()

    start = time.perf_counter()

    seed_stats = seed_wikihow_corpus(
        memflow=memflow,
        user_id=args.user_id,
        corpus_path=args.corpus_path,
        clear_existing=args.clear_existing,
    )
    corpus_size = seed_stats.active_corpus_size

    if args.seed_only:
        elapsed = time.perf_counter() - start
        seed_stats_payload = seed_stats.to_dict()
        results_payload = {
            "benchmark_name": "wikihow_procedure_silver_v1",
            "run_mode": "seed_only",
            "system_name": "memflow_seed_wikihow_procedure_silver",
            "system_info": {
                "method": "memflow.add",
                "backend": backend,
                "user_id": args.user_id,
                "corpus_size": corpus_size,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
            },
            "settings": {
                "user_id": args.user_id,
                "corpus_path": args.corpus_path,
                "results_dir": args.results_dir,
                "results_filename": args.results_filename,
                "seed_only": args.seed_only,
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
            "corpus_stats": seed_stats_payload,
            "execution_time_seconds": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = _results_path(results_dir, args.results_filename)
        output_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

        _print_seed_summary(
            backend=backend,
            corpus_size=corpus_size,
            execution_time=elapsed,
            seed_stats=seed_stats_payload,
        )
        print(f"\nSaved results: {output_path}")
        return

    assert top_k is not None
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

    eval_result = evaluate_wikihow_queries(
        retrieval_system=retrieval_system,
        queries=queries,
        k_values=k_values,
        top_k=top_k,
    )

    elapsed = time.perf_counter() - start
    system_name = retrieval_system.get_system_name()
    system_info = retrieval_system.get_system_info()

    results_payload = {
        "benchmark_name": "wikihow_procedure_silver_v1",
        "system_name": system_name,
        "system_info": system_info,
        "settings": {
            "user_id": args.user_id,
            "k_values": k_values,
            "top_k": top_k,
            "query_bank_path": args.query_bank_path,
            "corpus_path": args.corpus_path,
            "results_dir": args.results_dir,
            "results_filename": args.results_filename,
            "seed_only": args.seed_only,
            "clear_existing": args.clear_existing,
            "max_queries": args.max_queries,
            "source_holdout_policy": "per_query_source_procedure_id",
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
        "corpus_stats": seed_stats.to_dict(),
        "query_bank_stats": {
            "num_queries_total": query_bank_total,
            "num_queries_evaluated": len(queries),
            "max_queries": args.max_queries,
            "source_holdout_scope": "per_query",
        },
        "overall_metrics": eval_result.overall_metrics,
        "source_category_stratified_metrics": (
            eval_result.source_category_stratified_metrics
        ),
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
        corpus_size=corpus_size,
        num_queries=len(queries),
        execution_time=elapsed,
        overall=eval_result.overall_metrics,
        k_values=k_values,
    )
    print(f"\nSaved results: {output_path}")


if __name__ == "__main__":
    main()
