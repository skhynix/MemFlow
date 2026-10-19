#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run the SkillReason-Bench benchmark for MemFlow (seed + evaluate).

Usage:
    uv run benchmark/skill_reason_bench/run_skill_reason.py \
        --user-id benchmark --clear-existing --k-values 1 5 10 20

Pass --seed-only to seed (or reuse) the corpus and skip evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_env_file(env_path: str | None = None) -> None:
    """Load environment variables from .env file (.env overrides the shell)."""
    from dotenv import load_dotenv

    path = Path(env_path or ".env")
    if not path.exists():
        return
    load_dotenv(dotenv_path=path, override=True)


try:
    from memflow import MemFlow
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from memflow import MemFlow

from benchmark.skill_reason_bench.adapter import (  # noqa: E402
    seed_skill_reason_corpus,
)
from benchmark.skill_reason_bench.download_data import (  # noqa: E402
    verify_data_file_checksum,
)
from benchmark.skill_reason_bench.evaluation import (  # noqa: E402
    count_query_records,
    write_official_predictions_jsonl,
)
from benchmark.skill_reason_bench.run_retrieval import (  # noqa: E402
    DEFAULT_QUERIES_PATH,
    _install_query_instruction,
    _normalized_k_values,
    _ranked_ids_by_query,
    run_evaluation,
)
from benchmark.skill_reason_bench.run_seeding import (  # noqa: E402
    DEFAULT_CORPUS_PATH,
)

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MemFlow on the SkillReason-Bench benchmark."
    )
    parser.add_argument("--user-id", default="benchmark")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"Path to skills.jsonl (default: {DEFAULT_CORPUS_PATH})",
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help=f"Path to queries.jsonl (default: {DEFAULT_QUERIES_PATH})",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory for results JSON (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument("--results-filename")
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Seed or reuse the SkillReason corpus and skip evaluation.",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help=(
            "Delete matching existing SkillReason skills for the user before "
            "seeding every valid corpus record. By default, existing IDs are reused."
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum number of corpus records to seed (default: all).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Evaluate only the first N queries for smoke tests.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Number of texts sent per embedding API request (default: 64).",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous sequential mode (default: False)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=48,
        help=(
            "Max concurrent embedding/upsert workers during seeding "
            "(default: 48, matching run_seeding.py)."
        ),
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=64,
        help="Max concurrent queries (default: 64)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help=(
            "Skip the stats.json sha256 re-check of the corpus and query files "
            "(it runs by default when stats.json sits beside them)."
        ),
    )
    return parser.parse_args()


def _results_path(results_dir: Path, filename: str | None) -> Path:
    if filename:
        final_name = filename if filename.endswith(".json") else f"{filename}.json"
        return results_dir / final_name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return results_dir / f"memflow_skill_reason_bench_{stamp}.json"


def _print_summary(
    system_name: str,
    system_info: dict[str, Any],
    corpus_size: int,
    num_queries: int,
    execution_time: float,
    overall: dict[str, Any],
    official: dict[str, Any],
    k_values: list[int],
) -> None:
    print("\n=== SkillReason-Bench Benchmark Summary ===")
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
            f"NDCG@k={overall.get('ndcg_at_k', {}).get(sk, 0.0):.4f}"
        )

    print("Official protocol block (matches upstream evaluate.py):")
    print(
        f"  Recall@1={official['Recall@1']:.4f}  Recall@5={official['Recall@5']:.4f}  "
        f"Recall@10={official['Recall@10']:.4f}  Recall@20={official['Recall@20']:.4f}"
    )
    print(f"  NDCG@10={official['NDCG@10']:.4f}  MRR@20={official['MRR@20']:.4f}")


def _print_seed_summary(
    backend: str,
    corpus_size: int,
    execution_time: float,
    seed_stats: dict[str, Any],
) -> None:
    print("\n=== SkillReason-Bench Seed Summary ===")
    print("Mode: seed-only")
    print(f"Backend: {backend}")
    print(f"Corpus size: {corpus_size}")
    print(f"Seeded: {seed_stats.get('num_seeded', 0)}")
    print(f"Reused: {seed_stats.get('num_reused', 0)}")
    print(f"Truncated: {seed_stats.get('num_truncated', 0)}")
    print(f"Skipped: {seed_stats.get('num_skipped', 0)}")
    print(f"Deleted: {seed_stats.get('num_deleted', 0)}")
    print(f"Missing after seeding: {seed_stats.get('num_missing', 0)}")
    print(f"Execution time (s): {execution_time:.3f}")


def main() -> None:
    _load_env_file(str(Path(__file__).resolve().parents[2] / ".env"))
    args = _parse_args()

    k_values = _normalized_k_values(args.k_values)
    if not args.seed_only and not k_values:
        # CLI argument validation fails as SystemExit (argparse convention),
        # matching run_retrieval.py.
        raise SystemExit("--k-values must contain at least one positive integer")

    verify_data_file_checksum(args.corpus_path, skip_verify=args.skip_verify)
    if not args.seed_only:
        verify_data_file_checksum(args.queries_path, skip_verify=args.skip_verify)

    backend = os.environ.get("MEMFLOW_BACKEND", "qdrant").strip().lower()

    # Must run before MemFlow(): the store caches the instruction at init.
    _install_query_instruction()

    memflow = MemFlow()

    print("\n=== Starting SkillReason-Bench Benchmark ===")
    print(f"Corpus path: {args.corpus_path}")
    print(f"User ID: {args.user_id}")
    print(f"Clear existing: {args.clear_existing}")
    print(f"Mode: {'seed-only' if args.seed_only else 'seed + evaluate'}")

    seed_start = time.perf_counter()
    seed_stats = asyncio.run(
        seed_skill_reason_corpus(
            memflow=memflow,
            user_id=args.user_id,
            corpus_path=args.corpus_path,
            clear_existing=args.clear_existing,
            max_records=args.max_records,
            max_workers=args.max_workers,
            use_sync=args.sync,
            embedding_batch_size=args.embedding_batch_size,
        )
    )
    seed_elapsed = time.perf_counter() - seed_start
    corpus_size = seed_stats.active_corpus_size

    if args.seed_only:
        results_payload = {
            "benchmark_name": "skill_reason_bench",
            "run_mode": "seed_only",
            "system_name": "memflow_seed_skill_reason",
            "system_info": {
                "method": "store.add",
                "backend": backend,
                "user_id": args.user_id,
                "corpus_size": corpus_size,
                "embedding_model": os.environ.get("VECTOR_EMBEDDING_MODEL", ""),
            },
            "settings": {
                "user_id": args.user_id,
                "corpus_path": str(args.corpus_path),
                "clear_existing": args.clear_existing,
                "backend": backend,
                "sync_mode": args.sync,
                "max_workers": args.max_workers,
                "embedding_batch_size": args.embedding_batch_size,
            },
            "corpus_stats": seed_stats.to_dict(),
            "execution_time_seconds": seed_elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        results_dir = args.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = _results_path(results_dir, args.results_filename)
        output_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
        _print_seed_summary(backend, corpus_size, seed_elapsed, seed_stats.to_dict())
        print(f"\nSaved results: {output_path}")
        return

    eval_result, retrieval_system, queries, eval_corpus_size, eval_elapsed = (
        run_evaluation(
            memflow=memflow,
            args=args,
            backend=backend,
        )
    )

    system_name = retrieval_system.get_system_name()
    system_info = retrieval_system.get_system_info()

    results_payload = {
        "benchmark_name": "skill_reason_bench",
        "run_mode": "seed_and_evaluate",
        "system_name": system_name,
        "system_info": system_info,
        "settings": {
            "user_id": args.user_id,
            "k_values": k_values,
            "corpus_path": str(args.corpus_path),
            "queries_path": str(args.queries_path),
            "clear_existing": args.clear_existing,
            "max_records": args.max_records,
            "max_queries": args.max_queries,
            "max_workers": args.max_workers,
            "embedding_batch_size": args.embedding_batch_size,
            "backend": backend,
            "embedding_model": os.environ.get("VECTOR_EMBEDDING_MODEL", ""),
            "sync_mode": args.sync,
            "max_concurrency": args.max_concurrency,
        },
        "corpus_stats": {
            **seed_stats.to_dict(),
            "active_corpus_size": eval_corpus_size,
        },
        "query_bank_stats": {
            "num_queries_total": count_query_records(args.queries_path),
            "num_queries_evaluated": len(queries),
            "max_queries": args.max_queries,
        },
        "seeding_time_seconds": seed_elapsed,
        "evaluation_time_seconds": eval_elapsed,
        "overall_metrics": eval_result.overall_metrics,
        "official_metrics": eval_result.official_metrics,
        "domain_stratified_metrics": eval_result.domain_stratified_metrics,
        "query_results": eval_result.query_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = _results_path(results_dir, args.results_filename)
    output_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    predictions_path = output_path.with_name(
        output_path.name.replace(".json", "_predictions.jsonl")
    )
    write_official_predictions_jsonl(
        predictions_path, _ranked_ids_by_query(eval_result.query_results)
    )

    _print_summary(
        system_name=system_name,
        system_info=system_info,
        corpus_size=eval_corpus_size,
        num_queries=len(queries),
        execution_time=seed_elapsed + eval_elapsed,
        overall=eval_result.overall_metrics,
        official=eval_result.official_metrics,
        k_values=k_values,
    )
    print(f"\nSaved results: {output_path}")
    print(f"Saved official-format predictions: {predictions_path}")


if __name__ == "__main__":
    main()
