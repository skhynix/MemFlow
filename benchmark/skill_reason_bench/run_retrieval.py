#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run SkillReason-Bench retrieval evaluation.

Usage:
    uv run benchmark/skill_reason_bench/run_retrieval.py \
        --user-id benchmark \
        --k-values 1 5 10 20 \
        --max-queries 100

Reports the general binary IR view plus the **official protocol block**
(Recall@{1,5,10,20}, NDCG@10, MRR@20) computed exactly like the dataset's own
evaluate.py. Retrieval always fetches at least top-20 so the official windows
are complete regardless of --k-values, and a companion
``*_predictions.jsonl`` in the official format is written next to the results
JSON for re-scoring with the upstream script.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add repo root to path before other imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

from benchmark.skill_reason_bench.adapter import (  # noqa: E402
    SKILL_REASON_QUERY_INSTRUCTION,
    MemFlowSkillReasonAdapter,
)
from benchmark.skill_reason_bench.download_data import (  # noqa: E402
    verify_data_file_checksum,
)
from benchmark.skill_reason_bench.evaluation import (  # noqa: E402
    OFFICIAL_MRR_K,
    SkillReasonEvaluationResult,
    SkillReasonQuery,
    count_query_records,
    evaluate_skill_reason_queries,
    evaluate_skill_reason_queries_async,
    load_skill_reason_queries,
    write_official_predictions_jsonl,
)
from memflow import MemFlow  # noqa: E402

# Default paths relative to this script location
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_QUERIES_PATH = (
    SCRIPT_DIR / "data" / "skillreason-bench" / "data" / "queries.jsonl"
)
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent.parent / "results"


def _load_env_file(env_path: str | None = None) -> None:
    """Load environment variables from .env file (.env overrides the shell).

    ``load_dotenv(..., override=True)`` makes .env win over variables already
    exported in the shell — so a stale .env can silently change a run; export
    the intended value in the shell AND keep .env consistent, or edit .env.
    """
    path = Path(env_path or ".env")
    if not path.exists():
        return
    load_dotenv(dotenv_path=path, override=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval on the SkillReason-Bench query bank."
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help=f"Path to queries.jsonl (default: {DEFAULT_QUERIES_PATH})",
    )
    parser.add_argument(
        "--user-id",
        default="benchmark",
        help="User ID for seeded skills (default: benchmark)",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20],
        help="K values for the binary IR view (default: 1 5 10 20)",
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
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help=(
            "Skip the stats.json sha256 re-check of queries.jsonl "
            "(it runs by default when stats.json sits beside the file)."
        ),
    )
    return parser.parse_args()


def _normalized_k_values(raw_values: list[int]) -> list[int]:
    return sorted({k for k in raw_values if k > 0})


def _install_query_instruction() -> str:
    """Pin the official SkillReason query instruction via env, pre-store-init.

    The embedding store reads ``VECTOR_EMBEDDING_QUERY_INSTRUCTION`` once, at
    construction, and formats queries as ``Instruct: <instruction>\\nQuery:
    <request>`` (Qwen3-Embedding asymmetric form). Documents are seeded WITHOUT
    a prefix, so only the query side is prefixed here and corpus embeddings stay
    valid — no re-seeding.

    Must be called AFTER ``.env`` is loaded but BEFORE ``MemFlow()`` is built,
    because the store caches the value at init. An explicit
    ``VECTOR_EMBEDDING_QUERY_INSTRUCTION`` (from .env or the shell) is honored;
    otherwise the built-in constant is installed. Returns the effective string
    so callers can log it.
    """
    instruction = (
        os.environ.get("VECTOR_EMBEDDING_QUERY_INSTRUCTION", "").strip()
        or SKILL_REASON_QUERY_INSTRUCTION
    )
    os.environ["VECTOR_EMBEDDING_QUERY_INSTRUCTION"] = instruction
    return instruction


def _results_path(results_dir: Path, filename: str | None) -> Path:
    if filename:
        final_name = filename if filename.endswith(".json") else f"{filename}.json"
        return results_dir / final_name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return results_dir / f"skillreason_retrieval_{stamp}.json"


def _print_summary(
    system_info: dict[str, Any],
    num_queries: int,
    overall_metrics: dict[str, Any],
    official_metrics: dict[str, Any],
    k_values: list[int],
) -> None:
    print("\n=== SkillReason-Bench Retrieval Summary ===")
    print(f"Backend: {system_info.get('backend')}")
    print(f"Queries evaluated: {num_queries}")
    print(f"MRR: {overall_metrics.get('mrr', 0.0):.4f}")
    print(f"MAP: {overall_metrics.get('map', 0.0):.4f}")
    for k in k_values:
        sk = str(k)
        print(
            f"  @{sk}:  hit={overall_metrics['hit_at_k'][sk]:.4f}  "
            f"ndcg={overall_metrics['ndcg_at_k'][sk]:.4f}  "
            f"recall={overall_metrics['recall_at_k'][sk]:.4f}"
        )
    print("Official protocol block (matches upstream evaluate.py):")
    print(
        f"  Recall@1={official_metrics['Recall@1']:.4f}  "
        f"Recall@5={official_metrics['Recall@5']:.4f}  "
        f"Recall@10={official_metrics['Recall@10']:.4f}  "
        f"Recall@20={official_metrics['Recall@20']:.4f}"
    )
    print(
        f"  NDCG@10={official_metrics['NDCG@10']:.4f}  "
        f"MRR@20={official_metrics['MRR@20']:.4f}"
    )


def _ranked_ids_by_query(
    query_results: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Extract official-format ranked skill ID lists from query results."""
    return {
        str(entry["query_id"]): [
            str(item["procedure_id"]) for item in entry["retrieved"]
        ]
        for entry in query_results
    }


def run_evaluation(
    *,
    memflow: MemFlow,
    args: argparse.Namespace,
    backend: str,
) -> tuple[
    SkillReasonEvaluationResult,
    MemFlowSkillReasonAdapter,
    list[SkillReasonQuery],
    int,
    float,
]:
    """Shared evaluation core used by run_retrieval and the combined runner."""
    k_values = _normalized_k_values(args.k_values)
    if not k_values:
        raise SystemExit("--k-values must contain at least one positive integer")
    # The official protocol needs rank positions up to 20 (MRR@20/NDCG@10),
    # so fetch deeper than the requested cutoffs when they are smaller.
    fetch_k = max(max(k_values), OFFICIAL_MRR_K)

    print("\n=== Starting SkillReason-Bench Retrieval Evaluation ===")
    print(f"Queries: {args.queries_path}")
    print(f"User ID: {args.user_id}")
    print(f"K values: {k_values} (fetch depth: {fetch_k})")

    # The official query instruction was pinned into the environment before
    # MemFlow construction (see _install_query_instruction), so the store
    # already embeds queries with it.
    print(
        "Query instruction: "
        f"{os.environ.get('VECTOR_EMBEDDING_QUERY_INSTRUCTION', '')!r}"
    )

    start = time.perf_counter()

    queries = load_skill_reason_queries(
        queries_path=args.queries_path,
        max_queries=args.max_queries,
    )

    # Server-side count; listing full payloads just to count them transfers
    # every skill body over the wire.
    corpus_size = memflow.store.count(user_id=args.user_id)
    if corpus_size == 0:
        print("\n[WARNING] No skills found in corpus. Run run_seeding.py first.")
        sys.exit(1)
    print(f"Corpus size: {corpus_size}")

    retrieval_system = MemFlowSkillReasonAdapter(
        memflow=memflow,
        user_id=args.user_id,
        corpus_size=corpus_size,
        backend=backend,
    )

    if args.sync:
        eval_result = evaluate_skill_reason_queries(
            retrieval_system=retrieval_system,
            queries=queries,
            k_values=k_values,
            top_k=fetch_k,
        )
    else:
        import asyncio

        eval_result = asyncio.run(
            evaluate_skill_reason_queries_async(
                retrieval_system=retrieval_system,
                queries=queries,
                k_values=k_values,
                top_k=fetch_k,
                max_concurrency=args.max_concurrency,
            )
        )

    elapsed = time.perf_counter() - start
    return eval_result, retrieval_system, queries, corpus_size, elapsed


def main() -> None:
    _load_env_file(str(Path(__file__).resolve().parents[2] / ".env"))
    args = _parse_args()

    verify_data_file_checksum(args.queries_path, skip_verify=args.skip_verify)

    backend = os.environ.get("MEMFLOW_BACKEND", "qdrant").strip().lower()

    # Must run before MemFlow(): the store caches the instruction at init.
    _install_query_instruction()

    memflow = MemFlow()
    eval_result, retrieval_system, queries, corpus_size, elapsed = run_evaluation(
        memflow=memflow,
        args=args,
        backend=backend,
    )

    system_name = retrieval_system.get_system_name()
    system_info = retrieval_system.get_system_info()
    k_values = _normalized_k_values(args.k_values)

    results_payload = {
        "benchmark_name": "skill_reason_bench",
        "run_mode": "retrieval_evaluation",
        "system_name": system_name,
        "system_info": system_info,
        "settings": {
            "user_id": args.user_id,
            "k_values": k_values,
            "queries_path": str(args.queries_path),
            "max_queries": args.max_queries,
            "backend": backend,
            "embedding_model": os.environ.get("VECTOR_EMBEDDING_MODEL", ""),
            "sync_mode": args.sync,
            "max_concurrency": args.max_concurrency,
        },
        "corpus_stats": {
            "corpus_size": corpus_size,
        },
        "query_bank_stats": {
            "num_queries_total": count_query_records(args.queries_path),
            "num_queries_evaluated": len(queries),
            "max_queries": args.max_queries,
        },
        "overall_metrics": eval_result.overall_metrics,
        "official_metrics": eval_result.official_metrics,
        "domain_stratified_metrics": eval_result.domain_stratified_metrics,
        "execution_time_seconds": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_results": eval_result.query_results,
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
        system_info=system_info,
        num_queries=len(queries),
        overall_metrics=eval_result.overall_metrics,
        official_metrics=eval_result.official_metrics,
        k_values=k_values,
    )
    print(f"\nSaved results: {output_path}")
    print(f"Saved official-format predictions: {predictions_path}")


if __name__ == "__main__":
    main()
