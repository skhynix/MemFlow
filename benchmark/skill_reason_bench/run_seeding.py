#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Seed the SkillReason-Bench corpus into MemFlow.

Usage:
    uv run benchmark/skill_reason_bench/run_seeding.py \
        --user-id benchmark \
        --clear-existing --embedding-batch-size 256

``--corpus-path`` is the single ``skills.jsonl`` downloaded by
``download_data.py`` (a directory of JSONL files also works for smoke subsets).
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
import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

from benchmark.skill_reason_bench.adapter import (  # noqa: E402
    seed_skill_reason_corpus,
)
from benchmark.skill_reason_bench.download_data import (  # noqa: E402
    verify_data_file_checksum,
)
from memflow import MemFlow  # noqa: E402

# Default paths relative to this script location
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = (
    SCRIPT_DIR / "data" / "skillreason-bench" / "data" / "skills.jsonl"
)
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent.parent / "results"

# Concurrency / batching defaults
DEFAULT_MAX_BATCHES = 100
DEFAULT_MAX_WORKERS = 48


def _load_env_file(env_path: str | None = None) -> None:
    """Load environment variables from .env file (.env overrides the shell)."""
    path = Path(env_path or ".env")
    if not path.exists():
        return
    load_dotenv(dotenv_path=path, override=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed the SkillReason-Bench corpus into MemFlow"
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=(
            f"Path to skills.jsonl (or a directory of JSONL files) "
            f"(default: {DEFAULT_CORPUS_PATH})"
        ),
    )
    parser.add_argument(
        "--user-id",
        default="benchmark",
        help="User ID for seeded procedures (default: benchmark)",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete existing procedures before seeding (default: False)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous sequential mode (default: False)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=DEFAULT_MAX_BATCHES,
        help=f"Max number of procedures per batch for seeding (default: {DEFAULT_MAX_BATCHES})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Max concurrent workers (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum number of records to seed (default: all)",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help=(
            "Number of texts sent per embedding API request (default: 64). "
            "Larger values use the batch embedding endpoint more efficiently; "
            "reduce if the embedding server rejects large batches or times out."
        ),
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
        "--skip-verify",
        action="store_true",
        help=(
            "Skip the stats.json sha256 re-check before seeding "
            "(it runs by default when stats.json sits beside the corpus)."
        ),
    )
    return parser.parse_args()


def _results_path(results_dir: Path, filename: str | None) -> Path:
    if filename:
        final_name = filename if filename.endswith(".json") else f"{filename}.json"
        return results_dir / final_name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return results_dir / f"skillreason_seeding_{stamp}.json"


def _print_summary(
    backend: str,
    corpus_size: int,
    execution_time: float,
    seed_stats: dict[str, Any],
) -> None:
    print("\n=== SkillReason-Bench Corpus Seeding Summary ===")
    print(f"Backend: {backend}")
    print(f"Corpus size: {corpus_size}")
    print(f"Seeded: {seed_stats.get('num_seeded', 0)}")
    print(f"Reused: {seed_stats.get('num_reused', 0)}")
    print(f"Truncated: {seed_stats.get('num_truncated', 0)}")
    print(f"Skipped: {seed_stats.get('num_skipped', 0)}")
    print(f"Deleted: {seed_stats.get('num_deleted', 0)}")
    print(f"Missing after seeding: {seed_stats.get('num_missing', 0)}")
    print(f"Execution time: {execution_time:.3f}s")


def main() -> None:
    _load_env_file(str(Path(__file__).resolve().parents[2] / ".env"))
    args = _parse_args()

    backend = os.environ.get("MEMFLOW_BACKEND", "qdrant").strip().lower()

    print("\n=== Starting SkillReason-Bench Corpus Seeding ===")
    print(f"Corpus path: {args.corpus_path}")
    print(f"User ID: {args.user_id}")
    print(f"Clear existing: {args.clear_existing}")
    if args.sync:
        print("Mode: synchronous (sequential) processing")
    else:
        print("Mode: async parallel processing")
    print(f"Max batches: {args.max_batches}, Max workers: {args.max_workers}")

    verify_data_file_checksum(args.corpus_path, skip_verify=args.skip_verify)

    memflow = MemFlow()

    start = time.perf_counter()

    seed_stats = asyncio.run(
        seed_skill_reason_corpus(
            memflow=memflow,
            user_id=args.user_id,
            corpus_path=args.corpus_path,
            clear_existing=args.clear_existing,
            batch_size=args.max_batches,
            max_records=args.max_records,
            max_workers=args.max_workers,
            use_sync=args.sync,
            embedding_batch_size=args.embedding_batch_size,
        )
    )

    elapsed = time.perf_counter() - start
    corpus_size = seed_stats.active_corpus_size

    seed_stats_payload = seed_stats.to_dict()
    results_payload = {
        "benchmark_name": "skill_reason_bench",
        "run_mode": "seeding",
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
            "embedding_model": os.environ.get("VECTOR_EMBEDDING_MODEL", ""),
            "sync_mode": args.sync,
            "max_batches": args.max_batches,
            "max_workers": args.max_workers,
            "max_records": args.max_records,
            "embedding_batch_size": args.embedding_batch_size,
        },
        "corpus_stats": seed_stats_payload,
        "execution_time_seconds": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = _results_path(results_dir, args.results_filename)
    output_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    _print_summary(
        backend=backend,
        corpus_size=corpus_size,
        execution_time=elapsed,
        seed_stats=seed_stats_payload,
    )
    print(f"\nSaved results: {output_path}")


if __name__ == "__main__":
    main()
