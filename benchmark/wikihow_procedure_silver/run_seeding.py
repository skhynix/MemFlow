#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Seed WikiHow Procedure Silver corpus into MemFlow.

This script loads pre-processed WikiHow procedures from a JSONL file
and seeds them into MemFlow storage.

Usage:
    # Basic usage (uses all defaults, async parallel mode)
    uv run benchmark/wikihow_procedure_silver/run_seeding.py

    # Custom corpus path
    uv run benchmark/wikihow_procedure_silver/run_seeding.py \\
        --corpus-path benchmark/wikihow_procedure_silver/data/wikihow_procedures.jsonl

    # Force reseed (clear existing before seeding)
    uv run benchmark/wikihow_procedure_silver/run_seeding.py --clear-existing

    # Synchronous (sequential) mode
    uv run benchmark/wikihow_procedure_silver/run_seeding.py --sync

    # Custom concurrency settings
    uv run benchmark/wikihow_procedure_silver/run_seeding.py \\
        --max-concurrency 32 --batch-size 25
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
    seed_wikihow_corpus,
)

# Default paths relative to this script location
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = SCRIPT_DIR / "data" / "wikihow_procedures.jsonl"
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
        description="Seed WikiHow Procedure Silver corpus into MemFlow."
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"Path to wikihow_procedures.jsonl (default: {DEFAULT_CORPUS_PATH})",
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
        "--max-concurrency",
        type=int,
        default=64,
        help="Max concurrent workers (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for parallel processing (default: 50)",
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
    return parser.parse_args()


def _results_path(results_dir: Path, filename: str | None) -> Path:
    if filename:
        final_name = filename if filename.endswith(".json") else f"{filename}.json"
        return results_dir / final_name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return results_dir / f"wikihow_seeding_{stamp}.json"


def _print_summary(
    backend: str,
    corpus_size: int,
    execution_time: float,
    seed_stats: dict[str, Any],
) -> None:
    print("\n=== WikiHow Corpus Seeding Summary ===")
    print(f"Backend: {backend}")
    print(f"Corpus size: {corpus_size}")
    print(f"Seeded: {seed_stats.get('num_seeded', 0)}")
    print(f"Reused: {seed_stats.get('num_reused', 0)}")
    print(f"Skipped: {seed_stats.get('num_skipped', 0)}")
    print(f"Deleted: {seed_stats.get('num_deleted', 0)}")
    print(f"Execution time: {execution_time:.3f}s")


def main() -> None:
    _load_env_file()
    args = _parse_args()

    backend = os.environ.get("MEMFLOW_BACKEND", "emulated")
    llm_provider = os.environ.get("LLM_PROVIDER", "ollama")
    llm_model = os.environ.get("LLM_MODEL", "llama3.2")

    print("\n=== Starting WikiHow Corpus Seeding ===")
    print(f"Corpus path: {args.corpus_path}")
    print(f"User ID: {args.user_id}")
    print(f"Clear existing: {args.clear_existing}")
    if args.sync:
        print("Mode: synchronous (sequential) processing")
    else:
        print("Mode: async parallel processing")
    print(f"Max concurrency: {args.max_concurrency}, Batch size: {args.batch_size}")

    memflow = MemFlow(sync_mode=args.sync)
    start = time.perf_counter()

    seed_stats = seed_wikihow_corpus(
        memflow=memflow,
        user_id=args.user_id,
        corpus_path=args.corpus_path,
        clear_existing=args.clear_existing,
        batch_size=args.batch_size,
    )
    corpus_size = seed_stats.active_corpus_size

    elapsed = time.perf_counter() - start
    seed_stats_payload = seed_stats.to_dict()

    results_payload = {
        "benchmark_name": "wikihow_procedure_silver_v1",
        "run_mode": "seeding",
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
            "corpus_path": str(args.corpus_path),
            "clear_existing": args.clear_existing,
            "backend": backend,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "sync_mode": args.sync,
            "max_concurrency": args.max_concurrency,
            "batch_size": args.batch_size,
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
