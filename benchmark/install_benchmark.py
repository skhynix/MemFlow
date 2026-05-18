#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Install benchmark dependencies.

Usage:
    uv run benchmark/install_benchmark.py proced_mem_bench [--force] [--commit-hash <hash>]
    uv run benchmark/install_benchmark.py wikihow_procedure_silver [--raw-dir <dir>]
    uv run benchmark/install_benchmark.py all [--force] [--commit-hash <hash>] [--raw-dir <dir>]
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROCED_MEM_BENCH_URL = "https://github.com/skhynix/Proced_mem_bench"
PROCED_MEM_BENCH_PATH = "benchmark/proced_mem_bench/Proced_mem_bench"
WIKIHOW_BENCHMARK_PATH = "benchmark/wikihow_procedure_silver"
WIKIHOW_QUERY_BANK_PATH = "benchmark_data/query_bank.jsonl"
WIKIHOW_CORPUS_OUTPUT_DIR = "data"


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=cwd)


def install_proced_mem_bench(
    force: bool = False, commit_hash: str | None = None
) -> None:
    project_root = Path(__file__).resolve().parent.parent
    bench_path = project_root / PROCED_MEM_BENCH_PATH

    if bench_path.exists():
        if force:
            print(f"Removing existing directory: {bench_path}")
            shutil.rmtree(bench_path)
        else:
            print("Directory exists, updating...")
            run_cmd(["git", "pull"], cwd=bench_path)

    if not bench_path.exists():
        print("Cloning repository...")
        run_cmd(["git", "clone", PROCED_MEM_BENCH_URL, str(bench_path)])

    if commit_hash:
        print(f"Checking out commit {commit_hash}...")
        run_cmd(["git", "checkout", commit_hash], cwd=bench_path)

    print("Installing package...")
    run_cmd(["uv", "pip", "install", "-e", str(bench_path)])

    print("proced_mem_bench installed successfully")


def install_wikihow_procedure_silver(
    raw_dir: str | Path | None = None,
) -> None:
    project_root = Path(__file__).resolve().parent.parent
    if os.fspath(project_root) not in sys.path:
        sys.path.insert(0, os.fspath(project_root))
    from benchmark.wikihow_procedure_silver.build_wikihow_procedures import (
        CorpusBuildError,
        build_wikihow_procedures,
    )

    bench_path = project_root / WIKIHOW_BENCHMARK_PATH
    query_bank_path = bench_path / WIKIHOW_QUERY_BANK_PATH
    corpus_output_dir = bench_path / WIKIHOW_CORPUS_OUTPUT_DIR
    corpus_path = corpus_output_dir / "wikihow_procedures.jsonl"
    bench_path.mkdir(parents=True, exist_ok=True)
    if not query_bank_path.exists():
        raise SystemExit(f"Missing vendored WikiHow query bank: {query_bank_path}")

    if raw_dir is None:
        print("WikiHow Procedure Silver query bank is vendored locally.")
        print(f"  query bank: {os.fspath(query_bank_path)}")
        print("  install CLI: uv sync --extra benchmark")
        print("  source: uv run kaggle datasets download \\")
        print("          -d paolop/human-instructions-dataset-updated-json-files \\")
        print("          -p benchmark/wikihow_procedure_silver/raw --unzip")
        print(
            "  corpus: not built. Provide --raw-dir pointing to Kaggle "
            "wikiHow*.json files to build data/wikihow_procedures.jsonl."
        )
        return

    print("Building WikiHow Procedure Silver corpus from Kaggle raw shards...")
    try:
        manifest = build_wikihow_procedures(
            input_dir=Path(raw_dir),
            output_dir=corpus_output_dir,
        )
    except CorpusBuildError as exc:
        raise SystemExit(f"WikiHow corpus build failed: {exc}") from exc

    print("wikihow_procedure_silver installed successfully")
    print(f"  corpus: {os.fspath(corpus_path)}")
    print(f"  corpus sha256: {manifest['outputs']['procedures']['sha256']}")
    print(f"  query bank: {os.fspath(query_bank_path)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install benchmark dependencies")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # proced_mem_bench
    p1 = subparsers.add_parser(
        "proced_mem_bench", help="Install procedural memory benchmark"
    )
    p1.add_argument("--force", action="store_true", help="Force reinstall")
    p1.add_argument("--commit-hash", help="Install specific commit")

    # wikihow_procedure_silver
    p_wikihow = subparsers.add_parser(
        "wikihow_procedure_silver",
        help="Build WikiHow Procedure Silver v1 local corpus",
    )
    p_wikihow.add_argument(
        "--raw-dir",
        help="Directory containing Kaggle wikiHow*.json raw shards",
    )

    # all
    p2 = subparsers.add_parser("all", help="Install all benchmarks")
    p2.add_argument("--force", action="store_true", help="Force reinstall")
    p2.add_argument("--commit-hash", help="Install specific commit")
    p2.add_argument(
        "--raw-dir",
        help="Directory containing Kaggle wikiHow*.json raw shards",
    )

    args = parser.parse_args()

    if args.command in ("proced_mem_bench", "all"):
        install_proced_mem_bench(force=args.force, commit_hash=args.commit_hash)
    if args.command in ("wikihow_procedure_silver", "all"):
        install_wikihow_procedure_silver(raw_dir=args.raw_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
