#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Install benchmark dependencies.

Usage:
    uv run benchmark/install_benchmark.py proced_mem_bench [--force] [--commit-hash <hash>]
    uv run benchmark/install_benchmark.py all [--force] [--commit-hash <hash>]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROCED_MEM_BENCH_URL = "https://github.com/qpiai/Proced_mem_bench"
PROCED_MEM_BENCH_PATH = "benchmark/proced_mem_bench/Proced_mem_bench"


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=cwd)


def install_proced_mem_bench(force: bool = False, commit_hash: str | None = None) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Install benchmark dependencies")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # proced_mem_bench
    p1 = subparsers.add_parser("proced_mem_bench", help="Install procedural memory benchmark")
    p1.add_argument("--force", action="store_true", help="Force reinstall")
    p1.add_argument("--commit-hash", help="Install specific commit")

    # all
    p2 = subparsers.add_parser("all", help="Install all benchmarks")
    p2.add_argument("--force", action="store_true", help="Force reinstall")
    p2.add_argument("--commit-hash", help="Install specific commit")

    args = parser.parse_args()

    if args.command in ("proced_mem_bench", "all"):
        install_proced_mem_bench(force=args.force, commit_hash=args.commit_hash)

    return 0


if __name__ == "__main__":
    sys.exit(main())
