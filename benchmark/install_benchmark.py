#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Install benchmark dependencies.

Usage:
    uv run benchmark/install_benchmark.py skill_ret_bench [--force] [--commit-hash-skillret <hash>]
    uv run benchmark/install_benchmark.py all [--force] [--commit-hash-skillret <hash>]
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SKILL_RET_BENCH_URL = "https://huggingface.co/datasets/ThakiCloud/SKILLRET"
SKILL_RET_BENCH_PATH = "benchmark/skill_ret_bench/data/SKILLRET"
SKILL_RET_QUERIES_JSONL = "data/queries/test.jsonl"
SKILL_RET_CORPUS_JSONL = "data/skills/test.jsonl"


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=cwd)


def install_skill_ret_bench(
    force: bool = False, commit_hash: str | None = None
) -> None:
    """Install SkillRet benchmark data from HuggingFace.

    Clones the SKILLRET dataset repository (requires git-lfs). Both the
    corpus (``data/skills/test.jsonl``) and query bank
    (``data/queries/test.jsonl``) are vendored as JSONL and need no
    conversion.
    """
    project_root = Path(__file__).resolve().parent.parent
    bench_path = project_root / SKILL_RET_BENCH_PATH

    if bench_path.exists():
        if force:
            print(f"Removing existing directory: {bench_path}")
            shutil.rmtree(bench_path)
        else:
            print("Directory exists, updating...")
            run_cmd(["git", "pull"], cwd=bench_path)

    if not bench_path.exists():
        print("Cloning HuggingFace dataset repository (requires git-lfs)...")
        run_cmd(["git", "clone", SKILL_RET_BENCH_URL, str(bench_path)])

    if commit_hash:
        print(f"Checking out commit {commit_hash}...")
        run_cmd(["git", "checkout", commit_hash], cwd=bench_path)

    corpus_jsonl = bench_path / SKILL_RET_CORPUS_JSONL
    queries_jsonl = bench_path / SKILL_RET_QUERIES_JSONL
    for label, path in (("corpus", corpus_jsonl), ("queries", queries_jsonl)):
        if not path.exists():
            raise SystemExit(
                f"Expected {label} file not found at {path}. "
                "Ensure git-lfs is installed and the clone succeeded."
            )

    print("skill_ret_bench installed successfully")
    print(f"  corpus: {os.fspath(corpus_jsonl)}")
    print(f"  queries: {os.fspath(queries_jsonl)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install benchmark dependencies")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # skill_ret_bench
    p_skill_ret = subparsers.add_parser(
        "skill_ret_bench",
        help="Install SkillRet benchmark data from HuggingFace",
    )
    p_skill_ret.add_argument("--force", action="store_true", help="Force reinstall")
    p_skill_ret.add_argument(
        "--commit-hash-skillret", help="Install specific commit of SkillRet dataset"
    )

    # all
    p2 = subparsers.add_parser("all", help="Install all benchmarks")
    p2.add_argument("--force", action="store_true", help="Force reinstall")
    p2.add_argument(
        "--commit-hash-skillret", help="Install specific commit of SkillRet dataset"
    )

    args = parser.parse_args()

    if args.command == "skill_ret_bench":
        install_skill_ret_bench(force=args.force, commit_hash=args.commit_hash_skillret)
    if args.command == "all":
        install_skill_ret_bench(force=args.force, commit_hash=args.commit_hash_skillret)

    return 0


if __name__ == "__main__":
    sys.exit(main())
