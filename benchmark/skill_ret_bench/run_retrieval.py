#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run SkillRet retrieval evaluation.

Default path evaluates ``MemFlow.search`` (env-routed hybrid/reranker).
With ``--mode`` the search arguments are controlled explicitly per query via
``store.search_hybrid(mode=...)`` instead — no monkey-patching, and the
reranker pipeline is bypassed on that path.

Usage:
    uv run benchmark/skill_ret_bench/run_retrieval.py \
        --query-bank-path data/SKILLRET/data/queries/test.jsonl \
        --user-id benchmark \
        --k-values 1 3 5 10 \
        --max-queries 100

    # Force search modes (per-mode results JSON; tuning knobs exposed):
    uv run benchmark/skill_ret_bench/run_retrieval.py \
        --mode dense sparse hybrid \
        --prefetch-top-k 200 --rrf-weights 0.6,0.4 --rrf-k 60 \
        --k-values 10 20 30 50 --max-queries 4392
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

from benchmark.skill_ret_bench.adapter import MemFlowSkillRetAdapter  # noqa: E402
from benchmark.skill_ret_bench.evaluation import (  # noqa: E402
    count_query_bank_records,
    evaluate_skill_ret_queries,
    evaluate_skill_ret_queries_async,
    load_skill_ret_query_bank,
)
from memflow import MemFlow  # noqa: E402

# Default paths relative to this script location
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_QUERY_BANK_PATH = (
    SCRIPT_DIR / "data" / "SKILLRET" / "data" / "queries" / "test.jsonl"
)
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent.parent / "results"

ALL_MODES = ("dense", "sparse", "hybrid")

# Reranker env keys snapshotted into results JSON (self-describing runs).
_RERANKER_ENV_KEYS = (
    "MEMFLOW_HYBRID_SEARCH_ENABLED",
    "MEMFLOW_HYBRID_RRF_TOP_K",
    "MEMFLOW_RERANKER_PIPELINE",
    "MEMFLOW_RERANKER_FETCH_TOP_K",
    "MEMFLOW_RERANKER_SCOPE",
    "MEMFLOW_RERANKER_CE_SCOPE",
    "MEMFLOW_RERANKER_LLM_SCOPE",
    "MEMFLOW_RERANKER_DOC_CAP",
    "MEMFLOW_RERANKER_TIMEOUT",
    "MEMFLOW_RERANKER_CE_TOP_K",
    "MEMFLOW_RERANKER_LLM_TOP_K",
    "RERANKER_CE_MODEL",
    "RERANKER_CE_API_BASE",
    "RERANKER_CE_QUERY_INSTRUCTION",
    "RERANKER_LLM_PROVIDER",
    "RERANKER_LLM_MODEL",
    "RERANKER_LLM_API_BASE",
)


def _reranker_effective() -> dict[str, Any]:
    """Effective reranker config after env fallbacks.

    The raw env snapshot records what was *set*; RERANKER_LLM_* fall back to
    LLM_* and stage scopes fall back through the global scope, so the
    actually-used values need resolving here too (mirrors the logic in
    memflow.reranker).
    """
    from memflow.reranker import resolve_stage_scope

    pipeline = (os.environ.get("MEMFLOW_RERANKER_PIPELINE") or "none").lower()
    llm_provider = os.environ.get("RERANKER_LLM_PROVIDER") or os.environ.get(
        "LLM_PROVIDER", "ollama"
    )
    llm_model = os.environ.get("RERANKER_LLM_MODEL") or os.environ.get("LLM_MODEL")
    effective: dict[str, Any] = {
        "pipeline": pipeline,
        "ce_top_k": int(os.environ.get("MEMFLOW_RERANKER_CE_TOP_K") or 10),
        "llm_top_k": int(os.environ.get("MEMFLOW_RERANKER_LLM_TOP_K") or 10),
        "ce_scope": resolve_stage_scope("ce"),
        "llm_scope": resolve_stage_scope("llm"),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
    }
    if pipeline != "none" and ("llm" in pipeline):
        if llm_provider == "openai-compatible" and not llm_model:
            effective["llm_warning"] = (
                "RERANKER_LLM_MODEL/LLM_MODEL unset — openai-compatible LLM "
                "reranker cannot be built; pipeline degrades to no-op at runtime"
            )
    return effective


def _load_env_file(env_path: str | None = None) -> None:
    """Load environment variables from .env file."""
    path = Path(env_path or ".env")
    if not path.exists():
        return
    load_dotenv(dotenv_path=path, override=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval on SkillRet query bank."
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
    # --- Explicit search-mode path (bypasses env-routed MemFlow.search) ---
    parser.add_argument(
        "--mode",
        nargs="+",
        default=None,
        choices=ALL_MODES,
        help="Force store.search_hybrid mode(s); one results JSON per mode. "
        "Default (unset) evaluates the env-routed MemFlow.search path "
        "(hybrid + reranker as configured).",
    )
    parser.add_argument(
        "--prefetch-top-k",
        type=int,
        default=200,
        help="Sparse AND dense prefetch depth for --mode (default: 200). "
        "Overridden by --sparse-top-k/--dense-top-k when either is given.",
    )
    parser.add_argument(
        "--sparse-top-k",
        type=int,
        default=None,
        help="Sparse prefetch depth for --mode (default: --prefetch-top-k)",
    )
    parser.add_argument(
        "--dense-top-k",
        type=int,
        default=None,
        help="Dense prefetch depth for --mode (default: --prefetch-top-k)",
    )
    parser.add_argument(
        "--rrf-weights",
        default="0.6,0.4",
        help='RRF weights "dense_w,sparse_w" for --mode (default: 0.6,0.4)',
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant k in score = 1/(k+rank) for --mode (default: 60)",
    )
    parser.add_argument(
        "--rrf-top-k",
        type=int,
        default=None,
        help="RRF fusion output depth for --mode (default: "
        "MEMFLOW_HYBRID_RRF_TOP_K env, else 100 — the same value serving "
        "uses, so benchmark numbers are serving-comparable by default; "
        "Recall@K is capped at this depth and a warning is printed when "
        "--k-values exceed it)",
    )
    return parser.parse_args()


def _parse_rrf_weights(raw: str) -> list[float]:
    """Parse "dense_w,sparse_w" into two non-negative floats."""
    parts = [w.strip() for w in raw.split(",") if w.strip()]
    if len(parts) != 2:
        raise ValueError(f"--rrf-weights expects 2 values, got {len(parts)}: {raw!r}")
    weights = [float(w) for w in parts]
    if any(w < 0 for w in weights):
        raise ValueError(f"--rrf-weights must be non-negative, got {raw!r}")
    return weights


def _normalized_k_values(raw_values: list[int]) -> list[int]:
    return sorted({k for k in raw_values if k > 0})


def _results_path(results_dir: Path, filename: str | None, mode: str | None) -> Path:
    suffix = f"_{mode}" if mode else ""
    if filename:
        base = filename[:-5] if filename.endswith(".json") else filename
        return results_dir / f"{base}{suffix}.json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return results_dir / f"skillret_retrieval_{stamp}{suffix}.json"


def _resolve_mode_params(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve effective search_hybrid params from args (pipeline order).

    ``rrf_top_k`` defaults to the serving value (MEMFLOW_HYBRID_RRF_TOP_K,
    else 100) so benchmark numbers are serving-comparable by default. It is
    never silently raised to max(k_values): fusion output caps Recall@K, and
    quietly serving a deeper pool than production would make benchmark and
    online numbers incomparable. Callers warn when measured K exceeds it.
    """
    sparse_top_k = args.sparse_top_k or args.prefetch_top_k
    dense_top_k = args.dense_top_k or args.prefetch_top_k
    if args.rrf_top_k is not None:
        rrf_top_k = args.rrf_top_k
        rrf_top_k_source = "cli"
    else:
        env_val = os.environ.get("MEMFLOW_HYBRID_RRF_TOP_K")
        try:
            rrf_top_k = int(env_val) if env_val else 100
            rrf_top_k_source = "env" if env_val else "default"
        except ValueError:
            rrf_top_k = 100
            rrf_top_k_source = "default"
    return {
        "sparse_top_k": sparse_top_k,
        "dense_top_k": dense_top_k,
        "rrf_weights": _parse_rrf_weights(args.rrf_weights),
        "rrf_k": args.rrf_k,
        "rrf_top_k": rrf_top_k,
        "rrf_top_k_source": rrf_top_k_source,
    }


def _make_mode_search_fn(memflow: "MemFlow", mode: str, params: dict[str, Any]):
    """Build an adapter ``search_fn`` forcing ``store.search_hybrid(mode=...)``.

    ``top_k`` truncates the fused output exactly like ``MemFlow.search``;
    the reranker pipeline is intentionally bypassed — this path controls the
    search arguments explicitly.
    """

    def search_fn(query, user_id=None, top_k=5, kind="skill"):
        # Same depth semantics as MemFlow.search hybrid path (reranker off):
        # max(RRF_TOP_K, top_k) — fusion never clips the requested top_k.
        candidates = memflow.store.search_hybrid(
            query=query,
            rrf_top_k=max(params["rrf_top_k"], top_k),
            user_id=user_id,
            kind=kind,
            mode=mode,
            rrf_weights=params["rrf_weights"],
            sparse_top_k=params["sparse_top_k"],
            dense_top_k=params["dense_top_k"],
            rrf_k=params["rrf_k"],
        )
        return candidates[:top_k]

    return search_fn


def _print_summary(
    system_name: str,
    system_info: dict[str, Any],
    corpus_size: int,
    num_queries: int,
    execution_time: float,
    overall: dict[str, Any],
    k_values: list[int],
) -> None:
    print("\n=== SkillRet Retrieval Evaluation Summary ===")
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
    # Load .env from repo root (two levels up from this script)
    _load_env_file(str(Path(__file__).resolve().parents[2] / ".env"))
    args = _parse_args()

    k_values = _normalized_k_values(args.k_values)
    if not k_values:
        raise ValueError("--k-values must contain at least one positive integer")
    top_k = max(k_values)

    backend = os.environ.get("MEMFLOW_BACKEND", "emulated")
    llm_provider = os.environ.get("LLM_PROVIDER", "ollama")
    llm_model = os.environ.get("LLM_MODEL", "llama3.2")

    print("\n=== Starting SkillRet Retrieval Evaluation ===")
    print(f"Query bank: {args.query_bank_path}")
    print(f"User ID: {args.user_id}")
    print(f"K values: {k_values}")
    print(f"Max queries: {args.max_queries or 'all'}")
    if args.sync:
        print("Mode: synchronous (sequential) processing")
    else:
        print("Mode: async parallel processing")

    memflow = MemFlow(sync_mode=args.sync)

    # Get corpus size from existing procedures (server-side count; listing
    # full payloads just to count them transfers every skill body over the wire)
    corpus_size = memflow.store.count(user_id=args.user_id)

    if corpus_size == 0:
        print("\n[WARNING] No procedures found in corpus. Run run_seeding.py first.")
        sys.exit(1)

    print(f"Corpus size: {corpus_size}")

    query_bank_total = count_query_bank_records(args.query_bank_path)
    queries = load_skill_ret_query_bank(
        query_bank_path=args.query_bank_path,
        max_queries=args.max_queries,
    )

    # Run list: one pass per forced mode; mode=None = env-routed search path.
    # Dedupe (dict.fromkeys keeps order) so `--mode hybrid hybrid` cannot
    # silently overwrite the same results file twice.
    modes: list[str | None] = list(dict.fromkeys(args.mode)) if args.mode else [None]

    effective = _reranker_effective()
    rerank_pipeline_on = effective["pipeline"] not in ("", "none")
    hybrid_on = (os.environ.get("MEMFLOW_HYBRID_SEARCH_ENABLED") or "").lower() == "on"
    # Sync mode goes through retrieve_batch -> memflow.search(list), and the
    # list branch uses plain dense store.search — bypassing BOTH hybrid and
    # the reranker. Warn loudly and record what actually ran — otherwise the
    # results JSON's env snapshot makes a dense-only run look like a
    # hybrid/rerank evaluation.
    executed_path: str | None = None
    if args.sync and not args.mode and (hybrid_on or rerank_pipeline_on):
        bypassed = []
        if hybrid_on:
            bypassed.append("hybrid search")
        if rerank_pipeline_on:
            bypassed.append("the reranker")
        print(
            f"\n[WARNING] --sync uses the batch query path, which BYPASSES "
            f"{' and '.join(bypassed)} (plain dense retrieval runs instead). "
            "The recorded reranker_env does NOT describe what ran. Use async "
            "(default) to evaluate the env-routed hybrid/rerank path."
        )
        executed_bypassed = []
        if hybrid_on:
            executed_bypassed.append("hybrid")
        if rerank_pipeline_on:
            executed_bypassed.append("reranker")
        executed_path = "dense_batch:bypassed=" + "+".join(executed_bypassed)

    # On the env-routed path, a stage Top-K smaller than max(--k-values)
    # silently caps deep Recall@K at the stage output depth.
    if not args.mode and rerank_pipeline_on:
        pipeline = effective["pipeline"]
        final_stage_cap = (
            effective["llm_top_k"]
            if pipeline.endswith("llm")
            else effective["ce_top_k"]
        )
        if final_stage_cap and top_k > final_stage_cap:
            print(
                f"\n[WARNING] reranker final-stage Top-K={final_stage_cap} "
                f"(pipeline={pipeline}) is below max(--k-values)={top_k} — "
                f"R@K for K>{final_stage_cap} measures the stage cap, not "
                "the system. Raise MEMFLOW_RERANKER_"
                f"{'LLM' if pipeline.endswith('llm') else 'CE'}_TOP_K or "
                "lower --k-values."
            )

    mode_params: dict[str, Any] | None = None
    if args.mode:
        if not getattr(memflow.store, "supports_hybrid", False):
            raise SystemExit(
                f"--mode {args.mode} requires store.search_hybrid; backend "
                f"{backend!r} (or a collection without sparse vectors) does "
                "not support it. Unset --mode to evaluate the env-routed "
                "MemFlow.search path."
            )
        mode_params = _resolve_mode_params(args)
        print("\nExplicit search-mode path (reranker bypassed).")
        print(
            f"prefetch: sparse_top_k={mode_params['sparse_top_k']}  "
            f"dense_top_k={mode_params['dense_top_k']}"
        )
        print(
            f"fusion:   rrf_weights={mode_params['rrf_weights']}  "
            f"rrf_k={mode_params['rrf_k']}  "
            f"rrf_top_k={mode_params['rrf_top_k']} "
            f"(source: {mode_params['rrf_top_k_source']}; serving uses the "
            "same default, MEMFLOW_HYBRID_RRF_TOP_K else 100)"
        )
        if top_k > mode_params["rrf_top_k"]:
            print(
                f"note: max(--k-values)={top_k} exceeds rrf_top_k="
                f"{mode_params['rrf_top_k']} — fusion depth is raised to "
                f"{top_k} per query, matching serving behavior "
                "(max(RRF_TOP_K, top_k)), so deeper R@K stays "
                "serving-comparable."
            )

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        start = time.perf_counter()

        retrieval_system = MemFlowSkillRetAdapter(
            memflow=memflow,
            user_id=args.user_id,
            corpus_size=corpus_size,
            backend=backend,
            llm_provider=llm_provider,
            llm_model=llm_model,
            search_fn=(
                _make_mode_search_fn(memflow, mode, mode_params) if mode else None
            ),
            mode_label=mode,
        )

        if args.sync:
            # Sync mode: batch embedding
            eval_result = evaluate_skill_ret_queries(
                retrieval_system=retrieval_system,
                queries=queries,
                k_values=k_values,
                top_k=top_k,
            )
        else:
            # Async mode: parallel processing
            import asyncio

            eval_result = asyncio.run(
                evaluate_skill_ret_queries_async(
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

        settings: dict[str, Any] = {
            "user_id": args.user_id,
            "k_values": k_values,
            "top_k": top_k,
            "query_bank_path": str(args.query_bank_path),
            "max_queries": args.max_queries,
            "backend": backend,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "sync_mode": args.sync,
            "max_concurrency": args.max_concurrency,
            # Reranker env snapshot — results must be self-describing;
            # past runs were uncomparable because these went unrecorded.
            "reranker_env": {k: os.environ.get(k) for k in _RERANKER_ENV_KEYS},
            # Effective config after env fallbacks (raw env alone cannot show
            # which LLM_MODEL / stage scope was actually in effect).
            "reranker_effective": effective,
        }
        if executed_path is not None:
            # What actually executed differs from the env snapshot (sync
            # batch path bypasses hybrid/reranker) — record it explicitly.
            settings["executed_path"] = executed_path
        if mode_params is not None:
            settings["search_mode"] = mode
            settings["search_hybrid_params"] = dict(mode_params)

        results_payload = {
            "benchmark_name": "skill_ret_bench",
            "run_mode": "retrieval_evaluation",
            "system_name": system_name,
            "system_info": system_info,
            "settings": settings,
            "corpus_stats": {
                "corpus_size": corpus_size,
            },
            "query_bank_stats": {
                "num_queries_total": query_bank_total,
                "num_queries_evaluated": len(queries),
                "max_queries": args.max_queries,
            },
            "overall_metrics": eval_result.overall_metrics,
            "category_stratified_metrics": eval_result.category_stratified_metrics,
            "execution_time_seconds": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_results": eval_result.query_results,
        }

        output_path = _results_path(results_dir, args.results_filename, mode)
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
