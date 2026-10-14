#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Plot Recall@k curves for a SkillReason-Bench retrieval run.

Reads a retrieval-result JSON produced by run_retrieval.py (run it with
--k-values reaching --k-max so each query's ``retrieved`` list is deep enough)
and recomputes mean Recall@k from the per-query ranked lists, so redrawing a
chart costs no embedding or retrieval calls.

SkillReason-Bench has exactly one gold skill per query, so Recall@k here equals
the hit rate at k; the computation still divides by the gold-set size so it
stays correct if a future split ships multiple golds.

With ``--by-domain`` each of the ten benchmark domains gets its own curve,
which is where the interesting structure lives: an aggregate curve can hide
that, say, ``software`` routes cleanly while ``legal`` never clears Recall@10.

Usage:
    uv run --extra benchmark benchmark/skill_reason_bench/plot_recall_curves.py \
        --results results/skillreason_full_topk100.json \
        --output benchmark/skill_reason_bench/plots/recall_at_k.png

    uv run --extra benchmark benchmark/skill_reason_bench/plot_recall_curves.py \
        --results results/skillreason_full_topk100.json \
        --by-domain \
        --output benchmark/skill_reason_bench/plots/recall_at_k_by_domain.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Recall@k curves for a SkillReason-Bench retrieval run."
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Retrieval-result JSON (run_retrieval.py output).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/skill_reason_bench/plots/recall_at_k.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--by-domain",
        action="store_true",
        help="Draw one curve per query domain instead of a single aggregate curve.",
    )
    parser.add_argument(
        "--k-min", type=int, default=1, help="Smallest k on the curve (default 1)."
    )
    parser.add_argument(
        "--k-max", type=int, default=100, help="Largest k on the curve (default 100)."
    )
    parser.add_argument(
        "--k-step",
        type=int,
        default=5,
        help=(
            "k step, anchored so the official cutoffs 5/10/20 always get a "
            "point (default 5 -> 1, 5, 10, 15, ..., 100)."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Chart title override.",
    )
    return parser.parse_args()


def _k_grid(k_min: int, k_max: int, k_step: int) -> list[int]:
    """k values on a step grid anchored at multiples of ``k_step``.

    Plain ``range(k_min, k_max + 1, k_step)`` walks the grid from ``k_min``,
    so with the default step of 5 starting at 1 the official cutoffs 5, 10,
    and 20 — the points a SkillReason reader looks for first — never land on
    a tick (1, 6, 11, ...). Anchoring on multiples of the step keeps them
    (1, 5, 10, 15, ...) and also lands exactly on ``k_max`` when it is a
    multiple, instead of silently stopping short.
    """
    if k_step < 1:
        raise SystemExit(f"--k-step must be >= 1, got {k_step}")
    if k_max < k_min:
        raise SystemExit(f"--k-max ({k_max}) must be >= --k-min ({k_min})")
    grid = set(range(k_min, k_max + 1)) & set(
        range(k_step * ((k_min + k_step - 1) // k_step), k_max + 1, k_step)
    )
    grid.add(k_min)
    return sorted(grid)


def _recall_series(
    query_results: list[dict[str, Any]], k_values: list[int]
) -> tuple[list[float], int, int]:
    """Recompute mean Recall@k over the given query results.

    Returns (mean_recall_per_k, num_queries_used, depth_of_deepest_ranking).
    Queries with an empty gold set are skipped (0/0 is undefined).
    """
    per_k: list[list[float]] = [[] for _ in k_values]
    used = 0
    max_depth = 0
    for qr in query_results:
        relevant = {str(sid) for sid in qr.get("relevant_skill_ids", []) if sid}
        if not relevant:
            continue
        used += 1
        retrieved: list[str] = []
        seen: set[str] = set()
        for item in qr.get("retrieved", []):
            rid = str(item["procedure_id"])
            if rid not in seen:
                seen.add(rid)
                retrieved.append(rid)
        max_depth = max(max_depth, len(retrieved))
        for i, k in enumerate(k_values):
            if k > len(retrieved):
                # Beyond the fetched depth recall cannot grow; score the
                # truncated window rather than pretending unseen ranks miss.
                topk = retrieved
            else:
                topk = retrieved[:k]
            hits = sum(1 for rid in topk if rid in relevant)
            per_k[i].append(hits / len(relevant))

    means = [(sum(v) / len(v)) if v else 0.0 for v in per_k]
    return means, used, max_depth


def main() -> None:
    args = _parse_args()
    import matplotlib

    matplotlib.use("Agg")  # headless: no display required
    import matplotlib.pyplot as plt

    data = json.loads(args.results.read_text(encoding="utf-8"))
    query_results = data.get("query_results", [])
    if not query_results:
        raise SystemExit(f"No query_results in {args.results}")

    k_values = _k_grid(args.k_min, args.k_max, args.k_step)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if args.by_domain:
        by_domain: dict[str, list[dict[str, Any]]] = {}
        for qr in query_results:
            by_domain.setdefault(str(qr.get("domain", "unknown")), []).append(qr)
        used_total = 0
        depth = 0
        for domain in sorted(by_domain):
            series, n, d = _recall_series(by_domain[domain], k_values)
            used_total += n
            depth = max(depth, d)
            ax.plot(
                k_values,
                series,
                "-o",
                linewidth=1.6,
                markersize=3,
                label=f"{domain} (n={n})",
            )
        title_suffix = "by domain"
        caption_n = used_total
    else:
        series, caption_n, depth = _recall_series(query_results, k_values)
        ax.plot(
            k_values,
            series,
            "-o",
            color="tab:red",
            linewidth=2,
            markersize=4,
            label=f"overall (n={caption_n})",
        )
        title_suffix = "overall"

    system = data.get("system_info", {}) or {}
    settings = data.get("settings", {}) or {}
    # The retrieval model is the embedding model; llm_model is the (unused on
    # this path) chat model, so prefer embedding_model when recorded.
    model = settings.get("embedding_model") or system.get("llm_model") or "?"
    corpus = system.get("corpus_size") or "?"

    ax.set_xlabel("k (top-k retrieved)")
    ax.set_ylabel("Recall@k")
    ax.set_title(
        args.title
        or f"SkillReason-Bench Recall@k ({title_suffix}) — corpus {corpus} skills"
    )
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8 if args.by_domain else 10)

    official = data.get("official_metrics") or {}
    official_note = ""
    if official:
        official_note = (
            f" | official R@1={official.get('Recall@1', 0):.3f} "
            f"R@10={official.get('Recall@10', 0):.3f} "
            f"NDCG@10={official.get('NDCG@10', 0):.3f} "
            f"MRR@20={official.get('MRR@20', 0):.3f}"
        )
    fig.text(
        0.5,
        0.01,
        f"n={caption_n} queries, fetched depth={depth}, model={model}{official_note}",
        ha="center",
        fontsize=8,
        color="gray",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.output, dpi=150)
    print(f"Saved: {args.output}")

    if args.by_domain:
        print(f"{'domain':<14} " + " ".join(f"{k:>6}" for k in k_values))
        for domain in sorted(by_domain):
            s, _, _ = _recall_series(by_domain[domain], k_values)
            print(f"{domain:<14} " + " ".join(f"{v:6.3f}" for v in s))
    else:
        print("   k  | recall")
        for k, v in zip(k_values, series):
            print(f"{k:>4}  | {v:.3f}")


if __name__ == "__main__":
    main()
