# Benchmark

This directory contains benchmark harnesses for evaluating MemFlow.

## Available Benchmarks

### `skill_ret_bench` - SkillRet Benchmark

The `skill_ret_bench` subdirectory evaluates **MemFlow retrieval** against the
SkillRet query bank for skill retrieval evaluation.

- **Location**: `benchmark/skill_ret_bench/`
- **Source**: HuggingFace dataset [`ThakiCloud/SKILLRET`](https://huggingface.co/datasets/ThakiCloud/SKILLRET)
- **Data shape**: JSONL skills (test split) as corpus, JSONL queries (test split) with merged `skill_ids` as query bank

For detailed instructions, see [skill_ret_bench/README.md](skill_ret_bench/README.md).

### `skill_reason_bench` - SkillReason-Bench (implicit requests)

The `skill_reason_bench` subdirectory evaluates **MemFlow retrieval** against
SkillReason-Bench, whose 3,729 queries are concise, **implicit** user requests
over a cross-domain corpus of 61,228 skills — each query has exactly one gold
skill, so the harness reports single-gold metrics and reproduces the dataset's
official protocol (query instruction, 2,048-token document cap,
Recall@{1,5,10,20}/NDCG@10/MRR@20).

- **Location**: `benchmark/skill_reason_bench/`
- **Source**: HuggingFace dataset `donghongjiang/skillreason-bench`
- **Data shape**: single `skills.jsonl` corpus plus `queries.jsonl` with gold
  `positive_ids` and checksums in `stats.json`

For detailed instructions, see
[skill_reason_bench/README.md](skill_reason_bench/README.md).

## Installation

Install the optional plotting dependencies:

```bash
uv sync --extra benchmark

# Install SkillRet data (requires git-lfs); all installs the same dataset.
uv run benchmark/install_benchmark.py skill_ret_bench
uv run benchmark/install_benchmark.py all --commit-hash-skillret <hash>

# SkillReason uses its separate downloader.
uv run benchmark/skill_reason_bench/download_data.py

# Plot existing SkillReason results without repeating retrieval.
uv run --extra benchmark benchmark/skill_reason_bench/plot_recall_curves.py \
  --results results/skillreason.json --by-domain
```

The `benchmark` extra provides matplotlib. SkillReason downloads through
`huggingface_hub` when separately installed, with a standard-library HTTPS
fallback when it is unavailable.

## Directory Structure

```
benchmark/
├── install_benchmark.py
├── README.md                    # Overall benchmark guide
├── __init__.py                  # Root package
├── skill_ret_bench/            # SkillRet Benchmark
│   ├── README.md
│   ├── __init__.py
│   ├── adapter.py
│   ├── evaluation.py
│   ├── run_skill_ret_bench.py
│   ├── run_seeding.py
│   ├── run_retrieval.py        # Retrieval eval (env path or --mode dense/sparse/hybrid)
│   └── data/                   # Cloned from HuggingFace (gitignored)
├── skill_reason_bench/          # SkillReason-Bench (implicit requests)
│   ├── README.md
│   ├── __init__.py
│   ├── adapter.py
│   ├── evaluation.py
│   ├── download_data.py
│   ├── plot_recall_curves.py
│   ├── run_seeding.py
│   ├── run_retrieval.py
│   └── run_skill_reason.py
└── results/                     # Benchmark outputs (gitignored)
```

## Adding New Benchmarks

To add a new benchmark, create a subdirectory under `benchmark/` following the structure of `skill_ret_bench/`:

```
benchmark/
└── your_benchmark/
    ├── README.md
    ├── __init__.py
    ├── adapter.py (or equivalent)
    ├── eval.py (or equivalent)
    └── run_benchmark.py
```
