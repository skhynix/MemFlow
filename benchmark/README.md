# Benchmark

This directory contains benchmark harnesses for evaluating MemFlow.

## Available Benchmarks

### `proced_mem_bench` - Procedural Memory Benchmark

The `proced_mem_bench` subdirectory provides a benchmark harness that evaluates **MemFlow retrieval** against raw gold relevance labels from the `procedural_memory_benchmark` query bank.

- **Location**: `benchmark/proced_mem_bench/`
- **Source**: Based on the [skhynix/Proced_mem_bench](https://github.com/skhynix/Proced_mem_bench) fork
- **Custom query banks**: Pass external Proced_mem_bench JSON files with `--query-bank-path`.

For detailed instructions, see [proced_mem_bench/README.md](proced_mem_bench/README.md).

### `wikihow_procedure_silver` - WikiHow Procedure Silver v1

The `wikihow_procedure_silver` subdirectory evaluates **MemFlow retrieval**
against explicit binary relevance sets from the WikiHow Procedure Silver v1
query bank vendored with MemFlow.

- **Location**: `benchmark/wikihow_procedure_silver/`
- **Source**: Kaggle dataset `paolop/human-instructions-dataset-updated-json-files`
- **Data shape**: streamed JSONL procedures plus JSONL query bank records

For detailed instructions, see
[wikihow_procedure_silver/README.md](wikihow_procedure_silver/README.md).

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

Use the install script for easy setup:

```bash
# Install optional Kaggle CLI support for WikiHow downloads
uv sync --extra benchmark

# Install proced_mem_bench
uv run benchmark/install_benchmark.py proced_mem_bench

# Install with specific commit
uv run benchmark/install_benchmark.py proced_mem_bench --commit-hash-procmem f7097bcaf6ca

# Print WikiHow Procedure Silver paths and source instructions
uv run benchmark/install_benchmark.py wikihow_procedure_silver

# Build the local WikiHow procedure corpus from Kaggle raw shards
uv run benchmark/install_benchmark.py wikihow_procedure_silver \
  --raw-dir /path/to/kaggle/raw/wikiHow-json-files

# Install SkillRet benchmark data from HuggingFace (requires git-lfs)
uv run benchmark/install_benchmark.py skill_ret_bench
uv run benchmark/install_benchmark.py skill_ret_bench --commit-hash-skillret <hash>

# Install all benchmark dependencies, building WikiHow if --raw-dir is set
uv run benchmark/install_benchmark.py all \
  --commit-hash-procmem <procmem_hash> \
  --commit-hash-skillret <skillret_hash> \
  --raw-dir /path/to/kaggle/raw
```

## Directory Structure

```
benchmark/
├── install_benchmark.py
├── README.md                    # Overall benchmark guide
├── __init__.py                  # Root package
├── proced_mem_bench/            # Procedural Memory Benchmark
│   ├── README.md                # proced_mem_bench-specific documentation
│   ├── __init__.py
│   ├── adapter.py
│   ├── evaluation.py
│   └── run_proced_mem_bench.py
├── wikihow_procedure_silver/    # WikiHow Procedure Silver v1
│   ├── README.md
│   ├── __init__.py
│   ├── adapter.py
│   ├── build_wikihow_procedures.py
│   ├── evaluation.py
│   ├── benchmark_data/          # Vendored query bank and metadata
│   └── run_wikihow_procedure_silver.py
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

To add a new benchmark, create a subdirectory under `benchmark/` following the structure of `proced_mem_bench/`:

```
benchmark/
└── your_benchmark/
    ├── README.md
    ├── __init__.py
    ├── adapter.py (or equivalent)
    ├── eval.py (or equivalent)
    └── run_benchmark.py
```
