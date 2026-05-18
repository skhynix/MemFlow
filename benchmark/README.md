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

## Installation

Use the install script for easy setup:

```bash
# Install optional Kaggle CLI support for WikiHow downloads
uv sync --extra benchmark

# Install proced_mem_bench
uv run benchmark/install_benchmark.py proced_mem_bench

# Install with specific commit
uv run benchmark/install_benchmark.py proced_mem_bench --commit-hash f7097bcaf6ca

# Print WikiHow Procedure Silver paths and source instructions
uv run benchmark/install_benchmark.py wikihow_procedure_silver

# Build the local WikiHow procedure corpus from Kaggle raw shards
uv run benchmark/install_benchmark.py wikihow_procedure_silver \
  --raw-dir /path/to/kaggle/raw/wikiHow-json-files

# Install all benchmark dependencies, building WikiHow if --raw-dir is set
uv run benchmark/install_benchmark.py all
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
