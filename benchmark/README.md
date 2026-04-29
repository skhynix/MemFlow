# Benchmark

This directory contains benchmark harnesses for evaluating MemFlow.

## Available Benchmarks

### `proced_mem_bench` - Procedural Memory Benchmark

The `proced_mem_bench` subdirectory provides a benchmark harness that evaluates **MemFlow retrieval** against raw gold relevance labels from the `procedural_memory_benchmark` query bank.

- **Location**: `benchmark/proced_mem_bench/`
- **Source**: Based on the [skhynix/Proced_mem_bench](https://github.com/skhynix/Proced_mem_bench) fork
- **Custom query banks**: Pass external Proced_mem_bench JSON files with `--query-bank-path`.

For detailed instructions, see [proced_mem_bench/README.md](proced_mem_bench/README.md).

## Installation

Use the install script for easy setup:

```bash
# Install proced_mem_bench
uv run benchmark/install_benchmark.py proced_mem_bench

# Install with specific commit
uv run benchmark/install_benchmark.py proced_mem_bench --commit-hash f7097bcaf6ca
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
└── ...                          # Future benchmarks
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
