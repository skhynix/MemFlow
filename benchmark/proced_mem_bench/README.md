# Proced_mem_bench

This directory provides a thin benchmark harness that evaluates **MemFlow retrieval** against raw gold relevance labels from the `procedural_memory_benchmark` query bank.

## What it does

The benchmark flow is intentionally narrow:

1. Load AgentInstruct trajectories from the external `procedural_memory_benchmark` package.
2. Ingest all trajectories into MemFlow using direct `Procedure` ingestion (`manager.add(procedure=...)`).
3. Run retrieval using `MemFlowManager.search()`.
4. Evaluate retrieved rankings against raw query-bank labels (`relevant_trajectories` + `relevance_score`).
5. Print a console summary and save a JSON results file.

## Why direct `procedure=` ingestion + `manager.search()`

This path benchmarks MemFlow retrieval behavior directly and avoids conflating retrieval quality
with LLM extraction/classification. No `messages=` ingestion path is used.
The adapter stores an action-only procedural trace in `Procedure.content` and
intentionally excludes state text from the benchmark payload.

## Setup

### 1. Clone the Procedural Memory Benchmark repository

Clone the external `Procedural_memory_benchmark` repository into the current directory.  You may use `install_benchmark.py` for easy setup:

```bash
uv run benchmark/install_benchmark.py proced_mem_bench --commit-hash 08048752
```

### 2. Install dependencies

Install the `procedural-memory-benchmark` package:

```bash
uv pip install -e benchmark/proced_mem_bench/Proced_mem_bench
```

### 3. Install LLM client packages

An LLM client package is required depending on your `LLM_PROVIDER`:

- `ollama` for `--llm-provider ollama`
- `openai` for `--llm-provider openai-compatible`

## Usage

From repo root, run with default settings:

```bash
uv run benchmark/proced_mem_bench/run_proced_mem_bench.py
```

For available options, see `--help`:

```bash
uv run benchmark/proced_mem_bench/run_proced_mem_bench.py --help
```

## `--clear-existing`

`--clear-existing` is enabled by default. Before seeding, the script lists existing procedures for `--user-id` and deletes only records whose IDs match AgentInstruct trajectory IDs. This avoids duplicate benchmark corpus accumulation across repeated runs (especially with persistent stores like file/memmachine backends).

Disable with `--no-clear-existing`.

## Results JSON

Each run saves a JSON artifact with:

- benchmark mode/system metadata
- resolved settings (excluding secrets)
- corpus/query-bank stats
- overall metrics (`MAP`, `P@k`, `R@k`, `F1@k`, `NDCG@k`)
- complexity-tier stratified metrics
- per-query retrieved ranking, gold labels, and per-query metrics
