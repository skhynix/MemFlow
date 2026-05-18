# WikiHow Procedure Silver v1 Benchmark

This harness evaluates MemFlow retrieval against the WikiHow Procedure Silver v1
query bank vendored in this repository.

The runner is separate from `proced_mem_bench` because the WikiHow corpus uses
plain JSONL procedures and explicit binary relevance sets instead of the
external `procedural_memory_benchmark` package.

This repository tracks the 3 MB query bank and metadata only. It does not
redistribute the full WikiHow-derived procedure corpus. To run retrieval
evaluation, first download the source dataset from Kaggle:

`paolop/human-instructions-dataset-updated-json-files`

## Data

The query bank and metadata are tracked under:

- `benchmark/wikihow_procedure_silver/benchmark_data/`

Install the optional benchmark dependencies to make the Kaggle CLI available,
then download and unzip the Kaggle source shards:

```bash
uv sync --extra benchmark
mkdir -p benchmark/wikihow_procedure_silver/raw
uv run kaggle datasets download \
  -d paolop/human-instructions-dataset-updated-json-files \
  -p benchmark/wikihow_procedure_silver/raw \
  --unzip
```

The Kaggle CLI requires a Kaggle API token, usually at
`~/.kaggle/kaggle.json`. Pass the directory containing `wikiHow*.json`
as `--raw-dir`.

Build the local procedure corpus from the Kaggle raw WikiHow JSON shards:

```bash
uv run benchmark/install_benchmark.py wikihow_procedure_silver \
  --raw-dir benchmark/wikihow_procedure_silver/raw
```

This writes:

- `data/wikihow_procedures.jsonl`
- `data/MANIFEST.json`

The full-corpus build is verified against:

- records: `132157`
- sha256: `d4364ff7dd35ceac71d0f7f86b5c1f2cae2de6120403fdcfc5ca36ec0bad0f8f`

Running the installer without `--raw-dir` only prints the vendored query bank
path and a reminder that the corpus must be built from the Kaggle raw files.

## Run

```bash
uv run benchmark/wikihow_procedure_silver/run_wikihow_procedure_silver.py \
  --corpus-path benchmark/wikihow_procedure_silver/data/wikihow_procedures.jsonl \
  --query-bank-path benchmark/wikihow_procedure_silver/benchmark_data/query_bank.jsonl \
  --results-dir benchmark/results
```

By default, seeding reuses existing WikiHow procedures for the selected
`--user-id`: the runner lists existing procedure IDs and only calls
`memflow.add()` for corpus records that are missing. This supports repeated
benchmark runs without duplicating the corpus.

For a fresh reseed, delete existing procedures whose IDs match the corpus for
that user before adding every valid corpus record:

```bash
uv run benchmark/wikihow_procedure_silver/run_wikihow_procedure_silver.py \
  --corpus-path benchmark/wikihow_procedure_silver/data/wikihow_procedures.jsonl \
  --query-bank-path benchmark/wikihow_procedure_silver/benchmark_data/query_bank.jsonl \
  --results-dir benchmark/results \
  --clear-existing
```

For a smoke test, limit query execution:

```bash
uv run benchmark/wikihow_procedure_silver/run_wikihow_procedure_silver.py \
  --corpus-path benchmark/wikihow_procedure_silver/data/wikihow_procedures.jsonl \
  --query-bank-path benchmark/wikihow_procedure_silver/benchmark_data/query_bank.jsonl \
  --results-dir benchmark/results \
  --max-queries 5
```

The runner loads `.env` before creating `MemFlow`, so backend and LLM settings
follow the same precedence as the existing benchmark harness.

## Metrics

The evaluator uses standard binary IR metrics over each query's explicit
`relevant_procedure_ids` set:

- Hit@k
- Precision@k
- Recall@k
- F1@k
- MRR
- MAP/AP
- NDCG@k with binary gains

Results include settings, corpus stats, query-bank stats, overall metrics,
source-category stratified metrics, and per-query rankings with retrieved
procedure ID, title, category, and score.
