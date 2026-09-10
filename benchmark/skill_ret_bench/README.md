# SkillRet Benchmark

This directory provides a benchmark harness that evaluates **MemFlow retrieval** against the SkillRet query bank for skill retrieval evaluation.

## What it does

The SkillRet benchmark flow:

1. Load skill corpus from JSONL files
2. Ingest all skills into MemFlow using direct `Procedure` ingestion (`memflow.add(procedure=...)`)
3. Run retrieval using `MemFlow.search()`
4. Evaluate retrieved rankings against gold relevance labels
5. Print a console summary and save a JSON results file

## Dataset

This benchmark uses the **SkillRet** dataset from HuggingFace:
- **Repository**: https://huggingface.co/datasets/ThakiCloud/SKILLRET
- **License**: Apache-2.0 (benchmark), MIT/Apache-2.0 (source skills)
- **Size**: 6,006 held-out candidate skills and 4,392 evaluation queries
  (17,810 skills and 63,259 train queries are included in the complete release)

Seeding only the 6,006-skill test split (instead of the full 17,810-skill library)
and the evaluation split's query count (4,392 vs 4,997 in prior runs) both affect
retrieval scores, so comparisons across corpus/query configurations are not
like-for-like.

The dataset ships JSONL files for all three subsets, each split into `train` and `test`:

```
data/
├── skills.jsonl           # Full skill library (17,810 skills, not used)
├── skills/
│   ├── train.jsonl        # 10,123 skills
│   └── test.jsonl         # 6,006 skills (corpus)
├── queries/
│   ├── train.jsonl        # 63,259 queries
│   └── test.jsonl         # 4,392 queries (query bank)
├── qrels/
│   ├── train.jsonl
│   └── test.jsonl         # Binary relevance labels (already merged into queries)
└── taxonomy.json
```

The `test` splits are used for evaluation. Train and test skill sets are disjoint (zero overlap), so seeding only test skills is sufficient — every `skill_ids` reference in `queries/test.jsonl` points to a skill present in `skills/test.jsonl`.

### Schema

**Skills (`skills/test.jsonl`):**
- `id`: Skill ID
- `name`: Skill name
- `namespace`: Skill namespace (author/repo)
- `description`: Short description
- `skill_md`: Full markdown content
- `major`, `sub`: Taxonomy categories (6 major, 18 sub)
- `author`, `stars`, `installs`, `license`, `repo`: Metadata

**Queries (`queries/test.jsonl`):**
- `id`: Query ID
- `query`: Natural language request
- `skill_ids`: List of relevant skill IDs (ground truth, merged from qrels)
- `k`: Number of relevant skills

**Qrels (`qrels/test.jsonl`):**
- `query_id`: Query ID
- `skill_id`: Relevant skill ID
- `relevance`: Binary relevance (1)

The `skill_ids` field in `queries/test.jsonl` is pre-merged from `qrels/test.jsonl`, so qrels are not loaded separately during evaluation.

## Installation

```bash
# Requires git-lfs
git lfs install
uv run benchmark/install_benchmark.py skill_ret_bench
uv run benchmark/install_benchmark.py skill_ret_bench --commit-hash-skillret <hash>
```

This clones the HuggingFace repository to `benchmark/skill_ret_bench/data/SKILLRET/`. No conversion is needed — all data files are already JSONL.

## Usage

### Seed corpus only

```bash
uv run benchmark/skill_ret_bench/run_skill_ret_bench.py \
  --corpus-path benchmark/skill_ret_bench/data/SKILLRET/data/skills/test.jsonl \
  --seed-only
```

### Run full benchmark

```bash
uv run benchmark/skill_ret_bench/run_skill_ret_bench.py \
  --corpus-path benchmark/skill_ret_bench/data/SKILLRET/data/skills/test.jsonl \
  --query-bank-path benchmark/skill_ret_bench/data/SKILLRET/data/queries/test.jsonl \
  --user-id benchmark \
  --k-values 1 3 5 10
```

### Options

- `--user-id`: User scope for memory operations (default: "benchmark")
- `--k-values`: List of k values for metrics (default: 1 3 5 10)
- `--query-bank-path`: Path to JSONL query bank file
- `--corpus-path`: Path to JSONL skill corpus file (required)
- `--results-dir`: Directory for results (default: "results")
- `--results-filename`: Custom filename for results
- `--seed-only`: Only seed corpus, skip evaluation
- `--clear-existing`: Clear existing procedures before seeding
- `--max-queries`: Limit number of queries for testing (smoke test)

## Results JSON

Each run saves a JSON artifact with:

- Benchmark metadata
- System configuration
- Corpus statistics
- Query bank statistics
- Overall metrics (MRR, MAP, Hit@k, P@k, R@k, F1@k, NDCG@k)
- Category-stratified metrics (by major/sub taxonomy)
- Per-query results with retrieved rankings and metrics

## Metrics

The benchmark computes:

- **MRR** (Mean Reciprocal Rank): Average of 1/rank of first relevant result
- **MAP** (Mean Average Precision): Average precision across queries
- **Hit@k**: Whether at least one relevant result appears in top-k
- **P@k** (Precision@k): Proportion of top-k results that are relevant
- **R@k** (Recall@k): Proportion of relevant results found in top-k
- **F1@k**: Harmonic mean of P@k and R@k
- **NDCG@k** (Normalized Discounted Cumulative Gain): Position-weighted relevance

## Data Source

- **HuggingFace**: https://huggingface.co/datasets/ThakiCloud/SKILLRET
- **Paper**: SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents (arxiv 2605.05726)
