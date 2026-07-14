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
- **Repository**: https://huggingface.co/datasets/anonymous-ed-benchmark/SKILLRET
- **License**: Apache-2.0 (benchmark), MIT/Apache-2.0 (source skills)
- **Size**: 17,810 skills, 63,259 train queries, 4,997 eval queries

### Download the dataset

```bash
# Using git (requires git-lfs)
git lfs install
git clone https://huggingface.co/datasets/anonymous-ed-benchmark/SKILLRET

# Or using Python
from datasets import load_dataset
skills = load_dataset("anonymous-ed-benchmark/SKILLRET", "skills", split="test")
queries = load_dataset("anonymous-ed-benchmark/SKILLRET", "queries", split="test")
qrels = load_dataset("anonymous-ed-benchmark/SKILLRET", "qrels", split="test")
```

### Schema

**Skills (`skills.jsonl`):**
- `id`: Skill ID
- `name`: Skill name
- `namespace`: Skill namespace (author/repo)
- `description`: Short description
- `skill_md`: Full markdown content
- `major`, `sub`: Taxonomy categories (6 major, 18 sub)
- `author`, `stars`, `installs`, `license`, `repo`: Metadata

**Queries (`queries.jsonl`):**
- `id`: Query ID
- `query`: Natural language request
- `skill_ids`: List of relevant skill IDs (ground truth)
- `k`: Number of relevant skills

**Qrels (`qrels.jsonl`):**
- `query_id`: Query ID
- `skill_id`: Relevant skill ID
- `relevance`: Binary relevance (1)

## Usage

### Convert HuggingFace dataset to JSONL

```python
from datasets import load_dataset
import json

# Load and export skills
skills = load_dataset("anonymous-ed-benchmark/SKILLRET", "skills", split="test")
with open("skills.jsonl", "w") as f:
    for skill in skills:
        f.write(json.dumps(skill) + "\n")

# Load and export queries with qrels combined
queries = load_dataset("anonymous-ed-benchmark/SKILLRET", "queries", split="test")
qrels = load_dataset("anonymous-ed-benchmark/SKILLRET", "qrels", split="test")

# Build qrels map: query_id -> [skill_ids]
from collections import defaultdict
qrels_map = defaultdict(list)
for r in qrels:
    if r["relevance"] == 1:
        qrels_map[r["query_id"]].append(r["skill_id"])

# Export queries with skill_ids
with open("queries.jsonl", "w") as f:
    for q in queries:
        q["skill_ids"] = qrels_map.get(q["id"], [])
        f.write(json.dumps(q) + "\n")
```

### Seed corpus only

```bash
uv run benchmark/skill_ret_bench/run_skill_ret_bench.py \
  --corpus-path benchmark/skill_ret_bench/data/SKILLRET/data/skills.jsonl \
  --seed-only
```

### Run full benchmark

```bash
uv run benchmark/skill_ret_bench/run_skill_ret_bench.py \
  --corpus-path benchmark/skill_ret_bench/data/SKILLRET/data/skills.jsonl \
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

- **HuggingFace**: https://huggingface.co/datasets/anonymous-ed-benchmark/SKILLRET
- **Paper**: SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents (arxiv 2605.05726)
