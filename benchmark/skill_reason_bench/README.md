# SkillReason-Bench Benchmark

This directory provides a benchmark harness that evaluates **MemFlow retrieval**
against **SkillReason-Bench**, the skill-retrieval benchmark of the SkillReason
paper ("Reasoning-Enhanced Agent Skill Retrieval for Implicit User Requests",
arXiv:2608.08640).

SkillReason-Bench targets **implicit** user requests: concise task goals that
state only the desired outcome and leave the required capabilities and
execution steps unstated, so the retriever must reason from a goal to the
capability that provides it. It differs from the sibling benchmarks in three
structural ways:

- queries are natural-language **user requests** (mean ~33 words), not skill-
  derived queries or task-routing instructions,
- **every query has exactly one gold skill** (`positive_ids` has a single
  element), so relevance is strictly binary and NDCG/MRR take their single-gold
  forms,
- the corpus spans **ten domains** (software, data science, biomedical, media
  & design, legal, finance, science, engineering, writing, other) over 61,228
  web-collected skills — cross-domain routing rather than within-domain
  ranking.

## Prerequisites

Same as the other MemFlow benchmarks: a running **Qdrant** server and an
**embedding server** (OpenAI-compatible embeddings endpoint, e.g.
`Qwen/Qwen3-Embedding-4B` via vLLM), configured in the repo-root `.env`. The
collection for this benchmark is selected via `QDRANT_COLLECTION_NAME`
(suggested value: `skillreason`). See the repo-root `.env.example` and
[`../skill_ret_bench/README.md`](../skill_ret_bench/README.md) for the full
`.env` template, and make sure `MEMFLOW_HYBRID_SEARCH_ENABLED` is unset/off
for a pure-dense comparison with the other benchmarks.

## Dataset

- **Repository**: https://huggingface.co/datasets/donghongjiang/skillreason-bench
- **Size**: 3,729 queries × 61,228 skills (single unsharded `skills.jsonl`,
  ~537 MB); per-file sha256 checksums are shipped in `data/stats.json`
- **License**: see the dataset card ("other" — treat as research-only per the
  upstream terms before redistribution)

Files:

- `data/queries.jsonl` — `query_id`, `query`, `positive_ids` (single gold),
  `category` (122 fine-grained labels), `domain` (10 labels)
- `data/skills.jsonl` — `skill_id`, `name`, `description`, `body`, plus source
  metadata (`author`, `repo_slug`, `path`, `github_url`, `skillsmp_url`,
  `stars`) preserved in `SkillRecord.metadata["skill"]` for downstream utility
  analysis (`stars` is a popularity confound worth keeping)
- `data/stats.json` — corpus stats and checksums

### Download

```bash
uv run benchmark/skill_reason_bench/download_data.py
```

`download_data.py` fetches via `huggingface_hub` when installed — any hub
failure (package missing, network down, 401, hub API error) falls back to a
plain HTTPS download — and then **verifies the sha256 of both JSONL files
against `stats.json`** so a corrupted or partial download fails before the
(expensive) seeding run. Use `--no-verify` only deliberately.

The download-time check only covers downloads made by this script, so the
runners **re-check the checksum against `stats.json` before seeding or
scoring** (a file placed by manual copy or truncated afterwards therefore also
fails loudly). Smoke corpora outside the download layout have no `stats.json`
beside them and skip the check with a notice; `--skip-verify` is the explicit
escape hatch on all three runners.

## Official retrieval protocol

This harness reproduces the dataset's official *evaluation* protocol (it
measures MemFlow retrieval as a baseline — it does not reproduce the SkillReason
model's training):

- **Full-corpus ranking**: rank all 61,228 skills per query; a hit is an exact
  `skill_id` match against `positive_ids`.
- **Query instruction** (identical string to the SkillRouter paper's): queries
  are encoded as `Instruct: <instruction>\nQuery: <request>`, documents as
  plain text (asymmetric Qwen3-Embedding format). The runners pin it into
  `VECTOR_EMBEDDING_QUERY_INSTRUCTION` before building the store (the store
  reads the env once at init), so seeded corpus embeddings stay valid and an
  explicit `VECTOR_EMBEDDING_QUERY_INSTRUCTION` in `.env`/shell overrides the
  built-in constant.
- **Document cap of 2,048 tokens**: implemented as a 7,000-character budget on
  the composed `description + body` content (≈3.5 chars/token, rounded down to
  leave room for the name/description fields `skill_search_text` prepends).
  This is an approximation of the official token cap, not a tokenizer-exact
  boundary. The seeding stats report how many corpus documents hit the cap
  (`num_truncated`).
- **Official metrics**: Recall@{1,5,10,20}, NDCG@10, MRR@20, computed exactly
  like the dataset's bundled `evaluate.py` (single-gold NDCG with IDCG = 1;
  MRR window of 20). Retrieval always fetches at least top-20 so the official
  windows are complete regardless of `--k-values`.

## Usage

```bash
# 1. Download + checksum-verify (one-time, ~537 MB)
uv run benchmark/skill_reason_bench/download_data.py

# 2. Seed the full corpus (≈61k documents at ~2k tokens each — roughly twice
#    the SkillRouter hard-pool embedding cost; budget accordingly)
uv run benchmark/skill_reason_bench/run_seeding.py \
    --user-id benchmark --clear-existing --embedding-batch-size 256

# 3. Evaluate all 3,729 queries
uv run benchmark/skill_reason_bench/run_retrieval.py \
    --user-id benchmark --k-values 1 5 10 20
```

`--clear-existing` clears through the bulk store primitives (id-only listing
plus one batched `delete_many`), so even a full 61k reseed clears in seconds
instead of per-id round trips.

After seeding, the runners **re-list the store and verify every promised
document is actually present** (`num_missing` in the stats; a shortfall prints
a `[WARNING]` with an id sample). The seeded/reused counters only record what
the backend accepted, so without this check a dropped upsert would silently
shrink the corpus and skew every metric. Relatedly, seeding runs with
`MEMFLOW_EMBEDDING_STRICT` **recommended**: when unset, the runners print a
note, because a flaky embedding endpoint otherwise substitutes hash/zero
fallback vectors and still reports success.

Or seed + evaluate in one step, and the smoke-test knobs work as elsewhere:

```bash
uv run benchmark/skill_reason_bench/run_skill_reason.py \
    --user-id benchmark --clear-existing \
    --max-records 1000 --max-queries 20
```

Each retrieval run writes two files under `results/`:

- `*_retrieval_*.json` — binary IR metrics, the `official_metrics` block, and
  per-domain stratified metrics, plus full ranked lists per query;
- `*_predictions.jsonl` — predictions in the **official**
  `{"query_id", "retrieved_skill_ids"}` format, re-scorable with the dataset's
  own `evaluate.py`:

```bash
python evaluate.py --queries data/skillreason-bench/data/queries.jsonl \
    --predictions results/<run>_predictions.jsonl
```

The results JSON numbers and the upstream script must agree — the
`official_metrics` block is unit-tested against a transcription of the upstream
scoring loop (see `tests/test_skill_reason_bench.py`).

## Recall@k curves

To see how recall grows with the cutoff, evaluate with a deep `--k-values` cap
so each query's ranked list is long enough, then replot from the saved result
file (no re-retrieval):

```bash
uv run benchmark/skill_reason_bench/run_retrieval.py \
    --user-id benchmark --k-values 1 5 10 20 50 100 \
    --results-filename skillreason_topk100

uv run --extra benchmark benchmark/skill_reason_bench/plot_recall_curves.py \
    --results results/skillreason_topk100.json \
    --output benchmark/skill_reason_bench/plots/recall_at_k.png

# One curve per domain — an aggregate curve can hide that some domains route
# cleanly while others never clear Recall@10:
uv run --extra benchmark benchmark/skill_reason_bench/plot_recall_curves.py \
    --results results/skillreason_topk100.json --by-domain \
    --output benchmark/skill_reason_bench/plots/recall_at_k_by_domain.png
```

Generated charts land under `plots/`, which is gitignored: they are run
outputs regenerable from any saved results JSON, not sources.

## Baseline result

MemFlow dense retrieval on the full corpus (61,228 skills; 42.5% of documents
truncated by the official content cap), Qwen3-Embedding-4B via a vLLM
embedding server, all 3,729 queries, official query instruction. Official
metrics re-scored bit-identical by the dataset's own `evaluate.py`:

| Recall@1 | Recall@5 | Recall@10 | Recall@20 | NDCG@10 | MRR@20 |
|----------|----------|-----------|-----------|---------|--------|
| 0.374    | 0.571    | 0.654     | 0.719     | 0.505   | 0.463  |

Recall keeps growing well past the official cutoffs (0.81 at k=50, 0.86 at
k=100), so a large share of gold skills is ranked just outside the top-k —
substantial headroom for rerankers. Per-domain curves (regenerate with
`--by-domain` as above) show the difficulty is strongly domain-dependent:
`data_science`/`software` route cleanly while `writing_docs` plateaus around
0.56 even at k≈100 and `media_design` starts at R@1≈0.27.

## Metrics

Per `k` (binary, the single-gold relevant set): `hit@k`, `precision@k`,
`recall@k`, `f1@k`, `ndcg@k`, `MRR`, `MAP` — for cross-comparison with the
SkillRet/SkillRouter harnesses. The `official_metrics` block carries the
upstream keys (`Recall@1/5/10/20`, `NDCG@10`, `MRR@20`, `num_queries`,
`prediction_coverage`). Stratification is by the query `domain` field.

The query loader drops records with an empty `positive_ids` (with a warning):
the official split ships exactly one gold per query, and a gold-less record
would divide by zero in the official block while silently diluting the binary
averages — better to surface a truncated `queries.jsonl` than to score it.
