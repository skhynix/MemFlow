# MemFlow

Procedural Memory layer for AI agents.

MemFlow captures how-to knowledge (step-by-step instructions, workflows, SOPs)
and makes them retrievable and executable.

## Benchmarks

Benchmark harnesses live in [benchmark/](benchmark/README.md). The WikiHow
Procedure Silver benchmark vendors its query bank, but full retrieval
evaluation requires rebuilding the local procedure corpus from Kaggle source
shards.

Install the optional benchmark dependencies before downloading the WikiHow
source data:

```bash
uv sync --extra benchmark
uv run kaggle datasets download \
  -d paolop/human-instructions-dataset-updated-json-files \
  -p benchmark/wikihow_procedure_silver/raw \
  --unzip
uv run benchmark/install_benchmark.py wikihow_procedure_silver \
  --raw-dir benchmark/wikihow_procedure_silver/raw
```

The SkillRet benchmark is installed directly from HuggingFace (requires
git-lfs):

```bash
git lfs install
uv run benchmark/install_benchmark.py skill_ret_bench
uv run benchmark/install_benchmark.py skill_ret_bench --commit-hash-skillret <hash>
```
