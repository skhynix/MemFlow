# MemFlow Examples Guide

This guide provides an overview of all example scripts in the `examples/` directory and how to run them.

## Prerequisites

- Python 3.10+
- `uv` package manager
- Ollama server running with `llama3.2` model (default)

```bash
# Install dependencies
uv sync

# Include optional dependencies for full functionality
# --all-extras installs: ollama, memmachine, openai
uv sync --all-extras
```

## Running Examples

```bash
uv run ./examples/<example_name>.py
```

## Environment Configuration

### Using `--env` flag

```bash
# File Storage (persistent)
uv run --env MEMFLOW_BACKEND=file ./examples/06_file_persistence.py

# MemMachine (production VectorDB)
uv run --env MEMFLOW_BACKEND=memmachine ./examples/07_memmachine.py

# MemFlowStore (PostgreSQL + pgvector)
uv run --env MEMFLOW_BACKEND=memflow ./examples/10_run.py
```

### Using `.env` file

Create a `.env` file in the project root for persistent configuration:

```bash
# .env file
MEMFLOW_BACKEND=file
MEMFLOW_DATA_DIR=./memories

LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
LLM_API_BASE=http://localhost:11434
```

Then run examples without specifying `--env` flags:

```bash
uv run ./examples/01_quickstart.py
```

### Model Configuration

```bash
# Different Ollama model
uv run --env LLM_MODEL=llama3.1 ./examples/01_quickstart.py

# OpenAI-compatible server (vLLM, LM Studio)
uv run --env LLM_PROVIDER=openai-compatible \
       --env LLM_MODEL=meta-llama/Llama-3.2-3B \
       --env LLM_API_BASE=http://localhost:8000/v1 \
       ./examples/01_quickstart.py
```
