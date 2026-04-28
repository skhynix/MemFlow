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

All examples now use `.env`-based configuration by default. Simply run:

```bash
uv run ./examples/<example_name>.py
```

## Environment Configuration

### Using `.env` file (Required)

Create a `.env` file in the project root for persistent configuration:

```bash
# Copy the example
cp .env.example .env

# Edit .env with your settings
```

Example `.env` file:
```bash
# Backend: emulated | file | memmachine | pgvector
MEMFLOW_BACKEND=emulated

# LLM Configuration
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
LLM_API_BASE=http://localhost:11434
```

Then run examples:
```bash
uv run ./examples/01_quickstart.py
uv run ./examples/10_run.py
```

For examples that require a specific backend (e.g., `07_memmachine.py`), see the run instructions at the top of each file.
