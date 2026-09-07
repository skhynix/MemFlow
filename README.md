# MemFlow

MemFlow finds relevant skills for your task in Claude Code. Register your
`SKILL.md` files, then describe your task as usual. MemFlow searches your skills
and gives Claude a short list of matches to read through MCP.

The current workflow uses Qdrant and an OpenAI-compatible embedding service.
You do not need a separate generation LLM. MemFlow stores and serves the
`SKILL.md` text; accompanying scripts and reference files are not included.

## Quick start

You need Python 3.10+, [uv](https://docs.astral.sh/uv/getting-started/installation/),
[Claude Code](https://code.claude.com/docs/en/setup), and Docker with Compose.
You also need an existing OpenAI-compatible embedding service.
The commands work in a Linux/macOS shell.

### 1. Install MemFlow

```bash
git clone https://github.com/skhynix/MemFlow.git
cd MemFlow
uv tool install .
docker compose -f scripts/docker-compose.qdrant.yml up -d
```

`uv tool install` makes `memflow` available from any project. If uv reports that
its executable directory is missing from your PATH, run `uv tool update-shell`
and open a new terminal. See [uv's tool installation guide](https://docs.astral.sh/uv/concepts/tools/#tool-executables).

Qdrant runs at `http://localhost:6333` and keeps its data in
`scripts/qdrant_data/`. If you already have Qdrant, skip the Docker command and
use your server's URL below.

### 2. Add connection settings to your project

Go to the project where you use Claude Code:

```bash
cd /path/to/your-project
vi .env
```

Add the following entries to `.env`, keeping any settings your project already
has. The endpoint, model, dimensions, and API key below are example values;
replace them with your embedding service's settings.

```dotenv
QDRANT_BASE_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=memflow_skills

VECTOR_EMBEDDING_API_BASE=https://your_endpoint/v1
VECTOR_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
VECTOR_EMBEDDING_DIMENSIONS=2560
VECTOR_EMBEDDING_API_KEY=EMPTY
```

Use `EMPTY` if your service does not require an API key.
[.env.example](.env.example) includes optional settings.
Add `.env` to your project's `.gitignore` to keep credentials out of Git.

### 3. Connect Claude Code

From your project's root directory, run:

```bash
memflow claude configure --apply
```

This installs the hook, registers a private MCP server for this project, and
saves the path to your `.env`. Skill commands, the hook, and MCP reuse that
file. You can safely run the command again with the same settings.

The output should show `hook.installed_after: true` and
`mcp.installed_after: true`.

### 4. Register a skill

Use a skill you already have. The example below uses `skills/release-checklist`;
replace it with your skill's directory or `SKILL.md` path.

<details>
<summary>Need a skill to try?</summary>

Create a file in your project:

```bash
mkdir -p skills/release-checklist
vi skills/release-checklist/SKILL.md
```

Save this content:

```markdown
---
name: release-checklist
description: Prepare a software release by checking tests, release notes, and version updates.
---

# Release checklist

1. Find the project's test command and check the results.
2. Review recent changes and draft release notes.
3. Check version updates and summarize what remains before publishing.
```

</details>

```bash
memflow skill add skills/release-checklist --trust-state trusted
memflow skill list
```

Your skill should appear in the list. Only mark skills you trust as `trusted`;
this lets Claude use their contents as instructions. Registration stores a
snapshot, so run `skill sync` after editing the file.

### 5. Use Claude Code

Start a new Claude Code session in the same project:

```bash
claude
```

Ask for a task covered by your skill. For the release checklist, try:

> Help me prepare this project's next release.

When Claude reads a matching skill, you will see a MemFlow `read_skill` tool
call. You can also use `/mcp` to check that `memflow` is connected. If you do not
see a match, follow [Troubleshooting](#troubleshooting).

## Everyday use

Run these from a configured project's root, including in a new terminal:

```bash
# Register another skill.
memflow skill add skills/code-review --trust-state trusted

# Update the stored instructions after editing a skill.
memflow skill sync skills/release-checklist

# See your registered skills.
memflow skill list
```

Each skill must be registered explicitly. Adding a file to a Claude Code skill
directory does not automatically register it in MemFlow.

To disconnect MemFlow from the project:

```bash
memflow claude configure --hook off --mcp off --apply
```

Your stored skills remain in Qdrant. Reconnect with
`memflow claude configure --apply`. You can also change the hook or MCP
individually, for example with `--hook off --apply`.

## Configuration

To keep MemFlow settings separate from your application's `.env`, put them in
`.env.memflow` and select that file once:

```bash
memflow claude configure --env-file .env.memflow --apply
```

This saves the path while preserving the current hook, MCP, and catalog
settings, including disconnected integrations. For first-time setup with this
file, enable the hook and MCP explicitly:

```bash
memflow claude configure --hook on --mcp on --env-file .env.memflow --apply
```

`memflow claude configure --apply` enables both integrations using the saved
path, or the project's `.env` if no path has been saved.

Keep `.env.memflow` out of Git as well. Subsequent commands reuse the saved path.
A skill command's `--env-file` option overrides it for that command. Existing
shell environment variables take precedence over values in the file. Restart
Claude Code after changing the connection settings.

Use the same embedding model and dimensions for registration and retrieval.
If you change either, choose a new `QDRANT_COLLECTION_NAME` and register your
skills again. MemFlow creates the collection when it is first used.

For retrieval tuning, edit `.memflow/claude-hook.json`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `retrieval.top_k` | `3` | Maximum number of matching skills to provide. |
| `retrieval.min_score` | `0.2` | Minimum similarity score for a match. |
| `retrieval.timeout_ms` | `2000` | Time allowed to initialize the store and find skills. |

<details>
<summary>Optional: minimize Claude's native skill catalog</summary>

```bash
memflow claude configure --catalog hidden_or_minimized --apply
```

This disables bundled skills and makes discovered local skills user-invocable
only. Existing overrides are preserved; plugin skills are not managed. You
still need to register the skills you want MemFlow to retrieve.

Restore the settings with:

```bash
memflow claude configure --catalog visible --apply
```

Keep `.memflow/claude-catalog-state.json` so restoration can identify settings
managed by MemFlow. Disabling the hook or MCP does not restore catalog settings.

</details>

## Troubleshooting

Start with these commands in your project's root:

```bash
memflow claude status
claude mcp get memflow
```

`status` shows the saved environment file and hook/MCP configuration. Look for
`env_file_exists`, `hook.installed`, `mcp.installed`, and `mcp.matches_config`
to be `true`. It checks configuration, not Qdrant or embedding connectivity.
`claude mcp get` checks whether the MCP process connects.

| Problem | What to do |
| --- | --- |
| `memflow` command not found | Run `uv tool update-shell`, open a new terminal, and check that you installed MemFlow with `uv tool install .`. |
| Environment file not found | Create the file shown in the error, or rerun `configure --env-file` with the correct file and `--apply`. |
| An existing `memflow` MCP server conflicts | Inspect it with `claude mcp get memflow`. To replace an obsolete local registration, run `claude mcp remove --scope local memflow`, then `memflow claude configure --apply`. |
| No skill is selected | Run `memflow skill list` and check your embedding settings. Check the hook log for `no_results` or `fail_open`; adjust `min_score` or `timeout_ms` if needed. |
| MCP connects but cannot read a skill | Check the saved environment file and Qdrant connection with `memflow skill list`, then restart Claude Code. |
| A skill has old contents | Run `memflow skill sync` with its source path. |

The hook log is `.memflow/logs/skill_context_hook.jsonl`. Matching skill names
and IDs are added to Claude's context, so the catalog may not appear as a chat
message. Raw prompts and skill bodies are omitted from the log by default.

If an embedding API request fails, MemFlow can fall back to hash vectors, which
do not provide semantic search. Fix the endpoint and re-register affected
skills with `skill add`. When retrieval fails or times out, Claude continues
without additional skill context.

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
