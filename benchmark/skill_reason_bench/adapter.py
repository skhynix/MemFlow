# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the SkillReason-Bench corpus and retrieval.

SkillReason-Bench (https://huggingface.co/datasets/donghongjiang/skillreason-bench,
paper arXiv:2608.08640) evaluates agent skill retrieval for **implicit** user
requests: concise task goals that leave the required capabilities unstated.

  - ``data/queries.jsonl``: 3,729 requests, each with exactly one gold skill
    (``positive_ids``), a fine-grained ``category`` and a ``domain`` label.
  - ``data/skills.jsonl``: 61,228 candidate skills (single unsharded JSONL),
    web-collected with source metadata (``stars``, ``github_url``, ...).

Each skill record has the schema::

    {"skill_id": "...", "name": "...", "description": "...", "body": "...",
     "author": "...", "repo_slug": "...", "path": "...",
     "github_url": "...", "skillsmp_url": "...", "stars": <int>}

The official protocol encodes queries with the Qwen3-Embedding asymmetric
instruction prefix and skill documents as plain text under a 2,048-token cap;
the constants below implement that protocol for MemFlow.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from memflow import MemFlow, Procedure
from memflow.store import env_flag

SKILL_REASON_CORPUS_SOURCE = "skillreason-bench"

# Official SkillReason-Bench query instruction (dataset README; identical to
# the SkillRouter bi-encoder instruction). Documents are encoded WITHOUT the
# prefix, so the corpus embeddings seeded earlier stay valid and only the query
# side is prefixed at eval time.
SKILL_REASON_QUERY_INSTRUCTION = (
    "Given a task description, retrieve the most relevant skill document "
    "that would help an agent complete the task"
)

# Official protocol caps each skill document at 2,048 tokens. We approximate
# that with a character budget on the composed content (description + body)
# BEFORE tokenization: Qwen3 BPE averages ~3.5 chars/token for English-ish
# content, so 2,048 tokens is ~7,200 chars; we err on the low side at 7,000 so
# the extra fields skill_search_text prepends (name/description/category) keep
# the final embedded text within the official window. This is an approximation
# of a token budget, not the tokenizer-exact boundary.
SKILL_REASON_CONTENT_MAX_CHARS = 7_000


@dataclass
class SkillReasonRecord:
    """A single skill record from the SkillReason-Bench corpus."""

    id: str
    title: str
    content: str
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorpusSeedStats:
    """Statistics for corpus seeding."""

    source_repo: str
    corpus_path: str
    num_records: int = 0
    num_seeded: int = 0
    num_reused: int = 0
    num_skipped: int = 0
    num_deleted: int = 0
    num_truncated: int = 0
    num_missing: int = 0
    missing_ids_sample: list[str] = field(default_factory=list)

    @property
    def active_corpus_size(self) -> int:
        return self.num_seeded + self.num_reused

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["active_corpus_size"] = self.active_corpus_size
        return payload


@dataclass
class RetrievedSkill:
    """A retrieved skill from MemFlow search."""

    procedure_id: str
    title: str
    category: str
    score: float
    tags: list[str] = field(default_factory=list)


def _open_jsonl(path: str | Path):
    """Open a JSONL file, transparently handling ``.gz`` compression."""
    jsonl_path = Path(path)
    if jsonl_path.suffix == ".gz":
        return gzip.open(jsonl_path, "rt", encoding="utf-8")
    return jsonl_path.open("r", encoding="utf-8")


def iter_jsonl_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream a JSONL file (optionally gzip-compressed) as dictionaries."""
    jsonl_path = Path(path)
    with _open_jsonl(jsonl_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {jsonl_path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected object at {jsonl_path}:{line_number}, "
                    f"got {type(record).__name__}"
                )
            yield record


def normalize_skill_reason_record(raw: dict[str, Any]) -> SkillReasonRecord:
    """Normalize a raw SkillReason-Bench skill record.

    The official document rendering concatenates name, description, and body
    under the 2,048-token cap. Here the store's ``skill_search_text`` already
    prepends ``title``/``name``/``description``/``category`` from the procedure
    fields and metadata, so the composed ``content`` carries
    ``description + "\\n\\n" + body`` capped at
    ``SKILL_REASON_CONTENT_MAX_CHARS``. Corpus source metadata (``stars``,
    ``github_url``, ``repo_slug``, ``author``, ``path``) is preserved verbatim
    for later utility/popularity analysis; ``stars`` in particular is a
    popularity confound we want available downstream.
    """
    skill_id = str(raw.get("skill_id", "")).strip()
    name = str(raw.get("name", "")).strip()
    description = str(raw.get("description", "")).strip()
    body = str(raw.get("body", "")).strip()

    metadata: dict[str, Any] = {
        "skill_id": skill_id,
        "name": name,
        "description": description,
        "author": str(raw.get("author", "")).strip(),
        "repo_slug": str(raw.get("repo_slug", "")).strip(),
        "path": str(raw.get("path", "")).strip(),
        "github_url": str(raw.get("github_url", "")).strip(),
        "skillsmp_url": str(raw.get("skillsmp_url", "")).strip(),
    }
    if raw.get("stars") is not None:
        metadata["stars"] = raw["stars"]
    metadata = {k: v for k, v in metadata.items() if v}

    content_parts = [part for part in (description, body) if part]
    content = "\n\n".join(content_parts) if content_parts else name

    return SkillReasonRecord(
        id=skill_id,
        title=name or skill_id,
        content=content,
        category="general",
        tags=[],
        metadata=metadata,
    )


def truncate_skill_reason_content(
    record: SkillReasonRecord,
) -> tuple[SkillReasonRecord, bool]:
    """Apply the official document cap to a normalized record.

    Returns ``(possibly_new_record, was_truncated)``. Kept separate from
    normalization so tests and stats can observe the truncation rate of the
    corpus under the official budget.
    """
    if len(record.content) <= SKILL_REASON_CONTENT_MAX_CHARS:
        return record, False
    truncated = SkillReasonRecord(
        id=record.id,
        title=record.title,
        content=record.content[:SKILL_REASON_CONTENT_MAX_CHARS],
        category=record.category,
        tags=list(record.tags),
        metadata=dict(record.metadata),
    )
    return truncated, True


def iter_skill_reason_records(path: str | Path) -> Iterator[SkillReasonRecord]:
    """Iterate over SkillReason records from a (gzip) JSONL file."""
    for raw in iter_jsonl_records(path):
        yield normalize_skill_reason_record(raw)


def skill_record_to_procedure(
    record: SkillReasonRecord | dict[str, Any], user_id: str
) -> Procedure:
    """Convert a SkillReason corpus record into a MemFlow Procedure.

    All SkillReason metadata is preserved in ``Procedure.metadata['skill']`` so
    the vector-store payload keeps the source fields and the embedding text
    picks up name/description via ``skill_search_text``.
    """
    normalized = (
        normalize_skill_reason_record(record) if isinstance(record, dict) else record
    )
    return Procedure(
        id=normalized.id,
        title=normalized.title,
        content=normalized.content,
        user_id=user_id,
        category=normalized.category,
        tags=list(normalized.tags),
        kind="skill",
        metadata={
            "skill": normalized.metadata,
        },
    )


def _collect_corpus_ids(paths: list[Path]) -> set[str]:
    """Collect all skill IDs from the corpus files."""
    return {
        record.id
        for path in paths
        for record in iter_skill_reason_records(path)
        if record.id
    }


def _collect_existing_procedure_ids(memflow: MemFlow, user_id: str) -> set[str]:
    """Collect all existing procedure IDs from MemFlow store.

    Uses ``list_ids`` instead of ``list`` so id comparison avoids full
    documents where the backend supports id-only listing (Qdrant). Used both
    before seeding (reuse/clear bookkeeping) and after it (completeness
    verify), so a list failure is always fatal here.
    """
    try:
        return {pid for pid in memflow.store.list_ids(user_id=user_id) if pid}
    except Exception as exc:
        raise RuntimeError(
            "Failed to list existing procedures for SkillReason-Bench seeding "
            f"for user_id={user_id!r}; refusing to continue because duplicate "
            "corpus IDs could be created or the seeded corpus cannot be "
            "verified. Resolve the list error before rerunning; use "
            "--clear-existing only for an intentional fresh reseed."
        ) from exc


def _resolve_corpus_paths(corpus_path: str | Path) -> list[Path]:
    """Resolve a corpus path into an ordered list of JSONL files.

    The official corpus is a single ``skills.jsonl``; a directory is accepted
    too and expanded to its sorted ``*.jsonl(.gz)`` files for smoke subsets.
    """
    path = Path(corpus_path)
    if path.is_dir():
        shards = sorted(
            p
            for p in path.iterdir()
            if p.is_file()
            and (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz"))
        )
        if not shards:
            raise FileNotFoundError(
                f"No .jsonl/.jsonl.gz shards found in directory: {path}"
            )
        return shards
    if not path.exists():
        raise FileNotFoundError(f"Corpus path not found: {path}")
    return [path]


def _trim_to_budget(
    procedures: list[Procedure],
    stats: CorpusSeedStats,
    max_records: int | None,
) -> list[Procedure]:
    """Trim a pending batch to the remaining max_records seeding budget.

    Returns the trimmed list (possibly empty) and updates ``stats.num_skipped``
    for the dropped tail. A negative/zero budget returns ``[]``.
    """
    if max_records is None:
        return procedures
    remaining = max_records - stats.num_seeded
    if remaining <= 0:
        return []
    if len(procedures) > remaining:
        stats.num_skipped += len(procedures) - remaining
        return procedures[:remaining]
    return procedures


async def _add_batch(
    memflow: MemFlow,
    procedures: list[Procedure],
    *,
    use_sync: bool,
    embedding_batch_size: int,
    max_workers: int,
    stats: CorpusSeedStats,
) -> None:
    """Embed + upsert one batch of procedures, updating ``stats.num_seeded``.

    Calls the store's ``add`` / ``add_async`` directly (not MemFlow's manager
    wrappers) so the embedding ``batch_size`` is honored — the manager's
    ``add_async`` does not forward ``batch_size`` and defaults it to 10, which
    underutilizes a batch-capable embedding server. ``store.add*`` returns an
    int (count inserted) rather than the manager's dict, so we add it to
    ``num_seeded`` directly.
    """
    result = (
        memflow.store.add(procedures, batch_size=embedding_batch_size)
        if use_sync
        else await memflow.store.add_async(
            procedures,
            batch_size=embedding_batch_size,
            max_workers=max_workers,
        )
    )
    stats.num_seeded += result


async def seed_skill_reason_corpus(
    memflow: MemFlow,
    user_id: str,
    corpus_path: str | Path,
    clear_existing: bool = False,
    batch_size: int = 100,
    max_records: int | None = None,
    max_workers: int = 12,
    use_sync: bool = False,
    embedding_batch_size: int = 64,
) -> CorpusSeedStats:
    """Seed the SkillReason-Bench corpus into MemFlow using batch operations.

    Args:
        memflow: MemFlow instance
        user_id: User ID for procedures
        corpus_path: Path to ``skills.jsonl`` OR a directory of JSONL files
        clear_existing: Whether to clear existing corpus procedures first
        batch_size: Number of procedures to add in each batch
        max_records: Maximum number of records to seed (default: all)
        max_workers: Max concurrent embedding/upsert requests (async mode only)
        use_sync: Use synchronous ``store.add`` instead of ``store.add_async``
        embedding_batch_size: Number of texts per embedding API request.

    Returns:
        CorpusSeedStats with seeding statistics (``num_truncated`` records how
        many documents hit the official content cap)
    """
    corpus_paths = _resolve_corpus_paths(corpus_path)
    stats = CorpusSeedStats(
        source_repo=SKILL_REASON_CORPUS_SOURCE,
        corpus_path=str(corpus_path),
    )

    print("\n=== Seeding SkillReason-Bench Corpus ===")
    print(f"Corpus: {len(corpus_paths)} file(s) under {corpus_path}")
    if not env_flag("MEMFLOW_EMBEDDING_STRICT"):
        print(
            "[NOTE] MEMFLOW_EMBEDDING_STRICT is off: a failing embedding "
            "endpoint substitutes hash/zero fallback vectors and the upsert "
            "still reports success. For benchmark seeding set "
            "MEMFLOW_EMBEDDING_STRICT=on so failures abort instead of "
            "silently poisoning the corpus."
        )

    existing_ids: set[str] = set()
    if clear_existing:
        print("Clearing existing SkillReason procedures...")
        corpus_ids = _collect_corpus_ids(corpus_paths)
        existing = _collect_existing_procedure_ids(memflow, user_id=user_id)
        to_delete = sorted(corpus_ids & existing)
        print(f"  Clearing {len(to_delete)} existing procedures...")
        # delete_many propagates errors: a failed clear aborts here instead
        # of seeding onto a half-cleared store. The delete is idempotent, so
        # rerunning --clear-existing recovers from a mid-clear failure.
        stats.num_deleted = memflow.store.delete_many(to_delete)
        print(f"  Deleted {stats.num_deleted} existing procedures")
    else:
        existing_ids = _collect_existing_procedure_ids(memflow, user_id=user_id)
        print(f"Reusing existing procedures by ID ({len(existing_ids)} available)")

    print("Streaming procedures from (gzip) JSONL...")

    # Ids actually submitted to the store (after budget trimming), so the
    # post-seeding verify below compares against what was promised, not what
    # the file contained (max_records / blank-id records are not promises).
    promised_ids: set[str] = set()
    reused_ids: set[str] = set()
    procedures_to_seed: list[Procedure] = []
    for path in corpus_paths:
        for record in iter_skill_reason_records(path):
            stats.num_records += 1
            if not record.id:
                stats.num_skipped += 1
                continue

            if not clear_existing and record.id in existing_ids:
                stats.num_reused += 1
                reused_ids.add(record.id)
                continue

            capped, was_truncated = truncate_skill_reason_content(record)
            if was_truncated:
                stats.num_truncated += 1
            procedure = skill_record_to_procedure(capped, user_id=user_id)
            procedures_to_seed.append(procedure)

            if len(procedures_to_seed) >= batch_size:
                trimmed = _trim_to_budget(procedures_to_seed, stats, max_records)
                if (
                    not trimmed
                    and max_records is not None
                    and stats.num_seeded >= max_records
                ):
                    procedures_to_seed = []
                    break
                promised_ids.update(p.id for p in trimmed)
                await _add_batch(
                    memflow,
                    trimmed,
                    use_sync=use_sync,
                    embedding_batch_size=embedding_batch_size,
                    max_workers=max_workers,
                    stats=stats,
                )
                print(
                    f"\r  Seeded {stats.num_seeded} procedures "
                    f"({stats.num_reused} reused)",
                    end="",
                    flush=True,
                )
                procedures_to_seed = []

            if max_records is not None and stats.num_seeded >= max_records:
                break
        if max_records is not None and stats.num_seeded >= max_records:
            break

    if procedures_to_seed:
        trimmed = _trim_to_budget(procedures_to_seed, stats, max_records)
        if trimmed:
            promised_ids.update(p.id for p in trimmed)
            await _add_batch(
                memflow,
                trimmed,
                use_sync=use_sync,
                embedding_batch_size=embedding_batch_size,
                max_workers=max_workers,
                stats=stats,
            )

    # Completeness verify: the seeded/reused counters only say what the
    # backend *accepted*. Docs can still be missing from the store (e.g. a
    # dropped upsert under a flaky endpoint), and a silent shortfall skews
    # every retrieval metric — so re-list the store and report any promised
    # id that is not actually there. This is the check that turns the
    # 61,226-vs-61,228 discrepancy into a loud warning instead of a mystery.
    # Cost: one id-only listing (payload-light on Qdrant), negligible vs
    # embedding the corpus.
    print("Verifying seeded corpus completeness against the store...")
    present_ids = _collect_existing_procedure_ids(memflow, user_id=user_id)
    expected_ids = promised_ids | reused_ids
    missing = sorted(expected_ids - present_ids)
    stats.num_missing = len(missing)
    stats.missing_ids_sample = missing[:20]
    if missing:
        print(
            f"\n[WARNING] {len(missing)} of {len(expected_ids)} expected corpus "
            f"procedures are NOT in the store (sample: "
            f"{', '.join(stats.missing_ids_sample)}). Retrieval metrics will "
            "be computed against an incomplete corpus. Rerun with "
            "--clear-existing (the reuse path will re-add only the missing "
            "ids), and set MEMFLOW_EMBEDDING_STRICT=on to surface embedding "
            "failures directly."
        )
    else:
        print("  completeness OK: all expected procedures present")

    print()
    print(
        f"Corpus seeding complete: {stats.num_seeded} seeded, "
        f"{stats.num_reused} reused, {stats.num_truncated} truncated "
        f"({stats.num_skipped} skipped)\n"
    )
    return stats


class MemFlowSkillReasonAdapter:
    """Retrieval adapter around MemFlow.search() for SkillReason-Bench."""

    def __init__(
        self,
        memflow: MemFlow,
        user_id: str,
        corpus_size: int,
        backend: str,
        llm_provider: str,
        llm_model: str,
    ) -> None:
        self.memflow = memflow
        self.user_id = user_id
        self.corpus_size = corpus_size
        self.backend = backend
        self.llm_provider = llm_provider
        self.llm_model = llm_model

    def get_system_name(self) -> str:
        return "memflow_search_skill_reason_bench"

    def get_system_info(self) -> dict[str, Any]:
        return {
            "method": "memflow.search",
            "backend": self.backend,
            "user_id": self.user_id,
            "corpus_size": self.corpus_size,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
        }

    def retrieve(
        self,
        query: str,
        k: int = 5,
        exclude_procedure_ids: set[str] | None = None,
    ) -> list[RetrievedSkill]:
        """Retrieve results for a single query.

        Args:
            query: Search query string (an implicit user request)
            k: Number of results to return
            exclude_procedure_ids: Set of procedure IDs to exclude from results

        Returns:
            List of RetrievedSkill objects
        """
        excluded = exclude_procedure_ids or set()
        fetch_k = k + len(excluded)
        search_results = self.memflow.search(
            query, user_id=self.user_id, top_k=fetch_k, kind="skill"
        )
        retrieved: list[RetrievedSkill] = []
        for result in search_results:
            procedure = result.procedure
            if procedure.id in excluded:
                continue
            score = float(result.score) if result.score is not None else 0.0
            retrieved.append(
                RetrievedSkill(
                    procedure_id=procedure.id,
                    title=procedure.title,
                    category=procedure.category,
                    tags=list(procedure.tags),
                    score=score,
                )
            )
        return retrieved[:k]

    def retrieve_batch(
        self,
        queries: list[tuple[str, set[str]]],
        k: int = 5,
    ) -> list[list[RetrievedSkill]]:
        """Retrieve results for multiple queries using batch operations."""
        query_strings = [q for q, _ in queries]
        exclude_map = {i: excl for i, (_, excl) in enumerate(queries)}

        max_exclude = max((len(excl) for excl in exclude_map.values()), default=0)
        all_results = self.memflow.search(
            query=query_strings,
            user_id=self.user_id,
            top_k=k + max_exclude if exclude_map else k,
            kind="skill",
        )

        final_results = []
        for i, results in enumerate(all_results):
            excluded = exclude_map.get(i, set())
            filtered = []
            for result in results:
                if result.procedure.id not in excluded:
                    score = float(result.score) if result.score is not None else 0.0
                    filtered.append(
                        RetrievedSkill(
                            procedure_id=result.procedure.id,
                            title=result.procedure.title,
                            category=result.procedure.category,
                            tags=list(result.procedure.tags),
                            score=score,
                        )
                    )
            final_results.append(filtered[:k])

        return final_results

    async def retrieve_async(
        self,
        query: str,
        k: int = 5,
        exclude_procedure_ids: set[str] | None = None,
    ) -> list[RetrievedSkill]:
        """Retrieve results for a single query asynchronously."""
        import asyncio

        return await asyncio.to_thread(
            self.retrieve, query, k=k, exclude_procedure_ids=exclude_procedure_ids
        )

    async def retrieve_batch_async(
        self,
        queries: list[tuple[str, set[str]]],
        k: int = 5,
        max_concurrency: int = 64,
    ) -> list[list[RetrievedSkill]]:
        """Retrieve results for multiple queries in parallel."""
        import asyncio
        from asyncio import Semaphore

        semaphore = Semaphore(max_concurrency)

        async def retrieve_single(
            query: str, exclude_procedure_ids: set[str]
        ) -> list[RetrievedSkill]:
            async with semaphore:
                return await asyncio.to_thread(
                    self.retrieve,
                    query,
                    k=k,
                    exclude_procedure_ids=exclude_procedure_ids,
                )

        tasks = [retrieve_single(query, exclude_ids) for query, exclude_ids in queries]
        return await asyncio.gather(*tasks)
