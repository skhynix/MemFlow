# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import sys

import pytest

from benchmark import install_benchmark
from benchmark.wikihow_procedure_silver import (
    run_wikihow_procedure_silver as wikihow_runner,
)
from benchmark.wikihow_procedure_silver.adapter import (
    MemFlowWikiHowAdapter,
    RetrievedWikiHowProcedure,
    iter_wikihow_procedures,
    seed_wikihow_corpus,
    wikihow_record_to_procedure,
)
from benchmark.wikihow_procedure_silver.build_wikihow_procedures import (
    DEFAULT_CATEGORY_ALIASES,
    build_wikihow_procedures,
)
from benchmark.wikihow_procedure_silver.evaluation import (
    WikiHowBenchmarkQuery,
    compute_binary_ir_metrics,
    evaluate_wikihow_queries,
    load_wikihow_query_bank,
)
from memflow.models import Procedure, SearchResult


def _write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _sha256(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class FakeStore:
    def __init__(self, existing: list[Procedure] | None = None) -> None:
        self.existing = existing or []
        self.deleted: list[str] = []

    def list_all(
        self, user_id: str | None = None, kind: str | None = None
    ) -> list[Procedure]:
        procs = list(self.existing)
        if user_id is not None:
            procs = [proc for proc in procs if proc.user_id == user_id]
        if kind is not None:
            procs = [proc for proc in procs if proc.kind == kind]
        return procs

    def add(
        self,
        procedure: Procedure | list[Procedure],
        batch_size: int = 50,
    ) -> None | int:
        if isinstance(procedure, list):
            self.existing.extend(procedure)
            return len(procedure)
        else:
            self.existing.append(procedure)
            return None

    def search(
        self,
        query: str | list[str],
        top_k: int = 5,
        user_id: str | None = None,
        kind: str | None = "skill",
    ) -> list[SearchResult] | list[list[SearchResult]]:
        from memflow.models import SearchResult

        if isinstance(query, list):
            all_results = []
            for q in query:
                results = []
                for proc in self.existing:
                    if user_id and proc.user_id != user_id:
                        continue
                    if kind is not None and proc.kind != kind:
                        continue
                    results.append(SearchResult(procedure=proc, score=0.9))
                all_results.append(results[:top_k])
            return all_results
        else:
            results = []
            for proc in self.existing:
                if user_id and proc.user_id != user_id:
                    continue
                if kind is not None and proc.kind != kind:
                    continue
                results.append(SearchResult(procedure=proc, score=0.9))
            return results[:top_k]

    async def search_async(
        self,
        query: str | list[str],
        top_k: int = 5,
        user_id: str | None = None,
        kind: str | None = "skill",
        max_concurrency: int = 50,
    ) -> list[SearchResult] | list[list[SearchResult]]:
        import asyncio

        return await asyncio.to_thread(self.search, query, top_k, user_id, kind)

    def delete(
        self,
        id: str | list[str],
    ) -> bool | int:
        if isinstance(id, list):
            num_deleted = 0
            for i in id:
                self.deleted.append(i)
                num_deleted += 1
            return num_deleted
        else:
            self.deleted.append(id)
            return True

    async def delete_async(
        self,
        id: str | list[str],
        max_concurrency: int = 50,
    ) -> bool | int:
        import asyncio

        return await asyncio.to_thread(self.delete, id)

    async def add_async(
        self,
        procedure: Procedure | list[Procedure],
        batch_size: int = 50,
        max_concurrency: int = 50,
    ) -> int | None:
        return self.add(procedure, batch_size)


class FakeMemFlow:
    def __init__(self, existing: list[Procedure] | None = None) -> None:
        self.store = FakeStore(existing=existing)
        self.added: list[Procedure] = []

    def add(
        self,
        procedure: Procedure | list[Procedure],
        user_id: str = "default",
        batch_size: int = 50,
    ) -> dict | None:
        if isinstance(procedure, list):
            for proc in procedure:
                proc.user_id = user_id
                self.added.append(proc)
            return {
                "num_seeded": len(procedure),
                "num_skipped": 0,
                "total": len(procedure),
            }
        else:
            procedure.user_id = user_id
            self.added.append(procedure)
            return {"id": procedure.id, "title": procedure.title, "event": "ADD"}


class FakeSearchMemFlow:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.search_calls: list[dict] = []

    def search(
        self,
        query: str,
        user_id: str,
        top_k: int,
        kind: str | None = "skill",
    ) -> list[SearchResult]:
        self.search_calls.append(
            {"query": query, "user_id": user_id, "top_k": top_k, "kind": kind}
        )
        return self.results[:top_k]


def test_wikihow_installer_without_raw_dir_reports_local_query_bank(capsys) -> None:
    install_benchmark.install_wikihow_procedure_silver(raw_dir=None)

    output = capsys.readouterr().out
    assert "benchmark_data/query_bank.jsonl" in output
    assert "Provide --raw-dir" in output


def test_wikihow_builder_uses_temp_raw_shard_and_writes_manifest(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "data"
    raw_dir.mkdir()
    category_aliases = tmp_path / "category_aliases.json"
    category_aliases.write_text(
        json.dumps(
            {
                "alias_map": {"Raw Category": "Canonical Category"},
                "schema_version": 1,
                "unresolved": [],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "wikiHow0.json").write_text(
        json.dumps(
            [
                {
                    "AuthorsCount": 2,
                    "Categories": ["Root", "Raw Category"],
                    "MainTask": "How to Brew Test Tea",
                    "MainTaskSummary": "Make a reliable test cup.",
                    "Steps": [
                        {"Headline": "Heat water", "Description": "Use a kettle."},
                        {"Headline": "Add leaves", "Description": "Measure tea."},
                        {"Headline": "Steep", "Description": "Wait briefly."},
                    ],
                    "Time": "10 minutes",
                    "URL": "https://example.test/tea",
                    "Views": 42,
                },
                {
                    "Categories": ["Root"],
                    "MainTask": "Too Short",
                    "Steps": [
                        {"Headline": "Only one", "Description": "Skip me."},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    manifest = build_wikihow_procedures(
        input_dir=raw_dir,
        output_dir=output_dir,
        category_aliases=category_aliases,
        expected_procedures_sha256=None,
        expected_procedures_records=None,
    )

    procedures_path = output_dir / "wikihow_procedures.jsonl"
    manifest_path = output_dir / "MANIFEST.json"
    records = [json.loads(line) for line in procedures_path.read_text().splitlines()]
    written_manifest = json.loads(manifest_path.read_text())

    assert len(records) == 1
    assert records[0]["title"] == "How to Brew Test Tea"
    assert records[0]["category"] == "Canonical Category"
    assert records[0]["metadata"]["category_normalization"]["aliases_applied"] == [
        {"canonical_label": "Canonical Category", "raw_label": "Raw Category"}
    ]
    assert records[0]["tags"][-1] == "steps:3"
    assert not (output_dir / "wikihow_trajectories.jsonl").exists()
    assert manifest["counts"]["articles_read"] == 2
    assert manifest["counts"]["procedures"] == 1
    assert manifest["counts"]["skipped_too_few_steps"] == 1
    assert written_manifest["outputs"]["procedures"]["sha256"] == _sha256(
        procedures_path
    )
    assert written_manifest["verification"]["matched"] is True


def test_wikihow_builder_default_category_alias_asset_is_vendored() -> None:
    assert DEFAULT_CATEGORY_ALIASES.name == "category_aliases.json"
    assert DEFAULT_CATEGORY_ALIASES.parent.name == "assets"
    assert DEFAULT_CATEGORY_ALIASES.exists()


def test_wikihow_runner_allows_seed_only_without_query_bank(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_wikihow_procedure_silver.py",
            "--corpus-path",
            "wikihow_procedures.jsonl",
            "--seed-only",
        ],
    )

    args = wikihow_runner._parse_args()

    assert args.seed_only is True
    assert args.query_bank_path is None
    assert args.corpus_path == "wikihow_procedures.jsonl"


def test_wikihow_runner_requires_query_bank_for_evaluation(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_wikihow_procedure_silver.py",
            "--corpus-path",
            "wikihow_procedures.jsonl",
        ],
    )

    with pytest.raises(SystemExit):
        wikihow_runner._parse_args()


def test_wikihow_runner_seed_only_skips_query_evaluation(tmp_path, monkeypatch) -> None:
    corpus_path = tmp_path / "wikihow_procedures.jsonl"
    results_dir = tmp_path / "results"
    _write_jsonl(
        corpus_path,
        [
            {
                "id": "wh_001",
                "title": "How to Brew Tea",
                "content": "1. Heat water\n2. Steep leaves",
                "category": "Food and Entertaining",
            }
        ],
    )
    memflow = FakeMemFlow()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("seed-only mode must not evaluate queries")

    monkeypatch.setattr(wikihow_runner, "_load_env_file", lambda: None)
    monkeypatch.setattr(wikihow_runner, "MemFlow", lambda: memflow)
    monkeypatch.setattr(wikihow_runner, "count_query_bank_records", fail_if_called)
    monkeypatch.setattr(wikihow_runner, "load_wikihow_query_bank", fail_if_called)
    monkeypatch.setattr(wikihow_runner, "evaluate_wikihow_queries", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_wikihow_procedure_silver.py",
            "--corpus-path",
            str(corpus_path),
            "--seed-only",
            "--results-dir",
            str(results_dir),
            "--results-filename",
            "seed_only",
        ],
    )

    wikihow_runner.main()

    payload = json.loads((results_dir / "seed_only.json").read_text())
    assert payload["run_mode"] == "seed_only"
    assert payload["system_info"]["method"] == "memflow.add"
    assert payload["settings"]["seed_only"] is True
    assert payload["corpus_stats"]["num_seeded"] == 1
    assert payload["corpus_stats"]["active_corpus_size"] == 1
    assert "overall_metrics" not in payload
    assert [proc.id for proc in memflow.added] == ["wh_001"]


def test_seed_wikihow_corpus_reuses_full_existing_corpus(tmp_path) -> None:
    corpus_path = tmp_path / "wikihow_procedures.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {
                "id": "wh_001",
                "title": "How to Brew Tea",
                "content": "1. Heat water\n2. Steep leaves",
                "category": "Food and Entertaining",
            },
            {
                "id": "wh_002",
                "title": "How to Repair a Zipper",
                "content": "1. Inspect slider\n2. Realign teeth",
                "category": "Clothing",
            },
        ],
    )
    memflow = FakeMemFlow(
        existing=[
            Procedure(
                id="wh_001",
                title="Existing Tea",
                content="old",
                user_id="bench-user",
            ),
            Procedure(
                id="wh_002",
                title="Existing Zipper",
                content="old",
                user_id="bench-user",
            ),
        ]
    )

    stats = seed_wikihow_corpus(
        memflow=memflow,
        user_id="bench-user",
        corpus_path=corpus_path,
    )

    assert memflow.added == []
    assert stats.num_records == 2
    assert stats.num_seeded == 0
    assert stats.num_reused == 2
    assert stats.num_skipped == 0
    assert stats.num_deleted == 0
    assert stats.active_corpus_size == 2
    assert stats.category_counts == {"Clothing": 1, "Food and Entertaining": 1}


def test_seed_wikihow_corpus_reuses_existing_and_seeds_missing(tmp_path) -> None:
    corpus_path = tmp_path / "wikihow_procedures.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {
                "id": "wh_001",
                "title": "How to Brew Tea",
                "content": "1. Heat water\n2. Steep leaves",
                "category": "Food and Entertaining",
            },
            {
                "id": "wh_002",
                "title": "How to Repair a Zipper",
                "content": "1. Inspect slider\n2. Realign teeth",
                "category": "Clothing",
            },
        ],
    )
    memflow = FakeMemFlow(
        existing=[
            Procedure(
                id="wh_001",
                title="Existing Tea",
                content="old",
                user_id="bench-user",
            ),
            Procedure(
                id="wh_002",
                title="Other User Zipper",
                content="old",
                user_id="other-user",
            ),
        ]
    )

    stats = seed_wikihow_corpus(
        memflow=memflow,
        user_id="bench-user",
        corpus_path=corpus_path,
    )

    assert [proc.id for proc in memflow.added] == ["wh_002"]
    assert memflow.added[0].user_id == "bench-user"
    assert stats.num_records == 2
    assert stats.num_seeded == 1
    assert stats.num_reused == 1
    assert stats.num_skipped == 0
    assert stats.active_corpus_size == 2


def test_seed_wikihow_corpus_raises_when_reuse_listing_fails(tmp_path) -> None:
    corpus_path = tmp_path / "wikihow_procedures.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {
                "id": "wh_001",
                "title": "How to Brew Tea",
                "content": "1. Heat water\n2. Steep leaves",
                "category": "Food and Entertaining",
            },
        ],
    )

    class FailingStore(FakeStore):
        def list_all(
            self, user_id: str | None = None, kind: str | None = None
        ) -> list[Procedure]:
            raise ValueError("store unavailable")

    memflow = FakeMemFlow()
    memflow.store = FailingStore()

    try:
        seed_wikihow_corpus(
            memflow=memflow,
            user_id="bench-user",
            corpus_path=corpus_path,
        )
    except RuntimeError as exc:
        assert "Failed to list existing procedures" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when reuse listing fails")

    assert memflow.added == []


def test_seed_wikihow_corpus_clear_existing_reseeds_all_valid_records(
    tmp_path,
) -> None:
    corpus_path = tmp_path / "wikihow_procedures.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {
                "id": "wh_001",
                "title": "How to Brew Tea",
                "content": "1. Heat water\n2. Steep leaves",
                "category": "Food and Entertaining",
            },
            {
                "id": "wh_002",
                "title": "How to Repair a Zipper",
                "content": "1. Inspect slider\n2. Realign teeth",
                "category": "Clothing",
            },
        ],
    )
    memflow = FakeMemFlow(
        existing=[
            Procedure(
                id="wh_001",
                title="Existing Tea",
                content="old",
                user_id="bench-user",
            ),
            Procedure(
                id="unrelated",
                title="Unrelated",
                content="old",
                user_id="bench-user",
            ),
            Procedure(
                id="wh_002",
                title="Other User Zipper",
                content="old",
                user_id="other-user",
            ),
        ]
    )

    stats = seed_wikihow_corpus(
        memflow=memflow,
        user_id="bench-user",
        corpus_path=corpus_path,
        clear_existing=True,
    )

    assert memflow.store.deleted == ["wh_001"]
    assert [proc.id for proc in memflow.added] == ["wh_001", "wh_002"]
    assert stats.num_records == 2
    assert stats.num_deleted == 1
    assert stats.num_seeded == 2
    assert stats.num_reused == 0
    assert stats.num_skipped == 0
    assert stats.active_corpus_size == 2


def test_wikihow_loader_and_conversion_preserve_procedure_fields(tmp_path) -> None:
    corpus_path = tmp_path / "wikihow_procedures.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {
                "id": "wh_001",
                "title": "How to Brew Tea",
                "content": "1. Heat water\n2. Steep leaves",
                "category": "Food and Entertaining",
                "tags": ["tea", "kitchen"],
                "metadata": {"source_url": "https://example.test/tea"},
            },
            {
                "id": "",
                "title": "Missing ID",
                "content": "Skipped by seed",
                "category": "Other",
                "tags": [],
                "metadata": {},
            },
        ],
    )

    records = list(iter_wikihow_procedures(corpus_path))
    procedure = wikihow_record_to_procedure(records[0], user_id="bench-user")

    assert records[0].metadata == {"source_url": "https://example.test/tea"}
    assert procedure.id == "wh_001"
    assert procedure.title == "How to Brew Tea"
    assert procedure.content == "1. Heat water\n2. Steep leaves"
    assert procedure.category == "Food and Entertaining"
    assert procedure.tags == ["tea", "kitchen"]
    assert procedure.user_id == "bench-user"

    existing = [
        Procedure(
            id="wh_001",
            title="Old Tea",
            content="old",
            user_id="bench-user",
        ),
        Procedure(
            id="unrelated",
            title="Unrelated",
            content="old",
            user_id="bench-user",
        ),
        Procedure(
            id="wh_001",
            title="Other User Tea",
            content="old",
            user_id="other-user",
        ),
    ]
    memflow = FakeMemFlow(existing=existing)

    stats = seed_wikihow_corpus(
        memflow=memflow,
        user_id="bench-user",
        corpus_path=corpus_path,
        clear_existing=True,
    )

    assert memflow.store.deleted == ["wh_001"]
    assert [proc.id for proc in memflow.added] == ["wh_001"]
    assert stats.num_records == 2
    assert stats.num_seeded == 1
    assert stats.num_reused == 0
    assert stats.num_skipped == 1
    assert stats.category_counts == {"Food and Entertaining": 1}


def test_wikihow_adapter_excludes_query_source_and_refills_top_k() -> None:
    source_proc = Procedure(
        id="source_a",
        title="Source Procedure",
        content="source",
        user_id="bench-user",
    )
    rel_proc = Procedure(
        id="rel_a",
        title="Relevant Procedure",
        content="relevant",
        user_id="bench-user",
    )
    next_proc = Procedure(
        id="next_a",
        title="Next Procedure",
        content="next",
        user_id="bench-user",
    )
    memflow = FakeSearchMemFlow(
        results=[
            SearchResult(procedure=source_proc, score=0.99),
            SearchResult(procedure=rel_proc, score=0.90),
            SearchResult(procedure=next_proc, score=0.80),
        ]
    )
    adapter = MemFlowWikiHowAdapter(
        memflow=memflow,
        user_id="bench-user",
        corpus_size=3,
        backend="pgvector",
        llm_provider="ollama",
        llm_model="test-model",
    )

    results = adapter.retrieve(
        "find the source task",
        k=2,
        exclude_procedure_ids={"source_a"},
    )

    assert memflow.search_calls == [
        {
            "query": "find the source task",
            "user_id": "bench-user",
            "top_k": 3,
            "kind": "procedure",
        }
    ]
    assert [result.procedure_id for result in results] == ["rel_a", "next_a"]


def test_wikihow_query_bank_loader_streams_jsonl_records(tmp_path) -> None:
    query_bank_path = tmp_path / "query_bank.jsonl"
    _write_jsonl(
        query_bank_path,
        [
            {
                "query_id": "q_001",
                "query": "make a cup of tea",
                "source_procedure_id": "wh_001",
                "relevant_procedure_ids": ["wh_001", "wh_002"],
                "source_metadata": {"category": "Food and Entertaining"},
                "relevance_notes": {"reason": "same procedure"},
                "rejected_close_candidates": ["wh_099"],
            },
            {
                "query_id": "q_002",
                "query": "repair a zipper",
                "source_procedure_id": "wh_050",
                "relevant_procedure_ids": ["wh_050"],
                "source_metadata": {"category": "Clothing"},
                "relevance_notes": None,
                "rejected_close_candidates": [],
            },
        ],
    )

    queries = load_wikihow_query_bank(query_bank_path, max_queries=1)

    assert len(queries) == 1
    assert queries[0].query_id == "q_001"
    assert queries[0].query == "make a cup of tea"
    assert queries[0].source_procedure_id == "wh_001"
    assert queries[0].relevant_procedure_ids == ["wh_001", "wh_002"]
    assert queries[0].source_metadata == {"category": "Food and Entertaining"}
    assert queries[0].rejected_close_candidates == ["wh_099"]


def test_binary_ir_metrics_use_full_relevant_set_denominators() -> None:
    metrics = compute_binary_ir_metrics(
        retrieved_ids=["rel_a", "not_rel"],
        relevant_ids=["rel_a", "rel_b", "rel_c"],
        k_values=[1, 2, 3],
    )

    assert metrics["num_relevant"] == 3
    assert metrics["num_relevant_retrieved"] == 1
    assert metrics["hit_at_k"]["3"] == 1.0
    assert metrics["precision_at_k"]["3"] == pytest.approx(1 / 3)
    assert metrics["recall_at_k"]["2"] == pytest.approx(1 / 3)
    assert metrics["average_precision"] == pytest.approx(1 / 3)
    assert metrics["reciprocal_rank"] == 1.0

    ideal_dcg_at_3 = 1.0 + (1.0 / math.log2(3)) + (1.0 / math.log2(4))
    assert metrics["ndcg_at_k"]["3"] == pytest.approx(1.0 / ideal_dcg_at_3)


def test_evaluation_stratifies_by_source_normalized_root_category() -> None:
    class FakeRetrieval:
        def retrieve(
            self,
            query: str,
            k: int = 5,
            exclude_procedure_ids: set[str] | None = None,
        ):
            return [
                RetrievedWikiHowProcedure(
                    procedure_id="rel_a",
                    title="Relevant A",
                    category="Cleaning Your Computer",
                    tags=[],
                    score=1.0,
                )
            ]

        def retrieve_batch(
            self,
            queries: list[tuple[str, set[str]]],
            k: int = 5,
        ):
            results = []
            for query, exclude in queries:
                result = [
                    RetrievedWikiHowProcedure(
                        procedure_id="rel_a",
                        title="Relevant A",
                        category="Cleaning Your Computer",
                        tags=[],
                        score=1.0,
                    )
                ]
                results.append(result)
            return results

    queries = [
        WikiHowBenchmarkQuery(
            query_id="q_001",
            query="clean my laptop keys",
            source_procedure_id="source_a",
            relevant_procedure_ids=["rel_a"],
            source_metadata={
                "source_normalized_root_category": "Computers and Electronics"
            },
            relevance_notes=[],
            rejected_close_candidates=[],
        )
    ]

    result = evaluate_wikihow_queries(
        retrieval_system=FakeRetrieval(),
        queries=queries,
        k_values=[1],
        top_k=1,
    )

    assert set(result.source_category_stratified_metrics) == {
        "Computers and Electronics"
    }
    assert (
        result.source_category_stratified_metrics["Computers and Electronics"][
            "hit_at_k"
        ]["1"]
        == 1.0
    )


def test_evaluation_applies_per_query_source_holdout() -> None:
    class FakeRetrieval:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def retrieve(
            self,
            query: str,
            k: int = 5,
            exclude_procedure_ids: set[str] | None = None,
        ):
            self.calls.append(
                {
                    "query": query,
                    "k": k,
                    "exclude_procedure_ids": set(exclude_procedure_ids or set()),
                }
            )
            candidates = [
                RetrievedWikiHowProcedure(
                    procedure_id="source_a",
                    title="Source",
                    category="Cleaning Your Computer",
                    tags=[],
                    score=1.0,
                ),
                RetrievedWikiHowProcedure(
                    procedure_id="rel_a",
                    title="Relevant A",
                    category="Cleaning Your Computer",
                    tags=[],
                    score=0.9,
                ),
            ]
            return [
                item
                for item in candidates
                if item.procedure_id not in (exclude_procedure_ids or set())
            ][:k]

        def retrieve_batch(
            self,
            queries: list[tuple[str, set[str]]],
            k: int = 5,
        ):
            results = []
            for query, exclude in queries:
                self.calls.append(
                    {
                        "query": query,
                        "k": k,
                        "exclude_procedure_ids": set(exclude or set()),
                    }
                )
                candidates = [
                    RetrievedWikiHowProcedure(
                        procedure_id="source_a",
                        title="Source",
                        category="Cleaning Your Computer",
                        tags=[],
                        score=1.0,
                    ),
                    RetrievedWikiHowProcedure(
                        procedure_id="rel_a",
                        title="Relevant A",
                        category="Cleaning Your Computer",
                        tags=[],
                        score=0.9,
                    ),
                ]
                result = [
                    item
                    for item in candidates
                    if item.procedure_id not in (exclude or set())
                ][:k]
                results.append(result)
            return results

    retrieval = FakeRetrieval()
    queries = [
        WikiHowBenchmarkQuery(
            query_id="q_001",
            query="clean my laptop keys",
            source_procedure_id="source_a",
            relevant_procedure_ids=["rel_a"],
            source_metadata={},
        )
    ]

    result = evaluate_wikihow_queries(
        retrieval_system=retrieval,
        queries=queries,
        k_values=[1],
        top_k=1,
    )

    assert retrieval.calls == [
        {
            "query": "clean my laptop keys",
            "k": 1,
            "exclude_procedure_ids": {"source_a"},
        }
    ]
    assert result.query_results[0]["heldout_procedure_ids"] == ["source_a"]
    assert result.query_results[0]["retrieved"][0]["procedure_id"] == "rel_a"
    assert result.overall_metrics["hit_at_k"]["1"] == 1.0
