# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math

import pytest

from benchmark.wikihow_procedure_silver.adapter import (
    RetrievedWikiHowProcedure,
    iter_wikihow_procedures,
    seed_wikihow_corpus,
    wikihow_record_to_procedure,
)
from benchmark.wikihow_procedure_silver.evaluation import (
    WikiHowBenchmarkQuery,
    compute_binary_ir_metrics,
    evaluate_wikihow_queries,
    load_wikihow_query_bank,
)
from memflow.models import Procedure


def _write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


class FakeStore:
    def __init__(self, existing: list[Procedure] | None = None) -> None:
        self.existing = existing or []
        self.deleted: list[str] = []

    def list_all(self, user_id: str | None = None) -> list[Procedure]:
        if user_id is None:
            return list(self.existing)
        return [proc for proc in self.existing if proc.user_id == user_id]

    def delete(self, procedure_id: str) -> bool:
        self.deleted.append(procedure_id)
        return True


class FakeMemFlow:
    def __init__(self, existing: list[Procedure] | None = None) -> None:
        self.store = FakeStore(existing=existing)
        self.added: list[Procedure] = []

    def add(self, procedure: Procedure) -> None:
        self.added.append(procedure)


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
        def list_all(self, user_id: str | None = None) -> list[Procedure]:
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
        def retrieve(self, query: str, k: int = 5):
            return [
                RetrievedWikiHowProcedure(
                    procedure_id="rel_a",
                    title="Relevant A",
                    category="Cleaning Your Computer",
                    tags=[],
                    score=1.0,
                )
            ]

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
