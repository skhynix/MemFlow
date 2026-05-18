# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from benchmark.wikihow_procedure_silver.adapter import (
    iter_wikihow_procedures,
    seed_wikihow_corpus,
    wikihow_record_to_procedure,
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
