# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the SkillRet benchmark harness."""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from types import SimpleNamespace

from benchmark.skill_ret_bench.adapter import (
    normalize_skill_ret_record,
    seed_skill_ret_corpus,
    skill_record_to_procedure,
)
from memflow.models import skill_search_text


def test_skillret_search_text_keeps_description_and_full_body() -> None:
    raw = {
        "id": "skill-id",
        "name": "commit-craft",
        "description": "Split code changes into coherent commits.",
        "skill_md": "---\nname: commit-craft\n---\n\n# Commit Craft\n\nUse git add.",
        "major": "Software Engineering",
        "sub": "Development",
    }

    record = normalize_skill_ret_record(raw)
    procedure = skill_record_to_procedure(record, user_id="benchmark")
    text = skill_search_text(procedure)

    assert record.content == raw["skill_md"]
    assert procedure.metadata["skill"]["name"] == raw["name"]
    assert procedure.metadata["skill"]["description"] == raw["description"]
    assert raw["description"] in text
    assert "Use git add." in text


def test_skillret_record_accepts_body_compatibility_alias() -> None:
    raw = {
        "id": "skill-id",
        "name": "commit-craft",
        "description": "Split commits.",
        "body": "# Commit Craft\n\nUse git add.",
    }

    record = normalize_skill_ret_record(raw)

    assert record.content == raw["body"]


def test_skillret_record_falls_back_from_empty_skill_md() -> None:
    raw = {
        "id": "skill-id",
        "name": "commit-craft",
        "description": "Split commits.",
        "skill_md": "",
    }

    record = normalize_skill_ret_record(raw)

    assert record.content == raw["description"]


def test_integrated_runner_awaits_corpus_seeding(monkeypatch, tmp_path) -> None:
    from benchmark.skill_ret_bench import run_skill_ret_bench as runner

    awaited = False

    async def fake_seed_skill_ret_corpus(**kwargs):
        nonlocal awaited
        awaited = True
        return SimpleNamespace(
            active_corpus_size=3,
            to_dict=lambda: {"num_seeded": 3},
        )

    args = Namespace(
        user_id="benchmark",
        k_values=[1],
        query_bank_path=None,
        corpus_path=tmp_path / "skills.jsonl",
        results_dir=tmp_path,
        results_filename="seed-result",
        seed_only=True,
        clear_existing=False,
        max_queries=None,
    )
    monkeypatch.setattr(runner, "_load_env_file", lambda *_: None)
    monkeypatch.setattr(runner, "_parse_args", lambda: args)
    monkeypatch.setattr(runner, "MemFlow", object)
    monkeypatch.setattr(runner, "seed_skill_ret_corpus", fake_seed_skill_ret_corpus)

    runner.main()

    assert awaited
    assert (tmp_path / "seed-result.json").exists()


def test_runners_default_to_official_evaluation_skill_split() -> None:
    from benchmark.skill_ret_bench import run_seeding, run_skill_ret_bench

    expected_suffix = "data/SKILLRET/data/skills/test.jsonl"

    assert run_seeding.DEFAULT_CORPUS_PATH.as_posix().endswith(expected_suffix)
    assert run_skill_ret_bench.DEFAULT_CORPUS_PATH.as_posix().endswith(expected_suffix)


class _RecordingStore:
    """Store double that records which bulk primitives the seeder uses."""

    def __init__(self, existing_ids: list[str]) -> None:
        self.existing_ids = set(existing_ids)
        self.list_called = False
        self.list_ids_calls = 0
        self.delete_many_calls: list[list[str]] = []
        self.deleted: list[str] = []

    def list(self, user_id: str | None = None):
        self.list_called = True
        raise AssertionError("seed must not fetch full procedure payloads via list()")

    def list_ids(self, user_id: str | None = None) -> list[str]:
        self.list_ids_calls += 1
        return sorted(self.existing_ids)

    def delete_many(self, ids: list[str]) -> int:
        self.delete_many_calls.append(list(ids))
        self.deleted.extend(ids)
        self.existing_ids -= set(ids)
        return len(ids)

    def delete(self, id):  # pragma: no cover - per-id path must not be used
        raise AssertionError("seed must not delete one id at a time")


class _RecordingMemFlow:
    def __init__(self, store: _RecordingStore) -> None:
        self.store = store
        self.added: list = []

    async def add_async(self, procedure, max_workers: int = 4) -> dict:
        procs = procedure if isinstance(procedure, list) else [procedure]
        self.added.extend(procs)
        return {"num_seeded": len(procs), "num_skipped": 0}


def _write_corpus(tmp_path):
    corpus_path = tmp_path / "skills.jsonl"
    records = [
        {"id": "s_001", "name": "Alpha", "skill_md": "body a"},
        {"id": "s_002", "name": "Beta", "skill_md": "body b"},
    ]
    corpus_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return corpus_path


def test_seed_clear_existing_uses_bulk_primitives(tmp_path) -> None:
    """--clear-existing must not pull full payloads nor delete per id.

    Per-id deletion costs two serial round trips per document (get +
    delete). The bulk path is list_ids() + one batched delete_many().
    """
    corpus_path = _write_corpus(tmp_path)
    store = _RecordingStore(existing_ids=["s_001", "unrelated"])
    memflow = _RecordingMemFlow(store)

    stats = asyncio.run(
        seed_skill_ret_corpus(
            memflow=memflow,
            user_id="bench-user",
            corpus_path=corpus_path,
            clear_existing=True,
        )
    )

    assert store.list_called is False
    assert store.list_ids_calls == 1
    # Only corpus ids that exist are deleted; foreign ids stay untouched.
    assert store.deleted == ["s_001"]
    assert stats.num_deleted == 1
    assert stats.num_seeded == 2
    assert [p.id for p in memflow.added] == ["s_001", "s_002"]


def test_seed_reuse_path_uses_list_ids(tmp_path) -> None:
    """Reuse listing must also use list_ids, not full-payload list()."""
    corpus_path = _write_corpus(tmp_path)
    store = _RecordingStore(existing_ids=["s_001"])
    memflow = _RecordingMemFlow(store)

    stats = asyncio.run(
        seed_skill_ret_corpus(
            memflow=memflow,
            user_id="bench-user",
            corpus_path=corpus_path,
        )
    )

    assert store.list_called is False
    assert stats.num_reused == 1
    assert stats.num_seeded == 1
    assert [p.id for p in memflow.added] == ["s_002"]
