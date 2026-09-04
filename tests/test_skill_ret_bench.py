# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the SkillRet benchmark harness."""

from argparse import Namespace
from types import SimpleNamespace

from benchmark.skill_ret_bench.adapter import (
    normalize_skill_ret_record,
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
