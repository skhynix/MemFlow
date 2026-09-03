# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the SkillRet benchmark adapter."""

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
