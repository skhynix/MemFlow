# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib

import pytest

from memflow.manager import MemFlow
from memflow.models import Procedure
from memflow.skills import (
    build_resource_manifest,
    build_skill_metadata,
    indexed_skill_render_parts,
    load_skill,
    parse_skill_frontmatter,
    skill_id,
)
from memflow.store import EmulatedStore


def _write_skill(root, text: str) -> str:
    source = root / "SKILL.md"
    source.write_text(text, encoding="utf-8")
    return text


def test_parse_skill_frontmatter_with_supported_and_unknown_fields():
    text = (
        "---\n"
        "name: commit-craft\n"
        "description: Split commits.\n"
        "tags: [git, commits]\n"
        "aliases: [patch series]\n"
        "custom_hint: keep me\n"
        "---\n"
        "# Commit Craft\n"
    )

    frontmatter, body = parse_skill_frontmatter(text)

    assert frontmatter["name"] == "commit-craft"
    assert frontmatter["tags"] == ["git", "commits"]
    assert frontmatter["custom_hint"] == "keep me"
    assert body == "# Commit Craft\n"


def test_parse_skill_frontmatter_missing_returns_full_text():
    text = "# Skill\n\nNo frontmatter."

    frontmatter, body = parse_skill_frontmatter(text)

    assert frontmatter == {}
    assert body == text


def test_parse_skill_frontmatter_rejects_invalid_yaml():
    with pytest.raises(ValueError, match="Invalid SKILL.md frontmatter"):
        parse_skill_frontmatter("---\nname: [unterminated\n---\n# Body\n")


def test_parse_skill_frontmatter_rejects_non_mapping_yaml():
    with pytest.raises(ValueError, match="frontmatter must be a mapping"):
        parse_skill_frontmatter("---\n- item\n---\n# Body\n")


def test_build_resource_manifest_records_only_auxiliary_manifests(tmp_path):
    root = tmp_path / "commit-craft"
    scripts = root / "scripts"
    references = root / "references"
    assets = root / "assets"
    scripts.mkdir(parents=True)
    references.mkdir()
    assets.mkdir()
    script = scripts / "split.py"
    reference = references / "guide.md"
    asset = assets / "diagram.txt"
    script.write_text("print('split')\n", encoding="utf-8")
    reference.write_text("# Guide\n", encoding="utf-8")
    asset.write_text("asset\n", encoding="utf-8")

    manifest = build_resource_manifest(root)

    assert manifest["has_aux_files"] is True
    assert manifest["scripts"][0] == {
        "path": "scripts/split.py",
        "sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "size": script.stat().st_size,
    }
    assert set(manifest["references"][0]) == {"path", "sha256", "size"}
    assert set(manifest["assets"][0]) == {"path", "sha256", "size"}


def test_load_skill_from_directory_preserves_raw_content_and_metadata(tmp_path):
    root = tmp_path / "commit-craft"
    root.mkdir()
    source_text = _write_skill(
        root,
        "---\n"
        "name: commit-craft\n"
        "description: Split code changes into coherent commits.\n"
        "category: development\n"
        "tags: [git, commits]\n"
        "aliases: [patch series]\n"
        "file_patterns: ['*.py']\n"
        "tools: [git]\n"
        "unknown: preserved\n"
        "---\n"
        "# Commit Craft\n\n"
        "Use focused commits.\n",
    )

    proc = load_skill(root, user_id="alice", trust_state="trusted")

    assert proc.id == skill_id(root / "SKILL.md", "commit-craft")
    assert proc.title == "commit-craft"
    assert proc.content == source_text
    assert proc.user_id == "alice"
    assert proc.category == "development"
    assert proc.tags == ["git", "commits"]
    assert proc.kind == "skill"
    assert proc.source_path == str((root / "SKILL.md").resolve())
    skill = proc.metadata["skill"]
    assert skill["root_path"] == str(root.resolve())
    assert skill["source_path"] == proc.source_path
    assert skill["relative_path"].endswith("commit-craft/SKILL.md")
    assert skill["frontmatter"]["unknown"] == "preserved"
    assert skill["aliases"] == ["patch series"]
    assert skill["file_patterns"] == ["*.py"]
    assert skill["tools"] == ["git"]
    assert skill["sha256"] == hashlib.sha256(source_text.encode()).hexdigest()
    assert proc.metadata["governance"]["trust_state"] == "trusted"
    assert proc.metadata["governance"]["mode"] == "instruction"
    assert proc.metadata["governance"]["warnings"] == []
    index = proc.metadata["index"]
    body = "# Commit Craft\n\nUse focused commits.\n"
    assert index["frontmatter_present"] is True
    assert index["body_offset"] == source_text.index(body)
    assert index["body_start_line"] == 11
    assert index["body_sha256"] == hashlib.sha256(body.encode()).hexdigest()
    assert index["search_text_sha256"]


def test_build_skill_metadata_records_body_index_without_frontmatter(tmp_path):
    root = tmp_path / "simple-skill"
    root.mkdir()
    source = root / "SKILL.md"
    source_text = _write_skill(root, "# Simple\n\nBody\n")

    metadata = build_skill_metadata(
        root_path=root.resolve(),
        source_path=source.resolve(),
        raw_text=source_text,
        frontmatter={},
        trust_state="trusted",
    )

    index = metadata["index"]
    assert index["frontmatter_present"] is False
    assert index["body_offset"] == 0
    assert index["body_start_line"] == 1
    assert index["body_sha256"] == hashlib.sha256(source_text.encode()).hexdigest()


def test_indexed_skill_render_parts_uses_body_offset_and_legacy_fallback(tmp_path):
    root = tmp_path / "render-skill"
    root.mkdir()
    source_text = _write_skill(
        root,
        "---\nname: render-skill\n---\n# Render Skill\n\nPINEAPPLE_MARKER\n",
    )
    proc = load_skill(root, trust_state="trusted")

    frontmatter, body, warnings = indexed_skill_render_parts(proc)

    assert frontmatter["name"] == "render-skill"
    assert body == "# Render Skill\n\nPINEAPPLE_MARKER\n"
    assert warnings == ()

    legacy_proc = Procedure(
        title=proc.title,
        content=source_text,
        id=proc.id,
        kind="skill",
        metadata={"skill": proc.metadata["skill"], "index": {}},
    )

    legacy_frontmatter, legacy_body, legacy_warnings = indexed_skill_render_parts(
        legacy_proc
    )

    assert legacy_frontmatter["name"] == "render-skill"
    assert legacy_body == source_text
    assert legacy_warnings == (f"legacy_skill_render_metadata_missing:{proc.id}",)


@pytest.mark.parametrize(
    ("trust_state", "mode", "warning_fragment"),
    [
        ("trusted", "instruction", None),
        ("unknown", "data", "unknown"),
        ("blocked", "blocked", "blocked"),
    ],
)
def test_load_skill_governance_mode_tracks_trust_state(
    tmp_path, trust_state, mode, warning_fragment
):
    root = tmp_path / trust_state
    root.mkdir()
    _write_skill(root, f"---\nname: {trust_state}\n---\n# Skill\n")

    proc = load_skill(root, trust_state=trust_state)
    governance = proc.metadata["governance"]

    assert governance["trust_state"] == trust_state
    assert governance["mode"] == mode
    if warning_fragment is None:
        assert governance["warnings"] == []
    else:
        assert any(warning_fragment in warning for warning in governance["warnings"])


def test_load_skill_without_frontmatter_uses_directory_name(tmp_path):
    root = tmp_path / "simple-skill"
    root.mkdir()
    _write_skill(root, "# Simple\n\nBody\n")

    proc = load_skill(root)

    assert proc.title == "simple-skill"
    assert proc.category == "skill"
    assert proc.tags == []
    assert proc.metadata["skill"]["description"] == ""


def test_load_skill_from_direct_skill_md_path(tmp_path):
    root = tmp_path / "direct-skill"
    root.mkdir()
    _write_skill(root, "---\nname: direct-skill\n---\n# Direct\n")

    proc = load_skill(root / "SKILL.md")

    assert proc.title == "direct-skill"
    assert proc.source_path == str((root / "SKILL.md").resolve())


def test_memflow_skill_api_add_search_get_list_and_sync(tmp_path, fake_llm):
    root = tmp_path / "commit-craft"
    root.mkdir()
    source = root / "SKILL.md"
    original = _write_skill(
        root,
        "---\n"
        "name: commit-craft\n"
        "description: Split commits.\n"
        "tags: [git]\n"
        "aliases: [patch series]\n"
        "---\n"
        "# Commit Craft\n\n"
        "Keep commits reviewable.\n",
    )
    manager = MemFlow(llm=fake_llm, store=EmulatedStore(), use_env=False)

    added = manager.add_skill(root, user_id="alice", trust_state="trusted")

    assert source.read_text(encoding="utf-8") == original
    assert added["event"] == "ADD_SKILL"
    assert added["kind"] == "skill"
    assert added["sha256"] == hashlib.sha256(original.encode()).hexdigest()
    assert added["governance"] == {
        "trust_state": "trusted",
        "mode": "instruction",
        "warnings": [],
    }
    assert added["source_path"] == str(source.resolve())
    assert added["root_path"] == str(root.resolve())
    listed = manager.list_skills(user_id="alice", trust_state="trusted")
    assert [proc.id for proc in listed] == [added["id"]]
    assert (
        manager.search_skills("patch series", user_id="alice")[0].procedure.id
        == (added["id"])
    )
    assert manager.get_skill("commit-craft").id == added["id"]
    assert manager.get_skill(added["id"], include_content=False).content == ""

    created_at = listed[0].created_at
    initial_updated_at = listed[0].updated_at
    initial_indexed_at = listed[0].metadata["index"]["indexed_at"]

    unchanged = manager.sync_skill(added["id"])
    unchanged_proc = manager.get_skill(added["id"])

    assert unchanged["event"] == "NOOP_SYNC_SKILL"
    assert unchanged["status"] == "noop"
    assert unchanged["reason"] == "sha256_unchanged"
    assert unchanged["sha256"] == added["sha256"]
    assert unchanged_proc.updated_at == initial_updated_at
    assert unchanged_proc.metadata["index"]["indexed_at"] == initial_indexed_at

    updated = original + "\nNew sync content.\n"
    source.write_text(updated, encoding="utf-8")
    synced = manager.sync_skill(added["id"])
    refreshed = manager.get_skill(added["id"])

    assert synced["event"] == "SYNC_SKILL"
    assert synced["sha256"] == hashlib.sha256(updated.encode()).hexdigest()
    assert refreshed.created_at == created_at
    assert refreshed.content == updated
    assert source.read_text(encoding="utf-8") == updated


def test_sync_skill_missing_source_returns_stale_without_reindexing(tmp_path, fake_llm):
    root = tmp_path / "missing-source"
    root.mkdir()
    source = root / "SKILL.md"
    original = _write_skill(
        root,
        "---\nname: missing-source\n---\n# Missing Source\n\nKeep snapshot.\n",
    )
    manager = MemFlow(llm=fake_llm, store=EmulatedStore(), use_env=False)
    added = manager.add_skill(root, user_id="alice", trust_state="trusted")
    before = manager.get_skill(added["id"])
    source.unlink()

    result = manager.sync_skill(source)
    stale = manager.get_skill(added["id"])

    assert result["event"] == "STALE_SYNC_SKILL"
    assert result["status"] == "stale"
    assert result["metadata_updated"] is True
    assert "missing" in result["warning"]
    assert stale.content == original
    assert stale.created_at == before.created_at
    assert stale.updated_at == before.updated_at
    assert (
        stale.metadata["index"]["indexed_at"] == before.metadata["index"]["indexed_at"]
    )
    assert (
        stale.metadata["skill"]["sha256"]
        == hashlib.sha256(original.encode()).hexdigest()
    )
    assert stale.metadata["skill"]["stale"] is True
    assert any(
        "missing" in warning for warning in stale.metadata["governance"]["warnings"]
    )

    source.write_text(original, encoding="utf-8")
    restored = manager.sync_skill(added["id"])
    restored_proc = manager.get_skill(added["id"])

    assert restored["event"] == "SYNC_SKILL"
    assert "stale" not in restored_proc.metadata["skill"]
