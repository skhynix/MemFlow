# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MemFlow data models."""

from memflow.models import (
    SearchResult,
    SkillRecord,
    skill_search_text,
)


class TestSkillRecord:
    """Tests for SkillRecord model."""

    def test_skill_record_minimal(self):
        """Test creating a skill record with minimum required fields."""
        record = SkillRecord(title="Test Title", content="1. Step one")
        assert record.title == "Test Title"
        assert record.content == "1. Step one"
        assert record.id is not None
        assert record.user_id == "default"
        assert record.category == "general"
        assert record.tags == []
        assert record.kind == "skill"
        assert record.source_path is None
        assert record.metadata == {}
        assert record.created_at is not None
        assert record.updated_at is not None

    def test_skill_record_full(self):
        """Test creating a skill record with all fields."""
        record = SkillRecord(
            title="Full Procedure",
            content="1. First\n2. Second",
            user_id="user123",
            category="workflow",
            tags=["tag1", "tag2"],
            kind="procedure",
            source_path="/tmp/source.md",
            metadata={"key": "value"},
            created_at="2026-03-31T10:00:00",
            updated_at="2026-04-01T10:00:00",
        )
        assert record.title == "Full Procedure"
        assert record.content == "1. First\n2. Second"
        assert record.user_id == "user123"
        assert record.category == "workflow"
        assert record.tags == ["tag1", "tag2"]
        assert record.kind == "procedure"
        assert record.source_path == "/tmp/source.md"
        assert record.metadata == {"key": "value"}
        assert record.created_at == "2026-03-31T10:00:00"
        assert record.updated_at == "2026-04-01T10:00:00"

    def test_skill_record_unique_ids(self):
        """Test that each skill record gets a unique ID."""
        record1 = SkillRecord(title="Test 1", content="1. Step")
        record2 = SkillRecord(title="Test 2", content="1. Step")
        assert record1.id != record2.id

    def test_skill_search_text_includes_materialized_metadata(self):
        """Test skill search text includes supported skill metadata."""
        record = SkillRecord(
            title="commit-craft",
            content=(
                "---\n"
                "name: commit-craft\n"
                "unknown_hint: not boosted\n"
                "---\n"
                "# Commit Craft\n\n"
                "Split commits carefully."
            ),
            category="development",
            tags=["git", "commits"],
            metadata={
                "skill": {
                    "name": "commit-craft",
                    "description": "Split code changes into coherent commits.",
                    "aliases": ["patch series"],
                    "file_patterns": ["*.py"],
                    "tools": ["git"],
                    "relative_path": "commit-craft/SKILL.md",
                    "frontmatter": {"custom": "not boosted"},
                }
            },
        )

        text = skill_search_text(record)
        assert "Split code changes into coherent commits." in text
        assert "git commits" in text
        assert "patch series" in text
        assert "*.py" in text
        assert "commit-craft/SKILL.md" in text
        assert "not boosted" not in text
        assert "Split commits carefully." in text


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_search_result(self):
        """Test creating a search result."""
        record = SkillRecord(title="Test", content="1. Step")
        result = SearchResult(skill=record, score=0.85)
        assert result.skill == record
        assert result.score == 0.85

    def test_search_result_score_zero(self):
        """Test search result with zero score."""
        record = SkillRecord(title="Test", content="1. Step")
        result = SearchResult(skill=record, score=0.0)
        assert result.score == 0.0

    def test_search_result_score_one(self):
        """Test search result with perfect score."""
        record = SkillRecord(title="Test", content="1. Step")
        result = SearchResult(skill=record, score=1.0)
        assert result.score == 1.0

    def test_search_result_different_skills(self):
        """Test search results with different skills have different scores."""
        record1 = SkillRecord(title="Deploy guide", content="1. Run deploy.sh")
        record2 = SkillRecord(title="Install guide", content="1. Run install.sh")
        result1 = SearchResult(skill=record1, score=0.9)
        result2 = SearchResult(skill=record2, score=0.7)
        assert result1.skill.title != result2.skill.title
        assert result1.score != result2.score
