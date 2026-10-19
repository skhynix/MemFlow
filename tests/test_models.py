# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MemFlow data models."""

from memflow.models import (
    Procedure,
    SearchResult,
    skill_search_text,
)


class TestProcedure:
    """Tests for Procedure model."""

    def test_procedure_minimal(self):
        """Test creating a procedure with minimum required fields."""
        proc = Procedure(title="Test Title", content="1. Step one")
        assert proc.title == "Test Title"
        assert proc.content == "1. Step one"
        assert proc.id is not None
        assert proc.user_id == "default"
        assert proc.category == "general"
        assert proc.tags == []
        assert proc.kind == "skill"
        assert proc.source_path is None
        assert proc.metadata == {}
        assert proc.created_at is not None
        assert proc.updated_at is not None

    def test_procedure_full(self):
        """Test creating a procedure with all fields."""
        proc = Procedure(
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
        assert proc.title == "Full Procedure"
        assert proc.content == "1. First\n2. Second"
        assert proc.user_id == "user123"
        assert proc.category == "workflow"
        assert proc.tags == ["tag1", "tag2"]
        assert proc.kind == "procedure"
        assert proc.source_path == "/tmp/source.md"
        assert proc.metadata == {"key": "value"}
        assert proc.created_at == "2026-03-31T10:00:00"
        assert proc.updated_at == "2026-04-01T10:00:00"

    def test_procedure_unique_ids(self):
        """Test that each procedure gets a unique ID."""
        proc1 = Procedure(title="Test 1", content="1. Step")
        proc2 = Procedure(title="Test 2", content="1. Step")
        assert proc1.id != proc2.id

    def test_skill_search_text_includes_materialized_metadata(self):
        """Test skill search text includes supported skill metadata."""
        proc = Procedure(
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

        text = skill_search_text(proc)
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
        proc = Procedure(title="Test", content="1. Step")
        result = SearchResult(procedure=proc, score=0.85)
        assert result.procedure == proc
        assert result.score == 0.85

    def test_search_result_score_zero(self):
        """Test search result with zero score."""
        proc = Procedure(title="Test", content="1. Step")
        result = SearchResult(procedure=proc, score=0.0)
        assert result.score == 0.0

    def test_search_result_score_one(self):
        """Test search result with perfect score."""
        proc = Procedure(title="Test", content="1. Step")
        result = SearchResult(procedure=proc, score=1.0)
        assert result.score == 1.0

    def test_search_result_different_procedures(self):
        """Test search results with different procedures have different scores."""
        proc1 = Procedure(title="Deploy guide", content="1. Run deploy.sh")
        proc2 = Procedure(title="Install guide", content="1. Run install.sh")
        result1 = SearchResult(procedure=proc1, score=0.9)
        result2 = SearchResult(procedure=proc2, score=0.7)
        assert result1.procedure.title != result2.procedure.title
        assert result1.score != result2.score
