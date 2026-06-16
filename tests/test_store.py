# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MemFlow storage backends."""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from memflow.models import Procedure
from memflow.store import (
    EmulatedStore,
    FileStore,
    MemMachineBypass,
    MemMachineStore,
    PgVectorStore,
)


class TestEmulatedStore:
    """Tests for in-memory store."""

    def test_add_and_search(self):
        """Test adding and searching procedures."""
        store = EmulatedStore()
        proc = Procedure(
            title="How to deploy", content="1. Run deploy.sh", tags=["deploy"]
        )
        store.add(proc)

        results = store.search("deploy")
        assert len(results) == 1
        assert results[0].procedure.id == proc.id
        assert results[0].score > 0

    def test_search_no_results(self):
        """Test search with no matches."""
        store = EmulatedStore()
        store.add(Procedure(title="Test", content="1. Step"))

        results = store.search("nonexistent keyword xyz")
        assert len(results) == 0

    def test_get(self):
        """Test getting a procedure by ID."""
        store = EmulatedStore()
        proc = Procedure(title="Test", content="1. Step")
        store.add(proc)

        retrieved = store.get(proc.id)
        assert retrieved.id == proc.id
        assert retrieved.title == proc.title

    def test_get_not_found(self):
        """Test getting non-existent procedure."""
        store = EmulatedStore()
        assert store.get("nonexistent-id") is None

    def test_delete(self):
        """Test deleting a procedure."""
        store = EmulatedStore()
        proc = Procedure(title="Test", content="1. Step")
        store.add(proc)

        assert store.delete(proc.id) is True
        assert store.get(proc.id) is None

    def test_delete_not_found(self):
        """Test deleting non-existent procedure."""
        store = EmulatedStore()
        assert store.delete("nonexistent-id") is False

    def test_list_all(self):
        """Test listing all procedures."""
        store = EmulatedStore()
        store.add(Procedure(title="Test 1", content="1. Step"))
        store.add(Procedure(title="Test 2", content="1. Step"))

        all_procs = store.list_all()
        assert len(all_procs) == 2

    def test_user_id_filter(self):
        """Test filtering by user_id."""
        store = EmulatedStore()
        store.add(Procedure(title="User1 proc", content="1. Step", user_id="user1"))
        store.add(Procedure(title="User2 proc", content="1. Step", user_id="user2"))

        user1_procs = store.list_all(user_id="user1")
        assert len(user1_procs) == 1
        assert user1_procs[0].user_id == "user1"

    def test_search_top_k(self):
        """Test top_k limiting results."""
        store = EmulatedStore()
        for i in range(10):
            store.add(Procedure(title=f"Test {i}", content=f"1. Step {i} deploy"))

        results = store.search("deploy", top_k=3)
        assert len(results) == 3

    def test_kind_filter_for_search_and_list_all(self):
        """Test kind filtering on in-memory records."""
        store = EmulatedStore()
        skill = Procedure(title="Deploy skill", content="deploy with skill")
        proc = Procedure(
            title="Deploy procedure",
            content="deploy with procedure",
            kind="procedure",
        )
        store.add([skill, proc])

        assert [r.procedure.id for r in store.search("deploy")] == [skill.id]
        assert [r.procedure.id for r in store.search("deploy", kind="procedure")] == [
            proc.id
        ]
        assert {r.procedure.id for r in store.search("deploy", kind=None)} == {
            skill.id,
            proc.id,
        }
        assert [p.id for p in store.list_all(kind="procedure")] == [proc.id]

    def test_batch_search_respects_kind_filter(self):
        """Test kind filtering on batch search."""
        store = EmulatedStore()
        store.add(Procedure(title="Git skill", content="commit split"))
        store.add(
            Procedure(
                title="Git procedure",
                content="commit split",
                kind="procedure",
            )
        )

        results = store.search(["commit", "split"], kind="procedure")
        assert len(results) == 2
        assert all(len(batch) == 1 for batch in results)
        assert all(batch[0].procedure.kind == "procedure" for batch in results)


class TestFileStore:
    """Tests for file-based store."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        dirpath = tempfile.mkdtemp()
        yield dirpath
        shutil.rmtree(dirpath, ignore_errors=True)

    def test_add_and_persist(self, temp_dir):
        """Test that procedures persist to disk."""
        store = FileStore(file_dir=temp_dir)
        proc = Procedure(
            title="How to deploy", content="1. Run deploy.sh", tags=["deploy"]
        )
        store.add(proc)

        # Verify file exists
        assert os.path.exists(os.path.join(temp_dir, f"{proc.id}.md"))

        # Verify we can retrieve it
        retrieved = store.get(proc.id)
        assert retrieved.id == proc.id
        assert retrieved.title == proc.title

    def test_deserialize_file_format(self, temp_dir):
        """Test parsing the markdown file format."""
        store = FileStore(file_dir=temp_dir)
        proc = Procedure(
            title="Test Procedure",
            content="1. First step\n2. Second step",
            user_id="testuser",
            category="workflow",
            tags=["tag1", "tag2"],
            created_at="2026-03-31T10:00:00",
        )
        store.add(proc)

        # Read raw file and verify format
        filepath = os.path.join(temp_dir, f"{proc.id}.md")
        content = Path(filepath).read_text(encoding="utf-8")

        assert content.startswith("---")
        assert "id: " in content
        assert "user_id: testuser" in content
        assert "# Test Procedure" in content

    def test_list_all_loads_from_disk(self, temp_dir):
        """Test that list_all loads procedures from disk."""
        store = FileStore(file_dir=temp_dir)
        store.add(Procedure(title="Proc 1", content="1. Step"))
        store.add(Procedure(title="Proc 2", content="1. Step"))

        # Create new store instance pointing to same directory
        store2 = FileStore(file_dir=temp_dir)
        all_procs = store2.list_all()

        assert len(all_procs) == 2

    def test_search_filters_by_user_id(self, temp_dir):
        """Test search respects user_id filter."""
        store = FileStore(file_dir=temp_dir)
        store.add(Procedure(title="User1 deploy", content="1. Deploy", user_id="user1"))
        store.add(Procedure(title="User2 deploy", content="1. Deploy", user_id="user2"))

        results = store.search("deploy", user_id="user1")
        assert len(results) == 1
        assert results[0].procedure.user_id == "user1"

    def test_delete_removes_file(self, temp_dir):
        """Test that delete removes the file."""
        store = FileStore(file_dir=temp_dir)
        proc = Procedure(title="Test", content="1. Step")
        store.add(proc)

        filepath = os.path.join(temp_dir, f"{proc.id}.md")
        assert os.path.exists(filepath)

        store.delete(proc.id)
        assert not os.path.exists(filepath)

    def test_round_trips_skill_fields(self, temp_dir):
        """Test FileStore persists the expanded Procedure fields."""
        store = FileStore(file_dir=temp_dir)
        proc = Procedure(
            id="skill-id",
            title="commit-craft",
            content="---\nname: commit-craft\n---\n# Body",
            user_id="user1",
            category="development",
            tags=["git"],
            kind="skill",
            source_path="/tmp/commit-craft/SKILL.md",
            metadata={"skill": {"name": "commit-craft"}},
            created_at="2026-06-01T10:00:00",
            updated_at="2026-06-02T10:00:00",
        )

        store.add(proc)
        retrieved = store.get(proc.id)

        assert retrieved.kind == "skill"
        assert retrieved.source_path == "/tmp/commit-craft/SKILL.md"
        assert retrieved.metadata == {"skill": {"name": "commit-craft"}}
        assert retrieved.updated_at == "2026-06-02T10:00:00"

    def test_round_trips_raw_skill_content_exactly(self, temp_dir):
        """Test raw SKILL.md snapshots keep frontmatter-like text and final newlines."""
        store = FileStore(file_dir=temp_dir)
        original = "---\nname: commit-craft\n---\n# Body\n\nKeep the final newline.\n"
        proc = Procedure(
            id="skill-raw-id",
            title="commit-craft",
            content=original,
            user_id="user1",
            category="skill",
            tags=["git"],
            kind="skill",
            source_path="/tmp/commit-craft/SKILL.md",
            metadata={
                "skill": {
                    "name": "commit-craft",
                    "frontmatter": {"description": "contains --- marker"},
                    "sha256": "abc",
                }
            },
            created_at="2026-06-01T10:00:00",
            updated_at="2026-06-02T10:00:00",
        )

        store.add(proc)
        retrieved = store.get(proc.id)

        assert retrieved.content == original

    def test_legacy_files_default_new_fields(self, temp_dir):
        """Test old FileStore records load with compatible defaults."""
        path = Path(temp_dir) / "legacy.md"
        path.write_text(
            "---\n"
            "id: legacy\n"
            "user_id: default\n"
            "category: general\n"
            "tags: []\n"
            "created_at: 2026-06-01T10:00:00\n"
            "---\n"
            "# Legacy\n\n"
            "1. Step\n",
            encoding="utf-8",
        )
        store = FileStore(file_dir=temp_dir)

        proc = store.get("legacy")

        assert proc.kind == "skill"
        assert proc.source_path is None
        assert proc.metadata == {}
        assert proc.updated_at == proc.created_at


class TestMemMachineStore:
    """Tests for MemMachine store (mocked)."""

    @staticmethod
    def _episode(id: str, content: str, metadata: dict, score: float | None = None):
        return SimpleNamespace(
            id=id,
            content=content,
            metadata=metadata,
            score=score,
        )

    @staticmethod
    def _search_result(*episodes):
        return SimpleNamespace(
            content=SimpleNamespace(
                episodic_memory=SimpleNamespace(
                    long_term_memory=SimpleNamespace(episodes=list(episodes)),
                    short_term_memory=None,
                )
            )
        )

    def test_add(self, memmachine_mock):
        """Test adding a procedure."""
        mock_client, mock_memory, mock_module = memmachine_mock

        # Mock search to return the procedure for get() which calls list_all()
        mock_memory.search.return_value = self._search_result(
            self._episode(
                id="mm-episode-id",
                content="# Test\n\n1. Step",
                metadata={
                    "mm_type": "procedural",
                    "record_id": "proc-id-123",
                    "user_id": "default",
                    "category": "general",
                    "tags": "[]",
                    "created_at": "2026-03-31T10:00:00",
                },
            )
        )

        with patch.dict("sys.modules", {"memmachine_client": mock_module}):
            store = MemMachineStore()
            proc = Procedure(title="Test", content="1. Step")
            store.add(proc)

        mock_memory.add.assert_called_once()
        metadata = mock_memory.add.call_args.kwargs["metadata"]
        assert metadata["kind"] == "skill"
        assert metadata["metadata"] == "{}"
        # Verify procedure can be retrieved (indirectly confirms index population)
        retrieved = store.get("proc-id-123")
        assert retrieved is not None
        assert retrieved.title == "Test"

    def test_round_trips_expanded_metadata(self, memmachine_mock):
        """Test MemMachineStore persists the expanded Procedure fields."""
        mock_client, mock_memory, mock_module = memmachine_mock

        mock_memory.search.return_value = self._search_result(
            self._episode(
                id="mm-id-1",
                content="# Commit Craft\n\nraw skill text",
                metadata={
                    "mm_type": "procedural",
                    "record_id": "skill-id",
                    "user_id": "default",
                    "category": "development",
                    "tags": '["git"]',
                    "kind": "skill",
                    "source_path": "/tmp/commit-craft/SKILL.md",
                    "metadata": '{"skill": {"name": "commit-craft"}}',
                    "created_at": "2026-06-01T10:00:00",
                    "updated_at": "2026-06-02T10:00:00",
                },
            )
        )

        with patch.dict("sys.modules", {"memmachine_client": mock_module}):
            store = MemMachineStore()
            procs = store.list_all(kind="skill")

        assert len(procs) == 1
        assert procs[0].kind == "skill"
        assert procs[0].source_path == "/tmp/commit-craft/SKILL.md"
        assert procs[0].metadata == {"skill": {"name": "commit-craft"}}
        assert procs[0].updated_at == "2026-06-02T10:00:00"

    def test_search_filters_by_kind(self, memmachine_mock):
        """Test MemMachineStore applies kind filters client-side."""
        mock_client, mock_memory, mock_module = memmachine_mock

        mock_memory.search.return_value = self._search_result(
            self._episode(
                id="mm-skill",
                content="# Skill\n\ncommit split",
                metadata={
                    "mm_type": "procedural",
                    "record_id": "skill-id",
                    "kind": "skill",
                    "tags": "[]",
                },
                score=0.9,
            ),
            self._episode(
                id="mm-procedure",
                content="# Procedure\n\ncommit split",
                metadata={
                    "mm_type": "procedural",
                    "record_id": "procedure-id",
                    "kind": "procedure",
                    "tags": "[]",
                },
                score=0.8,
            ),
        )

        with patch.dict("sys.modules", {"memmachine_client": mock_module}):
            store = MemMachineStore()
            results = store.search("commit", kind="procedure")

        assert len(results) == 1
        assert results[0].procedure.id == "procedure-id"

    def test_search(self, memmachine_mock):
        """Test searching procedures."""
        mock_client, mock_memory, mock_module = memmachine_mock

        mock_memory.search.return_value = self._search_result(
            self._episode(
                id="mm-id-1",
                content="# Test Procedure\n\n1. Step one",
                metadata={
                    "mm_type": "procedural",
                    "record_id": "proc-id-1",
                    "user_id": "default",
                    "category": "general",
                    "tags": "[]",
                    "created_at": "2026-03-31T10:00:00",
                },
                score=0.85,
            )
        )

        with patch.dict("sys.modules", {"memmachine_client": mock_module}):
            store = MemMachineStore()
            results = store.search("test", top_k=5)

        assert len(results) == 1
        assert results[0].procedure.title == "Test Procedure"
        assert results[0].score == 0.85

    def test_search_filters_non_procedural(self, memmachine_mock):
        """Test that non-procedural items are filtered out."""
        mock_client, mock_memory, mock_module = memmachine_mock

        mock_memory.search.return_value = self._search_result(
            self._episode(
                id="mm-id-1",
                content="# Test",
                metadata={"mm_type": "semantic", "record_id": "proc-id-1"},
                score=0.9,
            )
        )

        with patch.dict("sys.modules", {"memmachine_client": mock_module}):
            store = MemMachineStore()
            results = store.search("test")

        assert len(results) == 0

    def test_delete(self, memmachine_mock):
        """Test deleting a procedure."""
        mock_client, mock_memory, mock_module = memmachine_mock

        # Mock list_all to return a procedure and populate the index
        mock_memory.search.return_value = self._search_result(
            self._episode(
                id="mm-episode-id",
                content="# Test\n\n1. Step",
                metadata={
                    "mm_type": "procedural",
                    "record_id": "proc-id-123",
                    "user_id": "default",
                    "category": "general",
                    "tags": "[]",
                    "created_at": "2026-03-31T10:00:00",
                },
            )
        )

        with patch.dict("sys.modules", {"memmachine_client": mock_module}):
            store = MemMachineStore()
            # list_all will populate the index
            store.list_all()

            result = store.delete("proc-id-123")

        assert result is True
        mock_memory.delete.assert_called_once_with("mm-episode-id")

    def test_delete_not_found(self, memmachine_mock):
        """Test deleting non-existent procedure."""
        mock_client, mock_memory, mock_module = memmachine_mock
        mock_memory.search.return_value = self._search_result()

        with patch.dict("sys.modules", {"memmachine_client": mock_module}):
            store = MemMachineStore()
            result = store.delete("nonexistent-id")

        assert result is False

    def test_list_all(self, memmachine_mock):
        """Test listing all procedures."""
        mock_client, mock_memory, mock_module = memmachine_mock

        mock_memory.search.return_value = self._search_result(
            self._episode(
                id="mm-id-1",
                content="# Proc 1\n\n1. Step",
                metadata={
                    "mm_type": "procedural",
                    "record_id": "proc-id-1",
                    "user_id": "default",
                    "category": "general",
                    "tags": "[]",
                    "created_at": "2026-03-31T10:00:00",
                },
            ),
            self._episode(
                id="mm-id-2",
                content="# Proc 2\n\n1. Step",
                metadata={
                    "mm_type": "procedural",
                    "record_id": "proc-id-2",
                    "user_id": "default",
                    "category": "general",
                    "tags": "[]",
                    "created_at": "2026-03-31T10:00:00",
                },
            ),
        )

        with patch.dict("sys.modules", {"memmachine_client": mock_module}):
            store = MemMachineStore()
            procs = store.list_all()

        assert len(procs) == 2
        assert procs[0].title == "Proc 1"
        assert procs[1].title == "Proc 2"


class TestPgVectorStore:
    """Tests for PostgreSQL + pgvector store."""

    def test_register_vector_called_on_init(self):
        """Verify register_vector() is called in _init_db."""
        with (
            patch("memflow.store.register_vector") as mock_register,
            patch("memflow.store.create_engine") as mock_create_engine,
        ):
            # Setup mock engine and connection
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_raw_conn = MagicMock()

            # Setup context manager
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            mock_conn.connection.dbapi_connection = mock_raw_conn

            mock_create_engine.return_value = mock_engine

            # Initialize PgVectorStore (register_vector should be called)
            with patch.dict(
                "os.environ",
                {
                    "PGVECTOR_EMBEDDING_API_BASE": "http://test-api",
                    "PGVECTOR_EMBEDDING_DIMENSIONS": "2560",
                },
            ):
                PgVectorStore(base_url="postgresql://test:5432/testdb")

            # Verify register_vector was called with raw_conn
            mock_register.assert_called_once_with(mock_raw_conn)
            executed_sql = "\n".join(
                str(call.args[0]) for call in mock_conn.execute.call_args_list
            )
            assert "kind TEXT NOT NULL DEFAULT 'skill'" in executed_sql
            assert "source_path TEXT" in executed_sql
            assert "metadata JSONB NOT NULL DEFAULT '{}'" in executed_sql
            assert "updated_at TEXT NOT NULL" in executed_sql
            assert "idx_procedures_kind" in executed_sql
            assert "idx_procedures_user_kind" in executed_sql
            assert "idx_procedures_source_path" in executed_sql

    def test_compute_emb_warns_on_hash_fallback(self, caplog):
        """Test embedding failures are visible when fallback is used."""
        store = object.__new__(PgVectorStore)
        store._emb_model = "test-model"
        store._emb_api_base = "http://test-api"
        store._emb_api_key = "EMPTY"
        store._emb_dim = 8

        with patch("memflow.store.httpx.post", side_effect=RuntimeError("boom")):
            with caplog.at_level(logging.WARNING, logger="memflow.store"):
                emb = store._compute_emb("deploy service")

        assert len(emb) == 8
        assert "RuntimeError: boom" in caplog.text
        assert "falling back to hash-based pseudo-embedding" in caplog.text

    def test_to_text_uses_skill_search_text(self):
        """Test PgVector embedding input uses skill-aware search text."""
        store = object.__new__(PgVectorStore)
        proc = Procedure(
            title="commit-craft",
            content="# Body",
            metadata={
                "skill": {
                    "description": "Split commits",
                    "aliases": ["patch series"],
                }
            },
        )

        text = store._to_text(proc)

        assert "Split commits" in text
        assert "patch series" in text

    def test_insert_persists_expanded_fields(self):
        """Test PgVector insert includes expanded Procedure fields."""
        store = object.__new__(PgVectorStore)
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        store._engine = mock_engine
        proc = Procedure(
            id="skill-id",
            title="commit-craft",
            content="raw",
            category="development",
            tags=["git"],
            source_path="/tmp/commit-craft/SKILL.md",
            metadata={"skill": {"name": "commit-craft"}},
            created_at="2026-06-01T10:00:00",
            updated_at="2026-06-02T10:00:00",
        )

        store._insert_procedure(proc, [0.1, 0.2])

        sql = str(mock_conn.execute.call_args.args[0])
        params = mock_conn.execute.call_args.args[1]
        assert "kind, source_path" in sql
        assert params["kind"] == "skill"
        assert params["source_path"] == "/tmp/commit-craft/SKILL.md"
        assert params["metadata"] == '{"skill": {"name": "commit-craft"}}'
        assert params["updated_at"] == "2026-06-02T10:00:00"

    def test_procedure_from_row_round_trips_expanded_fields(self):
        """Test PgVector row hydration restores expanded Procedure fields."""
        row = SimpleNamespace(
            id="skill-id",
            user_id="default",
            title="commit-craft",
            content="raw",
            category="development",
            tags='["git"]',
            kind="skill",
            source_path="/tmp/commit-craft/SKILL.md",
            metadata={"skill": {"name": "commit-craft"}},
            created_at="2026-06-01T10:00:00",
            updated_at="2026-06-02T10:00:00",
        )

        proc = PgVectorStore._procedure_from_row(row)

        assert proc.tags == ["git"]
        assert proc.kind == "skill"
        assert proc.source_path == "/tmp/commit-craft/SKILL.md"
        assert proc.metadata == {"skill": {"name": "commit-craft"}}
        assert proc.updated_at == "2026-06-02T10:00:00"


class TestMemMachineBypass:
    """Tests for MemMachine bypass bridge."""

    def test_add_semantic(self, memmachine_mock):
        """Test adding semantic content via bypass."""
        mock_client, mock_memory, mock_module = memmachine_mock

        with patch.dict("sys.modules", {"memmachine_client": mock_module}):
            bypass = MemMachineBypass()
            bypass.add("Some fact", memory_type="semantic", user_id="user1")

        mock_memory.add.assert_called_once()
        call_args = mock_memory.add.call_args
        assert call_args[1]["content"] == "Some fact"
        assert call_args[1]["metadata"]["mm_type"] == "semantic"

    def test_add_episodic(self, memmachine_mock):
        """Test adding episodic content via bypass."""
        mock_client, mock_memory, mock_module = memmachine_mock

        with patch.dict("sys.modules", {"memmachine_client": mock_module}):
            bypass = MemMachineBypass()
            bypass.add("Past event", memory_type="episodic", user_id="user1")

        call_args = mock_memory.add.call_args
        assert call_args[1]["metadata"]["mm_type"] == "episodic"
