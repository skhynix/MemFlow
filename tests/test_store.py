# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MemFlow storage backends."""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from memflow.models import Procedure
from memflow.store import EmulatedStore, FileStore, MemMachineStore, MemMachineBypass, PgVectorStore


class TestEmulatedStore:
    """Tests for in-memory store."""

    def test_add_and_search(self):
        """Test adding and searching procedures."""
        store = EmulatedStore()
        proc = Procedure(title="How to deploy", content="1. Run deploy.sh", tags=["deploy"])
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
        proc = Procedure(title="How to deploy", content="1. Run deploy.sh", tags=["deploy"])
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
            created_at="2026-03-31T10:00:00"
        )
        store.add(proc)

        # Read raw file and verify format
        filepath = os.path.join(temp_dir, f"{proc.id}.md")
        content = Path(filepath).read_text(encoding='utf-8')

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


class TestMemMachineStore:
    """Tests for MemMachine store (mocked)."""

    def test_add(self, memmachine_mock):
        """Test adding a procedure."""
        mock_client, mock_memory, mock_module = memmachine_mock

        # Mock search to return the procedure for get() which calls list_all()
        mock_memory.search.return_value = [
            {
                "id": "mm-episode-id",
                "content": "# Test\n\n1. Step",
                "metadata": {
                    "mm_type": "procedural",
                    "record_id": "proc-id-123",
                    "user_id": "default",
                    "category": "general",
                    "tags": "[]",
                    "created_at": "2026-03-31T10:00:00"
                }
            }
        ]

        with patch.dict("sys.modules", {"memmachine_client": mock_module}):
            store = MemMachineStore()
            proc = Procedure(title="Test", content="1. Step")
            store.add(proc)

        mock_memory.add.assert_called_once()
        # Verify procedure can be retrieved (indirectly confirms index population)
        retrieved = store.get("proc-id-123")
        assert retrieved is not None
        assert retrieved.title == "Test"

    def test_search(self, memmachine_mock):
        """Test searching procedures."""
        mock_client, mock_memory, mock_module = memmachine_mock

        mock_memory.search.return_value = [
            {
                "id": "mm-id-1",
                "content": "# Test Procedure\n\n1. Step one",
                "metadata": {
                    "mm_type": "procedural",
                    "record_id": "proc-id-1",
                    "user_id": "default",
                    "category": "general",
                    "tags": "[]",
                    "created_at": "2026-03-31T10:00:00"
                },
                "score": 0.85
            }
        ]

        with patch.dict("sys.modules", {"memmachine_client": mock_module}):
            store = MemMachineStore()
            results = store.search("test", top_k=5)

        assert len(results) == 1
        assert results[0].procedure.title == "Test Procedure"
        assert results[0].score == 0.85

    def test_search_filters_non_procedural(self, memmachine_mock):
        """Test that non-procedural items are filtered out."""
        mock_client, mock_memory, mock_module = memmachine_mock

        mock_memory.search.return_value = [
            {
                "id": "mm-id-1",
                "content": "# Test",
                "metadata": {"mm_type": "semantic", "record_id": "proc-id-1"},
                "score": 0.9
            }
        ]

        with patch.dict("sys.modules", {"memmachine_client": mock_module}):
            store = MemMachineStore()
            results = store.search("test")

        assert len(results) == 0

    def test_delete(self, memmachine_mock):
        """Test deleting a procedure."""
        mock_client, mock_memory, mock_module = memmachine_mock

        # Mock list_all to return a procedure and populate the index
        mock_memory.search.return_value = [
            {
                "id": "mm-episode-id",
                "content": "# Test\n\n1. Step",
                "metadata": {
                    "mm_type": "procedural",
                    "record_id": "proc-id-123",
                    "user_id": "default",
                    "category": "general",
                    "tags": "[]",
                    "created_at": "2026-03-31T10:00:00"
                }
            }
        ]

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

        with patch.dict("sys.modules", {"memmachine_client": mock_module}):
            store = MemMachineStore()
            result = store.delete("nonexistent-id")

        assert result is False

    def test_list_all(self, memmachine_mock):
        """Test listing all procedures."""
        mock_client, mock_memory, mock_module = memmachine_mock

        mock_memory.search.return_value = [
            {
                "id": "mm-id-1",
                "content": "# Proc 1\n\n1. Step",
                "metadata": {
                    "mm_type": "procedural",
                    "record_id": "proc-id-1",
                    "user_id": "default",
                    "category": "general",
                    "tags": "[]",
                    "created_at": "2026-03-31T10:00:00"
                }
            },
            {
                "id": "mm-id-2",
                "content": "# Proc 2\n\n1. Step",
                "metadata": {
                    "mm_type": "procedural",
                    "record_id": "proc-id-2",
                    "user_id": "default",
                    "category": "general",
                    "tags": "[]",
                    "created_at": "2026-03-31T10:00:00"
                }
            }
        ]

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
        with patch('memflow.store.register_vector') as mock_register, \
             patch('memflow.store.create_engine') as mock_create_engine:
            # Setup mock engine and connection
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_raw_conn = MagicMock()

            # Setup context manager
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            mock_conn.connection.dbapi_connection = mock_raw_conn

            mock_create_engine.return_value = mock_engine

            # Initialize PgVectorStore (register_vector should be called)
            with patch.dict('os.environ', {
                'PGVECTOR_EMBEDDING_API_BASE': 'http://test-api',
                'PGVECTOR_EMBEDDING_DIMENSIONS': '2560'
            }):
                store = PgVectorStore(base_url="postgresql://test:5432/testdb")

            # Verify register_vector was called with raw_conn
            mock_register.assert_called_once_with(mock_raw_conn)


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
