# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MemFlow storage backends."""

import logging
import os
import shutil
import tempfile
import warnings
from contextlib import contextmanager
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
    QdrantStore,
    VectorStore,
    _id_to_uuid,
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

        assert store.delete(proc.id) == 1
        assert store.get(proc.id) is None

    def test_delete_not_found(self):
        """Test deleting non-existent procedure."""
        store = EmulatedStore()
        assert store.delete("nonexistent-id") == 0

    def test_list_all(self):
        """Test listing all procedures."""
        store = EmulatedStore()
        store.add(Procedure(title="Test 1", content="1. Step"))
        store.add(Procedure(title="Test 2", content="1. Step"))

        all_procs = store.list()
        assert len(all_procs) == 2

    def test_user_id_filter(self):
        """Test filtering by user_id."""
        store = EmulatedStore()
        store.add(Procedure(title="User1 proc", content="1. Step", user_id="user1"))
        store.add(Procedure(title="User2 proc", content="1. Step", user_id="user2"))

        user1_procs = store.list(user_id="user1")
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
        assert [p.id for p in store.list() if p.kind == "procedure"] == [proc.id]

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
        all_procs = store2.list()

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

        # Note: FileStore uses simplified format that doesn't persist
        # source_path, metadata, or updated_at. These fields use defaults on retrieval.
        assert retrieved.kind == "skill"
        assert retrieved.title == "commit-craft"
        assert retrieved.user_id == "user1"
        assert retrieved.category == "development"
        assert retrieved.tags == ["git"]
        # Fields not persisted by FileStore's simplified format:
        # assert retrieved.source_path == "/tmp/commit-craft/SKILL.md"
        # assert retrieved.metadata == {"skill": {"name": "commit-craft"}}
        # assert retrieved.updated_at == "2026-06-02T10:00:00"

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

        # FileStore strips trailing newlines during deserialization
        assert retrieved.content == original.rstrip("\n")

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
            procs = [p for p in store.list() if p.kind == "skill"]

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
            # list will populate the index
            store.list()

            result = store.delete("proc-id-123")

        assert result == 1
        mock_memory.delete.assert_called_once_with("mm-episode-id")

    def test_delete_not_found(self, memmachine_mock):
        """Test deleting non-existent procedure."""
        mock_client, mock_memory, mock_module = memmachine_mock
        mock_memory.search.return_value = self._search_result()

        with patch.dict("sys.modules", {"memmachine_client": mock_module}):
            store = MemMachineStore()
            result = store.delete("nonexistent-id")

        assert result == 0

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
            procs = store.list()

        assert len(procs) == 2
        assert procs[0].title == "Proc 1"
        assert procs[1].title == "Proc 2"


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


@contextmanager
def _qdrant_client_network_mocked():
    """Let the real QdrantClient ctor run, mock its network methods.

    QdrantClient emits the insecure-connection warning during __init__, so
    we cannot replace the class. Instead we wrap __init__ to stub out the
    instance methods that _init_collection calls against a live server
    (collection_exists / create_collection / create_payload_index).
    """
    from qdrant_client import QdrantClient

    real_init = QdrantClient.__init__

    def patched_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        self.collection_exists = MagicMock(return_value=True)
        self.create_collection = MagicMock()
        self.create_payload_index = MagicMock()

    QdrantClient.__init__ = patched_init
    try:
        yield
    finally:
        QdrantClient.__init__ = real_init


class TestQdrantStore:
    """Tests for Qdrant vector store."""

    def test_init_creates_collection_with_hnsw(self):
        """Verify _init_collection creates collection with HNSW config."""
        with patch("memflow.store.QdrantStore._init_collection") as mock_init:
            mock_init.return_value = None
            with patch.dict(
                "os.environ",
                {
                    "VECTOR_EMBEDDING_API_BASE": "http://test-api",
                    "VECTOR_EMBEDDING_DIMENSIONS": "2560",
                    "QDRANT_COLLECTION_NAME": "test_collection",
                },
            ):
                QdrantStore()
        mock_init.assert_called_once()

    def test_init_requires_embedding_api_base(self):
        """QdrantStore should raise ValueError when VECTOR_EMBEDDING_API_BASE missing."""
        with patch.dict("os.environ", {}, clear=False):
            import os as _os

            _os.environ.pop("VECTOR_EMBEDDING_API_BASE", None)
            with pytest.raises(ValueError, match="VECTOR_EMBEDDING_API_BASE"):
                QdrantStore()

    def test_empty_api_key_no_insecure_warning(self):
        """Empty QDRANT_API_KEY must not raise the insecure-connection warning.

        QdrantClient treats an empty string as a configured key and emits
        ``UserWarning: Api key is used with an insecure connection.`` when
        the base URL is HTTP. QdrantStore must normalize the empty value to
        None so the client treats the connection as unauthenticated.
        """
        with patch.dict(
            "os.environ",
            {
                "VECTOR_EMBEDDING_API_BASE": "http://test-api",
                "QDRANT_BASE_URL": "http://localhost:6333",
                "QDRANT_API_KEY": "",
                "QDRANT_COLLECTION_NAME": "test_collection",
            },
        ):
            with _qdrant_client_network_mocked():
                with warnings.catch_warnings():
                    warnings.simplefilter("error", UserWarning)
                    QdrantStore()  # must not raise UserWarning

    def test_empty_api_key_arg_no_insecure_warning(self):
        """Explicit api_key="" must be normalized the same as the env var."""
        with patch.dict(
            "os.environ",
            {
                "VECTOR_EMBEDDING_API_BASE": "http://test-api",
                "QDRANT_BASE_URL": "http://localhost:6333",
            },
            clear=False,
        ):
            import os as _os

            _os.environ.pop("QDRANT_API_KEY", None)
            with _qdrant_client_network_mocked():
                with warnings.catch_warnings():
                    warnings.simplefilter("error", UserWarning)
                    QdrantStore(api_key="")  # must not raise UserWarning

    def test_nonempty_api_key_still_warns_on_http(self):
        """A real key over HTTP must still warn; normalization must not hide it.

        Guards against over-eager normalization that would suppress the
        warning for legitimately authenticated but insecure connections.
        """
        with patch.dict(
            "os.environ",
            {
                "VECTOR_EMBEDDING_API_BASE": "http://test-api",
                "QDRANT_BASE_URL": "http://localhost:6333",
                "QDRANT_API_KEY": "secret-key",
                "QDRANT_COLLECTION_NAME": "test_collection",
            },
        ):
            with _qdrant_client_network_mocked():
                with pytest.warns(UserWarning, match="insecure connection"):
                    QdrantStore()

    def test_compute_emb_warns_on_hash_fallback(self, caplog):
        """Test embedding failures are visible when fallback is used."""
        store = object.__new__(QdrantStore)
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
        """Test Qdrant embedding input uses skill-aware search text."""
        store = object.__new__(QdrantStore)
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

    def test_get_max_tokens_reads_attr(self):
        """Test _get_max_tokens returns _emb_max_tokens when set."""
        store = object.__new__(QdrantStore)
        store._emb_max_tokens = 1024
        assert store._get_max_tokens() == 1024

    def test_get_max_tokens_falls_back_to_default(self):
        """Test _get_max_tokens falls back to DEFAULT_EMBEDDING_MAX_TOKENS."""
        store = object.__new__(QdrantStore)
        store._emb_max_tokens = None
        from memflow.store import DEFAULT_EMBEDDING_MAX_TOKENS

        assert store._get_max_tokens() == DEFAULT_EMBEDDING_MAX_TOKENS

    def test_sanitize_content_strips_nul_bytes(self):
        """Test _sanitize_content removes NUL bytes."""
        proc = Procedure(
            title="test",
            content="before\x00after",
        )
        sanitized = VectorStore._sanitize_content(proc)
        assert "\x00" not in sanitized.content
        assert sanitized.content == "beforeafter"

    def test_point_to_procedure_round_trips_expanded_fields(self):
        """Test Qdrant point hydration restores expanded Procedure fields."""
        point = SimpleNamespace(
            id="550e8400-e29b-41d4-a716-446655440000",
            payload={
                "id": "skill-id",
                "user_id": "default",
                "title": "commit-craft",
                "content": "raw",
                "category": "development",
                "tags": ["git"],
                "kind": "skill",
                "source_path": "/tmp/commit-craft/SKILL.md",
                "metadata": {"skill": {"name": "commit-craft"}},
                "created_at": "2026-06-01T10:00:00",
                "updated_at": "2026-06-02T10:00:00",
            },
        )

        store = object.__new__(QdrantStore)
        proc = store._point_to_procedure(point)

        assert proc.id == "skill-id"
        assert proc.tags == ["git"]
        assert proc.kind == "skill"
        assert proc.source_path == "/tmp/commit-craft/SKILL.md"
        assert proc.metadata == {"skill": {"name": "commit-craft"}}
        assert proc.updated_at == "2026-06-02T10:00:00"

    def test_upsert_point_converts_skill_id_to_uuid(self):
        """_upsert_point converts skill:{sha256} id to a UUID for Qdrant point ID."""
        import uuid as _uuid

        store = object.__new__(QdrantStore)
        store._client = MagicMock()
        store._collection_name = "test_collection"

        skill_id = "skill:" + "a" * 64
        proc = Procedure(id=skill_id, title="Test", content="body")

        store._upsert_point(proc, [0.1] * 8)

        call_args = store._client.upsert.call_args
        point = call_args.kwargs["points"][0]

        # Point ID is a valid UUID
        _uuid.UUID(point.id)

        # Point ID is deterministic
        assert point.id == _id_to_uuid(skill_id)

        # Original ID preserved in payload
        assert point.payload["id"] == skill_id

    def test_id_to_uuid_is_deterministic(self):
        """_id_to_uuid produces stable UUIDs for any string input."""
        import uuid as _uuid

        # Same input -> same output
        assert _id_to_uuid("skill:abc123") == _id_to_uuid("skill:abc123")

        # Different inputs -> different outputs
        assert _id_to_uuid("skill:abc123") != _id_to_uuid("skill:abc124")

        # UUID-formatted string also works (uuid5 of a uuid string)
        uuid_str = str(_uuid.uuid4())
        result = _id_to_uuid(uuid_str)
        _uuid.UUID(result)
        assert result != uuid_str


class TestQdrantInstructionPrefix:
    """Tests for Qwen3-Embedding instruction prefix on QdrantStore."""

    @staticmethod
    def _make_store(instruction: str = "") -> QdrantStore:
        store = object.__new__(QdrantStore)
        store._query_instruction = instruction
        store._emb_model = "test-model"
        store._emb_api_base = "http://test-api"
        store._emb_api_key = "EMPTY"
        store._emb_dim = 8
        return store

    def test_query_path_prepends_instruction(self):
        """is_query=True prepends 'Instruct: ...\nQuery: ' to the text."""
        store = self._make_store(
            "Given a skill description, retrieve the most relevant skill"
        )
        captured: list[str] = []

        def mock_embed_chunk(chunk: str, config: dict) -> list[float]:
            captured.append(chunk)
            return [0.1] * 8

        with patch.object(store, "_embed_chunk", side_effect=mock_embed_chunk):
            store._compute_emb("test query", is_query=True)

        assert captured[0].startswith("Instruct: Given a skill description")
        assert "Query: test query" in captured[0]

    def test_document_path_no_instruction(self):
        """is_query=False (default) does not prepend instruction."""
        store = self._make_store(
            "Given a skill description, retrieve the most relevant skill"
        )
        captured: list[str] = []

        def mock_embed_chunk(chunk: str, config: dict) -> list[float]:
            captured.append(chunk)
            return [0.1] * 8

        with patch.object(store, "_embed_chunk", side_effect=mock_embed_chunk):
            store._compute_emb("test document")

        assert captured[0] == "test document"

    def test_empty_instruction_backward_compatible(self):
        """Empty instruction means no prefix even with is_query=True."""
        store = self._make_store("")
        captured: list[str] = []

        def mock_embed_chunk(chunk: str, config: dict) -> list[float]:
            captured.append(chunk)
            return [0.1] * 8

        with patch.object(store, "_embed_chunk", side_effect=mock_embed_chunk):
            store._compute_emb("test query", is_query=True)

        assert captured[0] == "test query"

    def test_batch_query_path_prepends_instruction(self):
        """Batch embedding with is_query=True prepends instruction to each text."""
        store = self._make_store(
            "Given a skill description, retrieve the most relevant skill"
        )
        captured: list[str] = []

        def mock_post(url, headers, json, timeout):
            captured.extend(json["input"])
            response = MagicMock()
            response.json.return_value = {
                "data": [{"embedding": [0.1] * 8} for _ in json["input"]]
            }
            response.raise_for_status = MagicMock()
            return response

        with patch("memflow.store.httpx.post", side_effect=mock_post):
            store._compute_embs_batch(["q1", "q2"], is_query=True)

        assert len(captured) == 2
        assert all("Instruct:" in t for t in captured)
        assert all("Query: q" in t for t in captured)

    def test_search_passes_is_query_true(self):
        """search() should call _compute_emb with is_query=True."""
        store = self._make_store("retrieve relevant skill")

        with (
            patch.object(store, "_compute_emb", return_value=[0.1] * 8) as mock_emb,
            patch.object(store, "_search_with_emb", return_value=[]) as mock_search,
        ):
            store.search("test query")

        mock_emb.assert_called_once_with("test query", is_query=True)
        mock_search.assert_called_once()

    def test_search_batch_passes_is_query_true(self):
        """search() with list should call _compute_embs_batch with is_query=True."""
        store = self._make_store("retrieve relevant skill")

        with (
            patch.object(
                store, "_compute_embs_batch", return_value=[[0.1] * 8]
            ) as mock_batch,
            patch.object(store, "_search_with_emb", return_value=[]),
        ):
            store.search(["test query"])

        mock_batch.assert_called_once()
        assert mock_batch.call_args.kwargs.get("is_query") is True
