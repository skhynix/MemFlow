# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MemFlow storage backends."""

import logging
import os
import warnings
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memflow.models import SkillRecord
from memflow.store import (
    QdrantStore,
    VectorStore,
    _id_to_uuid,
)
from tests.fakes import FakeSkillStore


@contextmanager
def _qdrant_client_network_mocked():
    """Let the real QdrantClient ctor run, mock its network methods.

    QdrantClient emits the insecure-connection warning during __init__, so
    we cannot replace the class. Instead we wrap __init__ to stub out the
    instance methods that _init_collection calls against a live server
    (collection_exists / create_collection / create_payload_index /
    get_collection).
    """
    from qdrant_client import QdrantClient

    real_init = QdrantClient.__init__

    def patched_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        self.collection_exists = MagicMock(return_value=True)
        self.create_collection = MagicMock()
        self.create_payload_index = MagicMock()
        # Schema auto-detection reads the collection config — present it as a
        # legacy unnamed-vector collection (real servers return VectorParams,
        # not a dict) so no live request is made.
        from qdrant_client import models as _models

        self.get_collection = MagicMock(
            return_value=SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=_models.VectorParams(size=8, distance="Cosine"),
                        sparse_vectors=None,
                    )
                )
            )
        )

    QdrantClient.__init__ = patched_init
    try:
        yield
    finally:
        QdrantClient.__init__ = real_init


class TestQdrantStore:
    """Tests for Qdrant vector store."""

    @pytest.mark.parametrize(
        "env_collection,explicit_collection,expected",
        [
            (None, None, "memflow_skills"),
            ("env_skills", None, "env_skills"),
            (None, "explicit_skills", "explicit_skills"),
            ("env_skills", "explicit_skills", "explicit_skills"),
        ],
    )
    def test_collection_selection(
        self, env_collection, explicit_collection, expected, clean_env, monkeypatch
    ):
        if env_collection is not None:
            monkeypatch.setenv("QDRANT_COLLECTION_NAME", env_collection)
        with patch.object(QdrantStore, "_init_collection"):
            store = QdrantStore(
                collection_name=explicit_collection,
                emb_api_base="http://test-api",
                emb_dim=8,
            )
        assert store._collection_name == expected

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
                "MEMFLOW_HYBRID_SEARCH_ENABLED": "off",
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
                "MEMFLOW_HYBRID_SEARCH_ENABLED": "off",
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
                "MEMFLOW_HYBRID_SEARCH_ENABLED": "off",
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

        with patch.dict(os.environ, {"MEMFLOW_EMBEDDING_STRICT": "off"}):
            with patch("memflow.store.httpx.post", side_effect=RuntimeError("boom")):
                with caplog.at_level(logging.WARNING, logger="memflow.store"):
                    emb = store._compute_emb("deploy service")

        assert len(emb) == 8
        assert "RuntimeError: boom" in caplog.text
        assert "falling back to hash-based pseudo-embedding" in caplog.text

    def _strict_store(self) -> QdrantStore:
        """Bare QdrantStore configured for embedding tests (no network)."""
        store = object.__new__(QdrantStore)
        store._emb_model = "test-model"
        store._emb_api_base = "http://test-api"
        store._emb_api_key = "EMPTY"
        store._emb_dim = 8
        return store

    # Multi-chunk text (sentence-per-word filler forces _split_text_by_tokens
    # to emit several chunks at max_tokens=4).
    LONG_TEXT = "alpha word. " * 12

    def test_compute_emb_raises_in_strict_mode(self):
        """Test MEMFLOW_EMBEDDING_STRICT=on turns the single-chunk hash
        fallback into a raise."""
        store = self._strict_store()

        with patch.dict(os.environ, {"MEMFLOW_EMBEDDING_STRICT": "on"}):
            with patch("memflow.store.httpx.post", side_effect=RuntimeError("boom")):
                with pytest.raises(RuntimeError, match="boom"):
                    store._compute_emb("deploy service")

    def test_compute_emb_multi_chunk_raises_in_strict_mode(self):
        """Test strict mode covers the multi-chunk zero-vector fallback path."""
        store = self._strict_store()

        with patch.dict(os.environ, {"MEMFLOW_EMBEDDING_STRICT": "on"}):
            with patch("memflow.store.httpx.post", side_effect=RuntimeError("boom")):
                with pytest.raises(RuntimeError, match="boom"):
                    store._compute_emb(self.LONG_TEXT, max_tokens=4)

    def test_compute_emb_multi_chunk_zero_vector_fallback_default_off(self):
        """Test default behavior is unchanged on the multi-chunk path: a
        failed chunk becomes a zero vector and the mean still returns."""
        store = self._strict_store()

        # Sanity: the fixture text really takes the multi-chunk path.
        assert len(store._split_text_by_tokens(self.LONG_TEXT, 4)) > 1

        with patch.dict(os.environ, {"MEMFLOW_EMBEDDING_STRICT": "off"}):
            with patch("memflow.store.httpx.post", side_effect=RuntimeError("boom")):
                emb = store._compute_emb(self.LONG_TEXT, max_tokens=4)

        # Every chunk failed, so each became a zero vector; the mean is
        # zeros of the right width, and nothing raised.
        assert emb == [0.0] * 8

    def test_compute_emb_async_raises_in_strict_mode(self):
        """Test the async single-chunk path raises too — seeding writes
        through add_async."""
        import asyncio

        store = self._strict_store()

        with patch.dict(os.environ, {"MEMFLOW_EMBEDDING_STRICT": "on"}):
            with patch(
                "memflow.store.VectorStore._embed_chunk_async",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ):
                with pytest.raises(RuntimeError, match="boom"):
                    asyncio.run(store._compute_emb_async("deploy service"))

    def test_compute_emb_async_multi_chunk_raises_in_strict_mode(self):
        """Test the async multi-chunk zero-vector path also raises strict."""
        import asyncio

        store = self._strict_store()

        with patch.dict(os.environ, {"MEMFLOW_EMBEDDING_STRICT": "on"}):
            with patch(
                "memflow.store.VectorStore._embed_chunk_async",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ):
                with pytest.raises(RuntimeError, match="boom"):
                    asyncio.run(store._compute_emb_async(self.LONG_TEXT, max_tokens=4))

    # --- batch seeding path (add(list) -> _compute_embs_batch[_async]) ---
    #
    # Real seeding goes through the batch API, which has an intermediate
    # step the single path lacks: a failed batch logs a warning and falls
    # back to per-text embedding, and strict fires inside that fallback.
    # So the exception the caller sees is the *individual* failure, with
    # the batch failure pushed to __context__ ("During handling of the
    # above exception..."). These tests pin that shape, per code review.

    def test_compute_embs_batch_raises_in_strict_mode(self, caplog):
        """Test strict mode raises on the sync batch seeding path: the
        batch API fails, the per-text fallback re-raises, and the surface
        exception is the individual failure with the batch failure kept as
        __context__."""
        store = self._strict_store()

        def fake_post(url, headers=None, json=None, timeout=None):
            if isinstance(json["input"], list):
                raise RuntimeError("batch boom")
            raise RuntimeError("individual boom")

        with caplog.at_level(logging.WARNING, logger="memflow.store"):
            with patch.dict(os.environ, {"MEMFLOW_EMBEDDING_STRICT": "on"}):
                with patch("memflow.store.httpx.post", side_effect=fake_post):
                    with pytest.raises(RuntimeError, match="individual boom") as exc:
                        store._compute_embs_batch(["alpha", "beta"])

        # The real cause (batch API) survives as exception context.
        assert isinstance(exc.value.__context__, RuntimeError)
        assert "batch boom" in str(exc.value.__context__)
        # The intermediate fallback step is still logged.
        assert "Batch embedding API failed" in caplog.text

    def test_compute_embs_batch_async_raises_in_strict_mode(self, caplog):
        """Test strict mode raises on the async batch seeding path with the
        same exception shape as the sync path: individual failure on the
        surface, batch failure in __context__."""
        import asyncio

        store = self._strict_store()

        async def batch_post(*args, **kwargs):
            raise RuntimeError("batch boom")

        client = MagicMock()
        client.post = batch_post
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with caplog.at_level(logging.WARNING, logger="memflow.store"):
            with patch.dict(os.environ, {"MEMFLOW_EMBEDDING_STRICT": "on"}):
                with patch("memflow.store.httpx.AsyncClient", return_value=ctx):
                    with patch(
                        "memflow.store.VectorStore._embed_chunk_async",
                        new=AsyncMock(side_effect=RuntimeError("individual boom")),
                    ):
                        with pytest.raises(
                            RuntimeError, match="individual boom"
                        ) as exc:
                            asyncio.run(
                                store._compute_embs_batch_async(["alpha", "beta"])
                            )

        # Same chain as sync: the batch failure is the context, not the face.
        assert isinstance(exc.value.__context__, RuntimeError)
        assert "batch boom" in str(exc.value.__context__)
        # The intermediate fallback step is still logged.
        assert "Async batch embedding API failed" in caplog.text

    def test_to_text_uses_skill_search_text(self):
        """Test Qdrant embedding input uses skill-aware search text."""
        store = object.__new__(QdrantStore)
        record = SkillRecord(
            title="commit-craft",
            content="# Body",
            metadata={
                "skill": {
                    "description": "Split commits",
                    "aliases": ["patch series"],
                }
            },
        )

        text = store._to_text(record)

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
        record = SkillRecord(
            title="test",
            content="before\x00after",
        )
        sanitized = VectorStore._sanitize_content(record)
        assert "\x00" not in sanitized.content
        assert sanitized.content == "beforeafter"

    def test_point_to_skill_round_trips_expanded_fields(self):
        """Test Qdrant point hydration restores expanded SkillRecord fields."""
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
        record = store._point_to_skill(point)

        assert record.id == "skill-id"
        assert record.tags == ["git"]
        assert record.kind == "skill"
        assert record.source_path == "/tmp/commit-craft/SKILL.md"
        assert record.metadata == {"skill": {"name": "commit-craft"}}
        assert record.updated_at == "2026-06-02T10:00:00"

    def test_upsert_point_converts_skill_id_to_uuid(self):
        """_upsert_point converts skill:{sha256} id to a UUID for Qdrant point ID."""
        import uuid as _uuid

        store = object.__new__(QdrantStore)
        store._client = MagicMock()
        store._collection_name = "test_collection"
        store._named_schema = False
        store._sparse_enabled = False

        skill_id = "skill:" + "a" * 64
        record = SkillRecord(id=skill_id, title="Test", content="body")

        store._upsert_point(record, [0.1] * 8)

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


def _make_bare_qdrant_store(sparse_enabled: bool) -> QdrantStore:
    """Build a QdrantStore without touching the network.

    Bypasses ``__init__`` (which would open a client + create a collection) and
    wires only the attributes the search paths read. ``sparse_enabled=True``
    implies the named (dense+sparse) schema, mirroring detection in
    ``_init_collection``.
    """
    store = object.__new__(QdrantStore)
    store._collection_name = "test_collection"
    store._named_schema = sparse_enabled
    store._sparse_enabled = sparse_enabled
    store._client = MagicMock()
    store._client.query_points.return_value = SimpleNamespace(points=[])
    return store


class TestQdrantStoreHybrid:
    """Tests for QdrantStore hybrid (dense + sparse RRF) search."""

    def test_hybrid_mode_sends_two_prefetches_and_rrf(self):
        """hybrid issues a single query_points with dense+sparse prefetch + RrfQuery."""
        from qdrant_client import models

        store = _make_bare_qdrant_store(sparse_enabled=True)
        with (
            patch.object(store, "_compute_emb", return_value=[0.1] * 8),
            patch.object(
                store,
                "_compute_sparse",
                return_value=models.SparseVector(indices=[1], values=[1.0]),
            ),
        ):
            store.search_hybrid(
                "deploy", mode="hybrid", rrf_weights=(0.6, 0.4), rrf_k=60
            )

        store._client.query_points.assert_called_once()
        kwargs = store._client.query_points.call_args.kwargs
        assert isinstance(kwargs["query"], models.RrfQuery)
        assert len(kwargs["prefetch"]) == 2
        # weights preserve (dense, sparse) order
        assert list(kwargs["query"].rrf.weights) == [0.6, 0.4]
        assert kwargs["query"].rrf.k == 60

    def test_hybrid_scores_normalized_to_unit_range(self):
        """RRF rank-sum scores (~0.016 max) are rescaled so the top hit = 1.0.

        Regression test: SkillContextSelector filters results below
        min_score=0.2 (cosine-calibrated); raw RRF scores are ~0.01 and get
        entirely filtered, so hybrid must emit 0~1-scale scores.
        """
        from qdrant_client import models

        store = _make_bare_qdrant_store(sparse_enabled=True)
        raw_scores = [0.0164, 0.0148, 0.0082]
        points = [SimpleNamespace(score=s, payload={}) for s in raw_scores]
        store._client.query_points.return_value = SimpleNamespace(points=points)
        with (
            patch.object(store, "_compute_emb", return_value=[0.1] * 8),
            patch.object(
                store,
                "_compute_sparse",
                return_value=models.SparseVector(indices=[1], values=[1.0]),
            ),
            patch.object(
                store,
                "_point_to_skill",
                side_effect=lambda p: SkillRecord(title="t", content="c"),
            ),
        ):
            results = store.search_hybrid("deploy", mode="hybrid")

        scores = [r.score for r in results]
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0148 / 0.0164)
        assert scores[2] == pytest.approx(0.5)
        # rank order preserved
        assert scores == sorted(scores, reverse=True)

    def test_dense_mode_scores_not_rescaled(self):
        """dense mode keeps raw similarity scores (already 0~1 scale)."""
        store = _make_bare_qdrant_store(sparse_enabled=True)
        points = [SimpleNamespace(score=0.83, payload={})]
        store._client.query_points.return_value = SimpleNamespace(points=points)
        with (
            patch.object(store, "_compute_emb", return_value=[0.1] * 8),
            patch.object(
                store,
                "_point_to_skill",
                side_effect=lambda p: SkillRecord(title="t", content="c"),
            ),
        ):
            results = store.search_hybrid("deploy", mode="dense")

        assert results[0].score == pytest.approx(0.83)

    def test_dense_mode_queries_named_vector(self):
        """mode=dense queries the named 'dense' vector directly, no prefetch."""
        store = _make_bare_qdrant_store(sparse_enabled=True)
        with patch.object(store, "_compute_emb", return_value=[0.1] * 8):
            store.search_hybrid("deploy", mode="dense", rrf_top_k=5)

        kwargs = store._client.query_points.call_args.kwargs
        assert kwargs["using"] == "dense"
        assert "prefetch" not in kwargs
        assert kwargs["limit"] == 5

    def test_sparse_mode_queries_sparse_vector(self):
        """mode=sparse queries the named 'sparse' vector directly."""
        from qdrant_client import models

        store = _make_bare_qdrant_store(sparse_enabled=True)
        with patch.object(
            store,
            "_compute_sparse",
            return_value=models.SparseVector(indices=[1], values=[1.0]),
        ):
            store.search_hybrid("deploy", mode="sparse", rrf_top_k=7)

        kwargs = store._client.query_points.call_args.kwargs
        assert kwargs["using"] == "sparse"
        assert "prefetch" not in kwargs
        assert kwargs["limit"] == 7

    def test_hybrid_raises_without_sparse_for_non_dense_mode(self):
        """A dense-only collection cannot serve hybrid/sparse → RuntimeError."""
        store = _make_bare_qdrant_store(sparse_enabled=False)
        with pytest.raises(RuntimeError, match="sparse vector"):
            store.search_hybrid("deploy", mode="hybrid")

    def test_supports_hybrid_reflects_schema(self):
        """The supports_hybrid probe mirrors the detected sparse schema."""
        with_sparse = _make_bare_qdrant_store(sparse_enabled=True)
        without = _make_bare_qdrant_store(sparse_enabled=False)
        assert with_sparse.supports_hybrid is True
        assert without.supports_hybrid is False

    def test_hybrid_empty_sparse_query_degrades_to_dense(self):
        """A token-less query (sparse vec None) runs the dense leg alone."""
        store = _make_bare_qdrant_store(sparse_enabled=True)
        with (
            patch.object(store, "_compute_emb", return_value=[0.1] * 8),
            patch.object(store, "_compute_sparse", return_value=None),
        ):
            store.search_hybrid("how do I do this", mode="hybrid")

        kwargs = store._client.query_points.call_args.kwargs
        assert isinstance(kwargs["query"], list)  # raw dense vector, not RrfQuery
        assert "prefetch" not in kwargs

    def test_degraded_hybrid_keeps_raw_scores(self):
        """The token-less hybrid path must NOT rescale: the fused path's
        top-hit=1.0 normalization applied to cosine scores would turn a weak
        top hit (0.15) into a guaranteed 1.0, breaking min_score consumers.
        Degraded hybrid returns raw cosine scores, same as mode="dense"."""
        store = _make_bare_qdrant_store(sparse_enabled=True)
        points = [
            SimpleNamespace(score=0.15, payload={}),
            SimpleNamespace(score=0.1, payload={}),
        ]
        store._client.query_points.return_value = SimpleNamespace(points=points)
        with (
            patch.object(store, "_compute_emb", return_value=[0.1] * 8),
            patch.object(store, "_compute_sparse", return_value=None),
            patch.object(
                store,
                "_point_to_skill",
                side_effect=lambda p: SkillRecord(title="t", content="c"),
            ),
        ):
            results = store.search_hybrid("how do I do this", mode="hybrid")

        assert [r.score for r in results] == [0.15, 0.1]

    def test_sparse_mode_empty_query_returns_empty(self):
        """mode=sparse with a token-less query has nothing to match → []."""
        store = _make_bare_qdrant_store(sparse_enabled=True)
        with patch.object(store, "_compute_sparse", return_value=None):
            assert store.search_hybrid("the", mode="sparse") == []
        store._client.query_points.assert_not_called()

    def test_compute_sparse_returns_none_for_tokenless_text(self):
        """_compute_sparse maps an empty BM25 embedding to None (not an empty
        SparseVector, which Qdrant rejects)."""

        class _Tok:
            def __init__(self, n):
                self.indices = list(range(n))
                self.values = [1.0] * n

        class _FakeModel:
            def query_embed(self, text):
                yield _Tok(0)

            def embed(self, text):
                yield _Tok(0)

        store = _make_bare_qdrant_store(sparse_enabled=True)
        with patch.object(store, "_get_sparse_model", return_value=_FakeModel()):
            assert store._compute_sparse("the", is_query=True) is None
            assert store._compute_sparse("") is None

    def test_compute_sparse_batch_yields_none_entries_with_warning(self, caplog):
        """Tokenless docs become None entries (upsert then writes dense-only)
        and the count is surfaced in one warning."""

        class _Tok:
            def __init__(self, n):
                self.indices = list(range(n))
                self.values = [1.0] * n

        class _FakeModel:
            def embed(self, texts):
                yield _Tok(3)
                yield _Tok(0)

        store = _make_bare_qdrant_store(sparse_enabled=True)
        with patch.object(store, "_get_sparse_model", return_value=_FakeModel()):
            with caplog.at_level("WARNING", logger="memflow.store"):
                vecs = store._compute_sparse_batch(["a b c", "the"])

        assert vecs[0] is not None
        assert vecs[1] is None
        assert "1/2" in caplog.text

    def test_upsert_skips_sparse_when_none_on_named_schema(self):
        """sparse_vec=None on a named schema writes the dense vector only."""
        store = _make_bare_qdrant_store(sparse_enabled=True)
        record = SkillRecord(title="t", content="c")

        store._upsert_point(record, emb=[0.1] * 8, sparse_vec=None)

        point = store._client.upsert.call_args.kwargs["points"][0]
        assert set(point.vector.keys()) == {"dense"}

    def test_hnsw_ef_only_applies_to_dense_prefetch(self):
        """hnsw_ef tunes the dense HNSW traversal only; sparse prefetch omits it."""
        from qdrant_client import models

        store = _make_bare_qdrant_store(sparse_enabled=True)
        with (
            patch.object(store, "_compute_emb", return_value=[0.1] * 8),
            patch.object(
                store,
                "_compute_sparse",
                return_value=models.SparseVector(indices=[1], values=[1.0]),
            ),
        ):
            store.search_hybrid("deploy", mode="hybrid", hnsw_ef=128)

        kwargs = store._client.query_points.call_args.kwargs
        dense_p, sparse_p = kwargs["prefetch"]
        assert dense_p.params is not None and dense_p.params.hnsw_ef == 128
        assert sparse_p.params is None

    def test_dense_mode_works_on_legacy_collection(self):
        """mode=dense on a legacy unnamed collection uses using=None."""
        store = _make_bare_qdrant_store(sparse_enabled=False)
        with patch.object(store, "_compute_emb", return_value=[0.1] * 8):
            store.search_hybrid("deploy", mode="dense")

        kwargs = store._client.query_points.call_args.kwargs
        assert kwargs["using"] is None

    def test_unknown_mode_raises(self):
        store = _make_bare_qdrant_store(sparse_enabled=True)
        with patch.object(store, "_compute_emb", return_value=[0.1] * 8):
            with pytest.raises(ValueError, match="Unknown search mode"):
                store.search_hybrid("deploy", mode="bm25")

    def test_bad_rrf_weights_length_raises(self):
        from qdrant_client import models

        store = _make_bare_qdrant_store(sparse_enabled=True)
        with (
            patch.object(store, "_compute_emb", return_value=[0.1] * 8),
            patch.object(
                store,
                "_compute_sparse",
                return_value=models.SparseVector(indices=[1], values=[1.0]),
            ),
        ):
            with pytest.raises(ValueError, match="2 entries"):
                store.search_hybrid(
                    "deploy", mode="hybrid", rrf_weights=(0.5, 0.3, 0.2)
                )

    def test_existing_collection_schema_detected(self):
        """The store adopts the existing collection's schema, not env config."""
        store = object.__new__(QdrantStore)
        store._collection_name = "c"
        store._emb_dim = 8
        store._distance = "Cosine"
        store._index_type = "hnsw"
        store._hnsw_m = 16
        store._hnsw_ef_construct = 100
        store._named_schema = False
        store._sparse_enabled = False
        # User wants hybrid but the collection predates sparse support.
        store._create_with_sparse = True
        store._base_url = "http://localhost:6333"
        store._api_key = None
        from qdrant_client import models

        # A real server returns VectorParams (a model, not a dict) for a
        # single unnamed-vector collection — mirror that shape exactly.
        legacy = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=models.VectorParams(size=8, distance="Cosine"),
                    sparse_vectors=None,
                )
            )
        )
        hybrid = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={"dense": object()},
                    sparse_vectors={"sparse": object()},
                )
            )
        )
        client = MagicMock()
        client.collection_exists.return_value = True

        for info, named, sparse in [(legacy, False, False), (hybrid, True, True)]:
            client.get_collection.return_value = info
            with patch("qdrant_client.QdrantClient", return_value=client):
                store._init_collection()
            assert store._named_schema is named
            assert store._sparse_enabled is sparse

    def test_sparse_vector_written_on_upsert(self):
        """When sparse enabled, upsert stores both named dense + sparse vectors."""
        from qdrant_client import models

        store = object.__new__(QdrantStore)
        store._collection_name = "c"
        store._named_schema = True
        store._sparse_enabled = True
        store._client = MagicMock()
        record = SkillRecord(title="t", content="c")
        sparse = models.SparseVector(indices=[1, 2], values=[1.0, 2.0])

        store._upsert_point(record, emb=[0.1] * 8, sparse_vec=sparse)

        point = store._client.upsert.call_args.kwargs["points"][0]
        assert set(point.vector.keys()) == {"dense", "sparse"}

    def test_new_collection_created_named_when_hybrid_enabled(self):
        """A fresh collection with MEMFLOW_HYBRID_SEARCH_ENABLED=on gets named dense+sparse vectors."""
        store = object.__new__(QdrantStore)
        store._collection_name = "c"
        store._emb_dim = 8
        store._distance = "Cosine"
        store._index_type = "hnsw"
        store._hnsw_m = 16
        store._hnsw_ef_construct = 100
        store._named_schema = False
        store._sparse_enabled = False
        store._create_with_sparse = True
        store._base_url = "http://localhost:6333"
        store._api_key = None
        client = MagicMock()
        client.collection_exists.return_value = False
        with patch("qdrant_client.QdrantClient", return_value=client):
            store._init_collection()

        assert store._named_schema is True
        assert store._sparse_enabled is True
        kwargs = client.create_collection.call_args.kwargs
        assert set(kwargs["vectors_config"].keys()) == {"dense"}
        assert set(kwargs["sparse_vectors_config"].keys()) == {"sparse"}


class TestQdrantStoreBulkOps:
    """Tests for lightweight bulk primitives (count/list_ids/delete_many)."""

    def test_count_uses_server_side_count(self):
        store = _make_bare_qdrant_store(sparse_enabled=False)
        store._client.count.return_value = SimpleNamespace(count=12345)

        assert store.count(user_id="benchmark") == 12345

        store._client.count.assert_called_once()
        kwargs = store._client.count.call_args.kwargs
        assert kwargs["collection_name"] == "test_collection"
        assert kwargs["exact"] is True
        assert kwargs["count_filter"] is not None
        store._client.scroll.assert_not_called()

    def test_count_without_user_id_sends_no_filter(self):
        store = _make_bare_qdrant_store(sparse_enabled=False)
        store._client.count.return_value = SimpleNamespace(count=5)

        assert store.count() == 5
        assert store._client.count.call_args.kwargs["count_filter"] is None

    def test_list_ids_pages_with_id_only_payload(self):
        from qdrant_client import models

        store = _make_bare_qdrant_store(sparse_enabled=False)
        page1 = (
            [
                SimpleNamespace(payload={"id": "a"}),
                SimpleNamespace(payload={"id": "b"}),
            ],
            "offset-1",
        )
        page2 = ([SimpleNamespace(payload={"id": "c"})], None)
        store._client.scroll.side_effect = [page1, page2]

        assert store.list_ids(user_id="benchmark") == ["a", "b", "c"]

        assert store._client.scroll.call_count == 2
        for call in store._client.scroll.call_args_list:
            kwargs = call.kwargs
            assert kwargs["with_payload"] == ["id"]
            assert kwargs["with_vectors"] is False
            assert isinstance(kwargs["scroll_filter"], models.Filter)

    def test_list_ids_skips_points_without_id_payload_with_warning(self, caplog):
        store = _make_bare_qdrant_store(sparse_enabled=False)
        store._client.scroll.return_value = (
            [
                SimpleNamespace(payload={"id": "a"}),
                SimpleNamespace(payload={}),
                SimpleNamespace(payload=None),
            ],
            None,
        )

        with caplog.at_level(logging.WARNING, logger="memflow.store"):
            assert store.list_ids() == ["a"]

        assert "skipped 2 point" in caplog.text

    def test_delete_many_batches_single_request_per_chunk(self):
        from qdrant_client import models

        store = _make_bare_qdrant_store(sparse_enabled=False)
        ids = [f"id-{i}" for i in range(300)]

        num_deleted = store.delete_many(ids)

        assert num_deleted == 300
        assert store._client.delete.call_count == 2  # 256 + 44
        first = store._client.delete.call_args_list[0].kwargs
        assert len(first["points_selector"].points) == 256
        assert first["wait"] is True
        assert isinstance(first["points_selector"], models.PointIdsList)

    def test_delete_many_empty_is_noop(self):
        store = _make_bare_qdrant_store(sparse_enabled=False)

        assert store.delete_many([]) == 0
        store._client.delete.assert_not_called()

    def test_delete_many_maps_ids_to_deterministic_uuids(self):
        store = _make_bare_qdrant_store(sparse_enabled=False)

        store.delete_many(["skill-x"])

        points = store._client.delete.call_args.kwargs["points_selector"].points
        assert points == [_id_to_uuid("skill-x")]

    def test_count_propagates_client_errors(self):
        """count must surface failures instead of masking them as zero."""
        store = _make_bare_qdrant_store(sparse_enabled=False)
        store._client.count.side_effect = RuntimeError("qdrant down")

        with pytest.raises(RuntimeError, match="qdrant down"):
            store.count(user_id="benchmark")

    def test_delete_many_propagates_batch_errors_after_partial_apply(self):
        """Mid-loop failure aborts loudly; earlier batches are already applied."""
        store = _make_bare_qdrant_store(sparse_enabled=False)
        store._client.delete.side_effect = [None, RuntimeError("qdrant down")]

        ids = [f"id-{i}" for i in range(300)]
        with pytest.raises(RuntimeError, match="qdrant down"):
            store.delete_many(ids)

        # First batch (256 ids) went through before the failure.
        assert store._client.delete.call_count == 2
        first = store._client.delete.call_args_list[0].kwargs
        assert len(first["points_selector"].points) == 256


class TestBaseStoreBulkDefaults:
    """Contract tests for the default bulk helpers on non-Qdrant backends."""

    def test_fake_delete_many_counts_actual_deletions(self):
        """Defaults report removed ids, unlike Qdrant's submitted-id count."""
        store = FakeSkillStore()
        store.add(SkillRecord(id="a", title="A", content="x", user_id="u"))

        assert store.count(user_id="u") == 1
        assert store.list_ids(user_id="u") == ["a"]
        assert store.delete_many(["a", "missing"]) == 1
