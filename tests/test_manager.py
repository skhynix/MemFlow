# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MemFlow."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memflow.manager import MemFlow, _load_env_file
from memflow.models import Procedure, SearchResult
from tests.fakes import FakeSkillStore


class TestMemFlowAdd:
    """Tests for MemFlow.add()."""

    def test_add_procedure_direct(self):
        """Test adding a procedure directly."""
        manager = MemFlow(store=FakeSkillStore(), use_env=False)
        proc = Procedure(title="Test", content="1. Step")

        result = manager.add(procedure=proc)

        assert result["event"] == "ADD"
        assert result["id"] == proc.id
        assert proc in [p for p in manager.store.list()]


class TestMemFlowSearch:
    """Tests for MemFlow.search()."""

    def test_search_delegates_to_store(self):
        """Test that search delegates to store."""
        store = FakeSkillStore()
        store.add(Procedure(title="Deploy guide", content="1. Deploy"))
        manager = MemFlow(store=store, use_env=False)

        store.search_results = [SearchResult(p, 0.8) for p in store.list()]
        results = manager.search("deploy")

        assert len(results) == 1
        assert results[0].procedure.title == "Deploy guide"

    def test_search_with_user_id(self):
        """Test search filters by user_id."""
        store = FakeSkillStore()
        store.add(Procedure(title="User1 deploy", content="1. Deploy", user_id="user1"))
        store.add(Procedure(title="User2 deploy", content="1. Deploy", user_id="user2"))
        manager = MemFlow(store=store, use_env=False)

        store.search_results = [SearchResult(p, 0.8) for p in store.list()]
        results = manager.search("deploy", user_id="user1")

        assert len(results) == 1
        assert results[0].procedure.user_id == "user1"

    def test_search_with_top_k(self):
        """Test search limits results to top_k."""
        store = FakeSkillStore()
        for i in range(10):
            store.add(Procedure(title=f"Deploy {i}", content=f"1. Deploy {i}"))
        manager = MemFlow(store=store, use_env=False)

        store.search_results = [SearchResult(p, 0.8) for p in store.list()]
        results = manager.search("deploy", top_k=3)

        assert len(results) == 3

    def test_search_kind_filter(self):
        """Test MemFlow.search passes kind filters to the store."""
        store = FakeSkillStore()
        skill = Procedure(title="Deploy skill", content="deploy")
        procedure = Procedure(
            title="Deploy procedure",
            content="deploy",
            kind="procedure",
        )
        store.records.update({skill.id: skill, procedure.id: procedure})
        store.search_results = [SearchResult(p, 0.8) for p in store.list()]
        manager = MemFlow(store=store, use_env=False)

        assert [r.procedure.id for r in manager.search("deploy")] == [skill.id]
        for kind in ("procedure", None):
            with pytest.raises(ValueError):
                manager.search("deploy", kind=kind)
        assert len(store.search_calls) == 1


class TestMemFlowSearchReranker:
    """Tests for reranker integration in MemFlow.search()."""

    def test_search_no_reranker_by_default(self, clean_env):
        """With MEMFLOW_RERANKER_PIPELINE unset, search delegates directly to store."""
        store = FakeSkillStore()
        store.search_results = [
            SearchResult(Procedure(title="Deploy guide", content="1. Deploy"), 0.8)
        ]
        manager = MemFlow(store=store)

        results = manager.search("deploy")

        assert len(results) == 1
        assert results[0].procedure.title == "Deploy guide"
        assert manager._rerank_pipeline_obj is None  # never built

    def test_search_reranks_when_pipeline_enabled(self, clean_env):
        """MEMFLOW_RERANKER_PIPELINE=ce routes candidates through the rerank pipeline."""
        os.environ["MEMFLOW_RERANKER_PIPELINE"] = "ce"
        os.environ["MEMFLOW_RERANKER_FETCH_TOP_K"] = "10"
        store = FakeSkillStore()
        store.search_results = [
            SearchResult(Procedure(title="Deploy guide", content="1. Deploy"), 0.8)
        ]
        manager = MemFlow(store=store)

        mock_pipeline = MagicMock()
        mock_pipeline.rerank.return_value = [
            SearchResult(procedure=Procedure(title="reranked", content="x"), score=1.0)
        ]
        with patch.object(
            manager, "_get_rerank_pipeline", return_value=mock_pipeline
        ) as mock_get:
            results = manager.search("deploy", top_k=1)

        mock_pipeline.rerank.assert_called_once()
        mock_get.assert_called_once()
        assert len(results) == 1
        assert results[0].procedure.title == "reranked"

    def test_search_over_fetches_when_rerank_enabled(self, clean_env):
        """store.search receives top_k=max(top_k, candidate_k) when reranking."""
        os.environ["MEMFLOW_RERANKER_PIPELINE"] = "ce"
        os.environ["MEMFLOW_RERANKER_FETCH_TOP_K"] = "20"
        mock_store = MagicMock()
        mock_store.search.return_value = []
        manager = MemFlow(store=mock_store)

        manager.search("deploy", top_k=5)

        # First positional arg is query, keyword top_k must be over-fetched.
        call_kwargs = mock_store.search.call_args.kwargs
        assert call_kwargs["top_k"] == 20

    def test_search_no_over_fetch_when_rerank_disabled(self, clean_env):
        """Without reranking, store.search receives the caller's top_k."""
        mock_store = MagicMock()
        mock_store.search.return_value = []
        manager = MemFlow(store=mock_store)

        manager.search("deploy", top_k=5)

        assert mock_store.search.call_args.kwargs["top_k"] == 5

    def test_search_hybrid_respects_fetch_cap_server_side(self, clean_env):
        """Hybrid+rerank: FETCH_TOP_K < RRF_TOP_K must shrink the server-side
        rrf_top_k, not just truncate client-side."""
        os.environ["MEMFLOW_RERANKER_PIPELINE"] = "ce"
        os.environ["MEMFLOW_RERANKER_FETCH_TOP_K"] = "20"
        os.environ["MEMFLOW_HYBRID_SEARCH_ENABLED"] = "on"
        os.environ["MEMFLOW_HYBRID_RRF_TOP_K"] = "100"
        store = MagicMock()
        store.supports_hybrid = True
        store.search_hybrid.return_value = []
        manager = MemFlow(store=store)

        with patch.object(manager, "_get_rerank_pipeline") as mock_get:
            mock_get.return_value.rerank.return_value = []
            manager.search("deploy", top_k=5)

        kwargs = store.search_hybrid.call_args.kwargs
        # min(RRF=100, FETCH=20) = 20 reaches the server — previously the
        # helper re-inflated this back to RRF_TOP_K=100.
        assert kwargs["rrf_top_k"] == 20

    def test_search_fetch_below_rrf_warns_once(self, clean_env, caplog):
        """FETCH_TOP_K < RRF_TOP_K silently shrinks the candidate pool below
        the configured RRF depth — search() must warn once (every other
        silent degradation in this pipeline warns)."""
        import logging

        os.environ["MEMFLOW_RERANKER_PIPELINE"] = "ce"
        os.environ["MEMFLOW_RERANKER_FETCH_TOP_K"] = "20"
        os.environ["MEMFLOW_HYBRID_SEARCH_ENABLED"] = "on"
        os.environ["MEMFLOW_HYBRID_RRF_TOP_K"] = "100"
        store = MagicMock()
        store.supports_hybrid = True
        store.search_hybrid.return_value = []
        manager = MemFlow(store=store)

        with patch.object(manager, "_get_rerank_pipeline") as mock_get:
            mock_get.return_value.rerank.return_value = []
            with caplog.at_level(logging.WARNING, logger="memflow.manager"):
                manager.search("deploy", top_k=5)
                manager.search("build", top_k=5)

        warnings = [r for r in caplog.records if "FETCH_TOP_K" in r.message]
        assert len(warnings) == 1

    def test_search_deep_top_k_exceeds_fetch_pool(self, clean_env):
        """A caller asking deeper than the pool always gets its full request:
        max(min(RRF, FETCH), top_k) — the pool is a floor for retrieval, not
        a ceiling on the caller's top_k."""
        os.environ["MEMFLOW_RERANKER_PIPELINE"] = "ce"
        os.environ["MEMFLOW_RERANKER_FETCH_TOP_K"] = "20"
        os.environ["MEMFLOW_HYBRID_SEARCH_ENABLED"] = "on"
        os.environ["MEMFLOW_HYBRID_RRF_TOP_K"] = "100"
        store = MagicMock()
        store.supports_hybrid = True
        store.search_hybrid.return_value = []
        manager = MemFlow(store=store)

        with patch.object(manager, "_get_rerank_pipeline") as mock_get:
            mock_get.return_value.rerank.return_value = []
            manager.search("deploy", top_k=200)

        assert store.search_hybrid.call_args.kwargs["rrf_top_k"] == 200

    def test_search_batch_bypasses_reranker(self, clean_env):
        """Batch queries (list input) skip the reranker entirely."""
        os.environ["MEMFLOW_RERANKER_PIPELINE"] = "ce"
        mock_store = MagicMock()
        mock_store.search.return_value = []
        manager = MemFlow(store=mock_store)

        manager.search(["deploy", "build"], top_k=3)

        mock_store.search.assert_called_once()
        assert mock_store.search.call_args.kwargs["top_k"] == 3
        assert manager._rerank_pipeline_obj is None

    def test_misconfigured_pipeline_warns_and_keeps_order(self, clean_env, caplog):
        """PIPELINE=ce without RERANKER_CE_API_BASE must not crash search():
        warn once, degrade to retrieval order, and cache the fallback so the
        failed build is not retried on every query."""
        import logging

        os.environ["MEMFLOW_RERANKER_PIPELINE"] = "ce"
        os.environ.pop("RERANKER_CE_API_BASE", None)
        store = FakeSkillStore()
        store.search_results = [
            SearchResult(Procedure(title="Deploy guide", content="1. Deploy"), 0.8)
        ]
        manager = MemFlow(store=store)

        with caplog.at_level(logging.WARNING, logger="memflow.manager"):
            first = manager.search("deploy", top_k=1)
            second = manager.search("deploy", top_k=1)

        assert first[0].procedure.title == "Deploy guide"
        assert second[0].procedure.title == "Deploy guide"
        assert "failed to build" in caplog.text
        # The NoopReranker fallback is cached — both outcomes are memoized.
        from memflow.reranker import NoopReranker

        assert isinstance(manager._rerank_pipeline_obj, NoopReranker)

    def test_pipeline_built_once_under_concurrency(self, clean_env):
        """The double-checked lock must construct the pipeline at most once
        even when many threads hit a cold cache simultaneously."""
        import threading

        from memflow.reranker import RerankPipeline

        os.environ["MEMFLOW_RERANKER_PIPELINE"] = "ce"
        manager = MemFlow(store=FakeSkillStore())

        build_calls = []
        real_init = RerankPipeline.__init__

        def counting_init(self, *args, **kwargs):
            build_calls.append(threading.current_thread().name)
            real_init(self, *args, **kwargs)
            # Slow down construction to widen the race window.
            threading.Event().wait(0.05)

        with patch.object(RerankPipeline, "__init__", counting_init):
            threads = [
                threading.Thread(target=manager._get_rerank_pipeline) for _ in range(8)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(build_calls) == 1

    def test_search_rerank_false_skips_overfetch_and_reranker(self, clean_env):
        """rerank=False bypasses the pipeline even when globally enabled:
        no candidate over-fetch, no pipeline construction."""
        os.environ["MEMFLOW_RERANKER_PIPELINE"] = "ce"
        os.environ["MEMFLOW_RERANKER_FETCH_TOP_K"] = "20"
        hit = SearchResult(procedure=Procedure(title="Deploy", content="x"), score=1.0)
        mock_store = MagicMock()
        mock_store.search.return_value = [hit]
        manager = MemFlow(store=mock_store)

        manager.search("deploy", top_k=3, rerank=False)

        assert mock_store.search.call_args.kwargs["top_k"] == 3
        assert manager._rerank_pipeline_obj is None

        # Default stays True — same manager still over-fetches.
        with patch.object(manager, "_get_rerank_pipeline") as mock_get:
            mock_get.return_value.rerank.return_value = []
            manager.search("deploy", top_k=3)
        assert mock_store.search.call_args.kwargs["top_k"] == 20
        mock_get.assert_called_once()


class TestMemFlowDelete:
    """Tests for MemFlow.delete()."""

    def test_delete_removes_procedure(self):
        """Test deleting a procedure by ID."""
        store = FakeSkillStore()
        proc = Procedure(title="Deploy guide", content="1. Deploy")
        store.add(proc)
        manager = MemFlow(store=store, use_env=False)

        result = manager.delete(proc.id)

        assert result == {
            "id": proc.id,
            "title": proc.title,
            "event": "DELETE",
            "deleted": True,
        }
        assert store.get(proc.id) is None

    def test_delete_missing_id_returns_not_found(self):
        """Test deleting a missing ID returns not_found."""
        manager = MemFlow(store=FakeSkillStore(), use_env=False)

        result = manager.delete("missing-id")

        assert result == {
            "id": "missing-id",
            "event": "DELETE",
            "deleted": False,
            "error": "not_found",
        }

    def test_delete_batch_removes_multiple_procedures(self):
        """Test deleting multiple procedures by ID."""
        store = FakeSkillStore()
        kept = Procedure(title="Keep", content="1. Keep")
        deleted = [
            Procedure(title="Deploy guide", content="1. Deploy"),
            Procedure(title="Rollback guide", content="1. Roll back"),
        ]
        store.add([kept, *deleted])
        manager = MemFlow(store=store, use_env=False)

        result = manager.delete([deleted[0].id, deleted[1].id, "missing-id"])

        assert result == {
            "num_deleted": 2,
            "num_failed": 1,
            "total": 3,
        }
        assert store.get(deleted[0].id) is None
        assert store.get(deleted[1].id) is None
        assert store.get(kept.id) is kept

    def test_delete_respects_user_scope(self):
        """Test scoped delete does not remove another user's procedure."""
        store = FakeSkillStore()
        proc = Procedure(
            id="shared-looking-id",
            title="Private deploy guide",
            content="1. Deploy",
            user_id="user-a",
        )
        store.add(proc)
        manager = MemFlow(store=store, use_env=False)

        result = manager.delete(proc.id, user_id="user-b")

        assert result == {
            "id": proc.id,
            "event": "DELETE",
            "deleted": False,
            "error": "not_found",
        }
        assert store.get(proc.id) is proc


class TestLoadEnvFile:
    """Tests for _load_env_file() function including inline comment handling."""

    def test_inline_comment_stripped(self):
        """Test that inline comments are stripped from values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "TEST_VAR=http://example.com # This is a comment\n", encoding="utf-8"
            )

            # Clear env var before test
            os.environ.pop("TEST_VAR", None)

            _load_env_file(str(env_file))

            assert os.environ.get("TEST_VAR") == "http://example.com"

    def test_quoted_value_preserves_hash(self):
        """Test that quoted values preserve # character."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text('QUOTED_HASH="value#with#hash"\n', encoding="utf-8")

            os.environ.pop("QUOTED_HASH", None)
            _load_env_file(str(env_file))

            assert os.environ.get("QUOTED_HASH") == "value#with#hash"

    def test_single_quoted_value_preserves_hash(self):
        """Test that single-quoted values preserve # character."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "SINGLE_QUOTED_HASH='another#value'\n", encoding="utf-8"
            )

            os.environ.pop("SINGLE_QUOTED_HASH", None)
            _load_env_file(str(env_file))

            assert os.environ.get("SINGLE_QUOTED_HASH") == "another#value"

    def test_url_with_inline_comment(self):
        """Test that URL with inline comment is properly stripped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "VECTOR_EMBEDDING_API_BASE=http://10.78.59.136:8001/v1 # Required\n",
                encoding="utf-8",
            )

            os.environ.pop("VECTOR_EMBEDDING_API_BASE", None)
            _load_env_file(str(env_file))

            assert (
                os.environ.get("VECTOR_EMBEDDING_API_BASE")
                == "http://10.78.59.136:8001/v1"
            )

    def test_full_line_comment_ignored(self):
        """Test that full line comments are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "# This is a comment\nVALID_VAR=value\n# Another comment\n",
                encoding="utf-8",
            )

            os.environ.pop("VALID_VAR", None)
            _load_env_file(str(env_file))

            assert os.environ.get("VALID_VAR") == "value"

    def test_value_with_space_before_comment(self):
        """Test that value with space before comment is stripped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "TEST_VAR=test_value   # comment after spaces\n", encoding="utf-8"
            )

            os.environ.pop("TEST_VAR", None)
            _load_env_file(str(env_file))

            assert os.environ.get("TEST_VAR") == "test_value"

    def test_nonexistent_file_no_error(self):
        """Test that nonexistent .env file does not raise error."""
        # Should not raise any exception
        _load_env_file("/nonexistent/path/.env")

        # No exception means test passed
        assert True


class TestMemFlowHybridSearch:
    """Tests for MemFlow hybrid search routing and env config."""

    def test_single_query_routes_to_hybrid_when_enabled(self, clean_env):
        """Enabled + hybrid-capable store → single query uses search_hybrid."""
        os.environ["MEMFLOW_HYBRID_SEARCH_ENABLED"] = "on"
        store = MagicMock()
        store.supports_hybrid = True
        store.search_hybrid.return_value = []
        manager = MemFlow(store=store, use_env=True)

        manager.search("deploy", top_k=3)

        store.search_hybrid.assert_called_once()
        store.search.assert_not_called()
        kwargs = store.search_hybrid.call_args.kwargs
        assert kwargs["mode"] == "hybrid"
        assert kwargs["rrf_top_k"] >= 3  # never under-sized vs top_k

    def test_batch_query_uses_dense_even_when_enabled(self, clean_env):
        """Batch (list) queries always go through the dense store.search path."""
        os.environ["MEMFLOW_HYBRID_SEARCH_ENABLED"] = "on"
        store = MagicMock()
        store.search.return_value = []
        manager = MemFlow(store=store, use_env=True)

        manager.search(["a", "b"], top_k=3)

        store.search.assert_called_once()
        store.search_hybrid.assert_not_called()

    def test_single_query_uses_dense_when_disabled(self, clean_env):
        """Disabled → single query uses plain store.search."""
        store = MagicMock()
        store.search.return_value = []
        manager = MemFlow(store=store, use_env=True)

        manager.search("deploy", top_k=3)

        store.search.assert_called_once()
        store.search_hybrid.assert_not_called()

    def test_store_without_search_hybrid_falls_back(self, clean_env):
        """A store lacking search_hybrid (e.g. FakeSkillStore) uses dense path."""
        os.environ["MEMFLOW_HYBRID_SEARCH_ENABLED"] = "on"
        store = MagicMock(spec=["search"])  # no supports_hybrid attribute
        store.search.return_value = []
        manager = MemFlow(store=store, use_env=True)

        manager.search("deploy", top_k=3)

        store.search.assert_called_once()

    def test_store_without_sparse_falls_back_to_dense(self, clean_env):
        """Enabled but store.supports_hybrid is False (collection without
        sparse vector) → dense search, search_hybrid never called."""
        os.environ["MEMFLOW_HYBRID_SEARCH_ENABLED"] = "on"
        store = MagicMock()
        store.supports_hybrid = False
        store.search.return_value = []
        manager = MemFlow(store=store, use_env=True)

        results = manager.search("deploy", top_k=3)

        store.search_hybrid.assert_not_called()
        store.search.assert_called_once_with(
            "deploy", top_k=3, user_id=None, kind="skill"
        )
        assert results == []

    def test_search_hybrid_runtime_error_propagates(self, clean_env):
        """Capability is probed up front, so a RuntimeError from a capable
        store (e.g. connection failure) is a real failure and must propagate,
        not be silently downgraded to dense."""
        os.environ["MEMFLOW_HYBRID_SEARCH_ENABLED"] = "on"
        store = MagicMock()
        store.supports_hybrid = True
        store.search_hybrid.side_effect = RuntimeError("connection refused")
        manager = MemFlow(store=store, use_env=True)

        with pytest.raises(RuntimeError, match="connection refused"):
            manager.search("deploy", top_k=3)
        store.search.assert_not_called()

    def test_search_hybrid_truncates_to_top_k(self, clean_env):
        """_search_hybrid returns only top_k results from the deeper RRF pool."""
        os.environ["MEMFLOW_HYBRID_SEARCH_ENABLED"] = "on"
        store = MagicMock()
        store.supports_hybrid = True
        store.search_hybrid.return_value = [MagicMock() for _ in range(10)]
        manager = MemFlow(store=store, use_env=True)

        results = manager.search("deploy", top_k=3)

        assert len(results) == 3

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("0.6,0.4", (0.6, 0.4)),
            (" 0.5 , 0.5 ", (0.5, 0.5)),
            ("1,2,3", (0.6, 0.4)),  # wrong count → fallback
            ("abc", (0.6, 0.4)),  # unparseable → fallback
            ("0.7", (0.6, 0.4)),  # single value → fallback
            ("0.5,-0.5", (0.6, 0.4)),  # negative → fallback
            ("-1,2", (0.6, 0.4)),  # negative → fallback
        ],
    )
    def test_parse_rrf_weights(self, clean_env, raw, expected):
        os.environ["MEMFLOW_HYBRID_RRF_WEIGHTS"] = raw
        manager = MemFlow(store=FakeSkillStore(), use_env=True)
        assert manager._parse_rrf_weights() == expected

    def test_bad_int_env_falls_back(self, clean_env):
        """Malformed int env vars degrade to defaults instead of crashing."""
        os.environ["MEMFLOW_HYBRID_SPARSE_SEARCH_TOP_K"] = "not-a-number"
        os.environ["MEMFLOW_HYBRID_RRF_K"] = ""
        manager = MemFlow(store=FakeSkillStore(), use_env=True)
        assert manager._hybrid_sparse_search_top_k == 200
        assert manager._hybrid_rrf_k == 60
