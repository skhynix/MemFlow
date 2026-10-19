# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for unit tests."""

import os
from unittest.mock import patch

import pytest


@pytest.fixture
def clean_env(monkeypatch):
    """Clear relevant environment variables before test and prevent .env file loading.

    This fixture:
    1. Saves original environment variables
    2. Clears relevant env vars
    3. Patches _load_env_file to prevent .env file loading during tests
    4. Restores original environment variables after test
    """
    vars_to_clear = [
        "MEMFLOW_BACKEND",
        "MEMFLOW_EMBEDDING_STRICT",
        "EMBEDDING_REQUEST_TIMEOUT",
        "EMBEDDING_MAX_RETRIES",
        # Vector DB embedding config (shared)
        "VECTOR_EMBEDDING_MODEL",
        "VECTOR_EMBEDDING_API_BASE",
        "VECTOR_EMBEDDING_API_KEY",
        "VECTOR_EMBEDDING_DIMENSIONS",
        "VECTOR_EMBEDDING_MAX_TOKENS",
        "VECTOR_EMBEDDING_QUERY_INSTRUCTION",
        # Qdrant-specific
        "QDRANT_BASE_URL",
        "QDRANT_API_KEY",
        "QDRANT_COLLECTION_NAME",
        "QDRANT_INDEX_TYPE",
        "QDRANT_INDEX_M",
        "QDRANT_INDEX_EF_CONSTRUCT",
        "QDRANT_DISTANCE",
        "QDRANT_SPARSE_MODEL",
        # Hybrid search pipeline
        "MEMFLOW_HYBRID_SEARCH_ENABLED",
        "MEMFLOW_HYBRID_SPARSE_SEARCH_TOP_K",
        "MEMFLOW_HYBRID_DENSE_SEARCH_TOP_K",
        "MEMFLOW_HYBRID_RRF_TOP_K",
        "MEMFLOW_HYBRID_RRF_WEIGHTS",
        "MEMFLOW_HYBRID_RRF_K",
        # Compatibility fallbacks used by the optional LLM reranker.
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_BASE",
        "LLM_API_KEY",
        # Reranker (order mirrors .env / .env.example)
        "MEMFLOW_RERANKER_PIPELINE",
        "MEMFLOW_RERANKER_FETCH_TOP_K",
        "MEMFLOW_RERANKER_SCOPE",
        "MEMFLOW_RERANKER_CE_SCOPE",
        "MEMFLOW_RERANKER_LLM_SCOPE",
        "MEMFLOW_RERANKER_DOC_CAP",
        "MEMFLOW_RERANKER_TIMEOUT",
        "MEMFLOW_RERANKER_CE_TOP_K",
        "MEMFLOW_RERANKER_LLM_TOP_K",
        "RERANKER_CE_MODEL",
        "RERANKER_CE_API_BASE",
        "RERANKER_CE_API_KEY",
        "RERANKER_CE_QUERY_INSTRUCTION",
        "RERANKER_LLM_PROVIDER",
        "RERANKER_LLM_MODEL",
        "RERANKER_LLM_API_BASE",
        "RERANKER_LLM_API_KEY",
        "RERANKER_LLM_TEMPERATURE",
        "RERANKER_LLM_MAX_PROMPT_CHARS",
    ]
    original = {k: os.environ.get(k) for k in vars_to_clear}
    for k in vars_to_clear:
        os.environ.pop(k, None)

    # These settings are evaluated at import time, before this fixture runs.
    monkeypatch.setattr("memflow.store.EMBEDDING_REQUEST_TIMEOUT", 60.0)
    monkeypatch.setattr("memflow.store.EMBEDDING_MAX_RETRIES", 2)

    # Patch _load_env_file to prevent .env file loading
    with patch("memflow.manager._load_env_file"):
        yield

    for k, v in original.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
