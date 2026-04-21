# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for unit tests."""

import os
from unittest.mock import MagicMock, patch

import pytest


class FakeLLM:
    """Fake LLM for deterministic unit tests."""

    def __init__(self, response: str = '{"has_procedure": true, "title": "Test", "content": "1. Step one"}'):
        self.response = response
        self.generate_calls = []
        self._custom_response = None

    def generate(self, messages: list[dict]) -> str:
        self.generate_calls.append(messages)
        if self._custom_response is not None:
            response = self._custom_response
            self._custom_response = None  # Clear after one use
            return response
        return self.response

    def set_response(self, response: str):
        """Set a custom response for the next generate call only."""
        self._custom_response = response


@pytest.fixture
def fake_llm():
    """Provide a FakeLLM instance."""
    return FakeLLM()


@pytest.fixture
def clean_env():
    """Clear relevant environment variables before test and prevent .env file loading.

    This fixture:
    1. Saves original environment variables
    2. Clears relevant env vars
    3. Patches _load_env_file to prevent .env file loading during tests
    4. Restores original environment variables after test
    """
    vars_to_clear = [
        'LLM_PROVIDER', 'LLM_MODEL', 'LLM_API_BASE', 'LLM_API_KEY',
        'MEMFLOW_BACKEND', 'MEMFLOW_DATA_DIR',
        'MEMMACHINE_BASE_URL', 'MEMMACHINE_ORG_ID', 'MEMMACHINE_PROJECT', 'MEMMACHINE_API_KEY',
        'MEMFLOW_BASE_URL', 'MEMFLOW_EMBEDDING_MODEL', 'MEMFLOW_EMBEDDING_API_BASE',
        'MEMFLOW_EMBEDDING_API_KEY', 'MEMFLOW_EMBEDDING_DIMENSIONS',
    ]
    original = {k: os.environ.get(k) for k in vars_to_clear}
    for k in vars_to_clear:
        os.environ.pop(k, None)

    # Patch _load_env_file to prevent .env file loading
    with patch("memflow.manager._load_env_file"):
        yield

    for k, v in original.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


@pytest.fixture
def memmachine_mock():
    """Mock memmachine_client module for MemMachineStore and MemMachineBypass tests."""
    mock_memory = MagicMock()
    mock_project = MagicMock()
    mock_project.memory.return_value = mock_memory
    mock_client = MagicMock()
    mock_client.get_or_create_project.return_value = mock_project
    mock_module = MagicMock()
    mock_module.MemMachineClient.return_value = mock_client

    yield mock_client, mock_memory, mock_module


@pytest.fixture
def mock_http_response():
    """Provide a mock HTTP response factory."""
    def _factory(content: bytes = b'{"status": "ok"}'):
        mock_response = MagicMock()
        mock_response.read.return_value = content
        return mock_response
    return _factory


@pytest.fixture
def mock_urlopen_context(mock_http_response):
    """Mock urllib.request.urlopen for HTTP tests."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_http_response()
        yield mock_urlopen
