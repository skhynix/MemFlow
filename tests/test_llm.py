# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MemFlow LLM components."""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestParseJson:
    """Tests for JSON parsing utility."""

    @pytest.mark.parametrize("text,expected", [
        ('{"has_procedure": true, "title": "Test"}',
         {"has_procedure": True, "title": "Test"}),
        ('```json\n{"has_procedure": true}\n```',
         {"has_procedure": True}),
        ('"""{"has_procedure": true, "title": "Test"}"""',
         {"has_procedure": True, "title": "Test"}),
        ('Here is the result: {"has_procedure": true, "title": "Test"} end.',
         {"has_procedure": True, "title": "Test"}),
        ('{"content": "line1\\nline2"}',
         {"content": "line1\nline2"}),
    ])
    def test_parse_json_valid(self, text, expected):
        """Test parsing various valid JSON formats."""
        from memflow.llm import parse_json
        result = parse_json(text)
        assert result == expected

    def test_parse_invalid_json_returns_empty(self):
        """Test that invalid JSON returns empty dict."""
        from memflow.llm import parse_json
        text = 'not valid json at all'
        result = parse_json(text)
        assert result == {}


class TestLLMFactory:
    """Tests for LLM factory."""

    @pytest.fixture
    def mock_ollama_module(self):
        """Mock ollama module."""
        mock_client_class = MagicMock()
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_ollama = MagicMock()
        mock_ollama.Client = mock_client_class

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            yield mock_client_class

    @pytest.fixture
    def mock_openai_module(self):
        """Mock openai module."""
        mock_openai_class = MagicMock()
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_openai = MagicMock()
        mock_openai.OpenAI = mock_openai_class

        with patch.dict("sys.modules", {"openai": mock_openai}):
            yield mock_openai_class

    def test_create_ollama(self, mock_ollama_module):
        """Test creating Ollama LLM."""
        from memflow.llm import LLMFactory, OllamaLLM

        llm = LLMFactory.create("ollama")
        assert isinstance(llm, OllamaLLM)
        mock_ollama_module.assert_called_once_with(host="http://localhost:11434")

    def test_create_ollama_with_custom_model(self, mock_ollama_module):
        """Test creating Ollama LLM with custom model."""
        from memflow.llm import LLMFactory, OllamaLLM

        llm = LLMFactory.create("ollama", model="llama3.1")
        assert isinstance(llm, OllamaLLM)
        mock_ollama_module.assert_called_once_with(host="http://localhost:11434")

    def test_create_ollama_with_custom_base(self, mock_ollama_module):
        """Test creating Ollama LLM with custom API base."""
        from memflow.llm import LLMFactory, OllamaLLM

        llm = LLMFactory.create("ollama", api_base="http://custom:11434")
        assert isinstance(llm, OllamaLLM)
        mock_ollama_module.assert_called_once_with(host="http://custom:11434")

    def test_create_openai_compatible_requires_model(self):
        """Test that openai-compatible requires model parameter."""
        from memflow.llm import LLMFactory

        with pytest.raises(ValueError, match="model"):
            LLMFactory.create("openai-compatible")

    def test_create_openai_compatible(self, mock_openai_module):
        """Test creating OpenAI compatible LLM."""
        from memflow.llm import LLMFactory, OpenAICompatibleLLM

        llm = LLMFactory.create("openai-compatible", model="gpt-4")
        assert isinstance(llm, OpenAICompatibleLLM)

    def test_create_openai_compatible_with_custom_base(self, mock_openai_module):
        """Test creating OpenAI compatible LLM with custom base."""
        from memflow.llm import LLMFactory, OpenAICompatibleLLM

        llm = LLMFactory.create(
            "openai-compatible",
            model="gpt-4",
            api_base="http://vllm:8000/v1"
        )
        assert isinstance(llm, OpenAICompatibleLLM)

    def test_create_openai_compatible_with_api_key(self, mock_openai_module):
        """Test creating OpenAI compatible LLM with API key."""
        from memflow.llm import LLMFactory, OpenAICompatibleLLM

        llm = LLMFactory.create(
            "openai-compatible",
            model="gpt-4",
            api_key="test-key"
        )
        assert isinstance(llm, OpenAICompatibleLLM)

    def test_create_unknown_provider(self):
        """Test that unknown provider raises error."""
        from memflow.llm import LLMFactory

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMFactory.create("unknown-provider")


class TestOllamaLLM:
    """Tests for Ollama LLM (mocked)."""

    def test_generate(self):
        """Test generating text with Ollama LLM."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = "Hello, I am an AI assistant."
        mock_client.chat.return_value = mock_response

        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = mock_client

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            from memflow.llm import OllamaLLM

            llm = OllamaLLM(model="llama3.2")
            response = llm.generate([{"role": "user", "content": "Hello"}])

            assert isinstance(response, str)
            assert response == "Hello, I am an AI assistant."
            mock_client.chat.assert_called_once_with(
                model="llama3.2",
                messages=[{"role": "user", "content": "Hello"}]
            )


class TestOpenAICompatibleLLM:
    """Tests for OpenAI Compatible LLM."""

    def test_initialization(self):
        """Test initialization creates OpenAI client with default api_key."""
        mock_openai_class = MagicMock()
        mock_client = MagicMock()
        mock_client.api_key = "EMPTY"
        mock_openai_class.return_value = mock_client

        mock_openai = MagicMock()
        mock_openai.OpenAI = mock_openai_class

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from memflow.llm import OpenAICompatibleLLM

            llm = OpenAICompatibleLLM(model="test-model")

            mock_openai_class.assert_called_once()
            call_kwargs = mock_openai_class.call_args.kwargs
            assert call_kwargs["api_key"] == "EMPTY"
            assert call_kwargs["base_url"] == "http://localhost:8000/v1"

    def test_initialization_with_custom_api_key(self):
        """Test initialization creates OpenAI client with custom api_key."""
        mock_openai_class = MagicMock()
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_openai = MagicMock()
        mock_openai.OpenAI = mock_openai_class

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from memflow.llm import OpenAICompatibleLLM

            llm = OpenAICompatibleLLM(model="test-model", api_key="my-key")

            mock_openai_class.assert_called_once()
            call_kwargs = mock_openai_class.call_args.kwargs
            assert call_kwargs["api_key"] == "my-key"
