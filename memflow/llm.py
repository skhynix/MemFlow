# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LLM abstraction for MemFlow.

Supported providers:
  - ollama:            Local Ollama server (default)
  - openai-compatible: Any OpenAI-compatible API endpoint (vLLM, LM Studio, etc.)

Example — vLLM:
  LLMFactory.create("openai-compatible", model="meta-llama/Llama-3.2-3B",
                    api_base="http://localhost:8000/v1")
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod


def parse_json(text: str) -> dict:
    """Extract a JSON object from LLM output, stripping markdown fences and triple quotes."""
    # Remove markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Replace triple quotes with single quotes
    text = re.sub(r'"""\s*', '"', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object by matching braces
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            json_str = text[start:end + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    return {}


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, messages: list[dict]) -> str: ...


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model: str = "llama3.2",
        api_base: str = "http://localhost:11434",
    ) -> None:
        from ollama import Client
        self._client = Client(host=api_base)
        self._model = model

    def generate(self, messages: list[dict]) -> str:
        resp = self._client.chat(model=self._model, messages=messages)
        return resp.message.content


class OpenAICompatibleLLM(BaseLLM):
    """
    OpenAI-compatible LLM client for self-hosted servers such as vLLM.

    Usage:
      llm = OpenAICompatibleLLM(
          model="meta-llama/Llama-3.2-3B-Instruct",
          api_base="http://localhost:8000/v1",
      )

    The api_key defaults to "EMPTY", which is the convention for local vLLM servers
    that do not enforce authentication.
    """

    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
    ) -> None:
        from openai import OpenAI
        self._client = OpenAI(base_url=api_base, api_key=api_key)
        self._model = model

    def generate(self, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return resp.choices[0].message.content


class LLMFactory:
    @staticmethod
    def create(
        provider: str,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> BaseLLM:
        if provider == "ollama":
            return OllamaLLM(
                model=model or "llama3.2",
                api_base=api_base or "http://localhost:11434",
            )
        if provider == "openai-compatible":
            if not model:
                raise ValueError("'model' is required for openai-compatible provider")
            return OpenAICompatibleLLM(
                model=model,
                api_base=api_base or "http://localhost:8000/v1",
                api_key=api_key or "EMPTY",
            )
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. Choose: ollama, openai-compatible"
        )
