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
            json_str = text[start : end + 1]
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
        timeout: float | None = None,
        temperature: float | None = None,
    ) -> None:
        from ollama import Client

        kwargs = {} if timeout is None else {"timeout": timeout}
        self._client = Client(host=api_base, **kwargs)
        self._model = model
        # None = omit from the request (server default applies). Callers that
        # need deterministic output (e.g. the LLM reranker) pass 0.
        self._temperature = temperature

    def generate(self, messages: list[dict]) -> str:
        kwargs = {}
        if self._temperature is not None:
            kwargs["options"] = {"temperature": self._temperature}
        resp = self._client.chat(model=self._model, messages=messages, **kwargs)
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
        timeout: float | None = None,
        temperature: float | None = None,
    ) -> None:
        from openai import OpenAI

        # timeout omitted keeps the openai client default (600s + retries);
        # callers that need a bounded request (e.g. the reranker under
        # concurrent load) pass one.
        kwargs = {} if timeout is None else {"timeout": timeout}
        self._client = OpenAI(base_url=api_base, api_key=api_key, **kwargs)
        self._model = model
        # None = omit from the request (server default applies — 1.0 for
        # vLLM, i.e. sampling). Callers that need deterministic output
        # (e.g. the LLM reranker) pass 0.
        self._temperature = temperature

    def generate(self, messages: list[dict]) -> str:
        kwargs = {}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs,
        )
        return resp.choices[0].message.content


class LLMFactory:
    @staticmethod
    def create(
        provider: str,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
    ) -> BaseLLM:
        # temperature=None omits it from every request (server default
        # applies); pass 0 for deterministic generation.
        if provider == "ollama":
            return OllamaLLM(
                model=model or "llama3.2",
                api_base=api_base or "http://localhost:11434",
                timeout=timeout,
                temperature=temperature,
            )
        if provider == "openai-compatible":
            if not model:
                raise ValueError("'model' is required for openai-compatible provider")
            return OpenAICompatibleLLM(
                model=model,
                api_base=api_base or "http://localhost:8000/v1",
                api_key=api_key or "EMPTY",
                timeout=timeout,
                temperature=temperature,
            )
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. Choose: ollama, openai-compatible"
        )
