# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared Qdrant runtime for skill commands, the Claude hook, and MCP."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from memflow.llm import BaseLLM

if TYPE_CHECKING:
    from memflow.manager import MemFlow


class SkillRetrievalLLM(BaseLLM):
    """Prevent skill operations from invoking a generation LLM."""

    def generate(self, messages: list[dict]) -> str:
        raise RuntimeError("skill retrieval does not support LLM calls")


def create_skill_manager(env_file: str | None = None) -> MemFlow:
    from memflow.manager import MemFlow, QdrantStore, _load_env_file

    if env_file is not None:
        path = Path(env_file).expanduser()
        if not path.is_file():
            raise ValueError(f"environment file is not a file: {path}")
        _load_env_file(str(path))
    else:
        _load_env_file()

    backend = os.getenv("MEMFLOW_BACKEND", "qdrant").strip().lower()
    if backend != "qdrant":
        raise ValueError(
            f"unsupported skill backend {backend!r}; set MEMFLOW_BACKEND to qdrant"
        )

    os.environ["MEMFLOW_BACKEND"] = backend
    return MemFlow(llm=SkillRetrievalLLM(), store=QdrantStore(), use_env=False)
