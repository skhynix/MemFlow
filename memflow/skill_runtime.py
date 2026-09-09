# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared Qdrant runtime for skill commands, the Claude hook, and MCP."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memflow.llm import BaseLLM

if TYPE_CHECKING:
    from memflow.manager import MemFlow

DEFAULT_CONFIG_PATH = ".memflow/claude-hook.json"


def read_skill_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"skill config must be valid JSON: {path}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"skill config must be a JSON object: {path}")
    return config


def resolve_env_file(
    env_file: str | Path | None = None,
    *,
    project_root: str | Path = ".",
    config: dict[str, Any] | None = None,
) -> Path:
    """Select an explicit file, the saved project file, or the project's .env."""
    project = Path(project_root).expanduser().resolve()
    if env_file is None:
        if config is None:
            path = project / DEFAULT_CONFIG_PATH
            config = read_skill_config(path) if path.exists() else {}
        settings = config.get("memflow", {})
        if not isinstance(settings, dict):
            raise ValueError("skill config memflow must be a JSON object")
        env_file = settings.get("env_file") or ".env"
        if not isinstance(env_file, str):
            raise ValueError("skill config memflow.env_file must be a path string")
    path = Path(env_file).expanduser()
    return (project / path).resolve()


class SkillRetrievalLLM(BaseLLM):
    """Prevent skill operations from invoking a generation LLM."""

    def generate(self, messages: list[dict]) -> str:
        raise RuntimeError("skill retrieval does not support LLM calls")


def create_skill_manager(
    env_file: str | None = None, *, config_path: str | None = None
) -> MemFlow:
    from memflow.manager import MemFlow, QdrantStore, _load_env_file

    config = read_skill_config(Path(config_path).expanduser()) if config_path else None
    path = resolve_env_file(env_file, config=config)
    file_required = (
        env_file is not None
        or config_path is not None
        or Path(DEFAULT_CONFIG_PATH).exists()
    )
    if path.exists() or file_required:
        if not path.is_file():
            raise ValueError(f"environment file is not a file: {path}")
        _load_env_file(str(path))

    backend = os.getenv("MEMFLOW_BACKEND", "qdrant").strip().lower()
    if backend != "qdrant":
        raise ValueError(
            f"unsupported skill backend {backend!r}; set MEMFLOW_BACKEND to qdrant"
        )

    os.environ["MEMFLOW_BACKEND"] = backend
    return MemFlow(llm=SkillRetrievalLLM(), store=QdrantStore(), use_env=False)
