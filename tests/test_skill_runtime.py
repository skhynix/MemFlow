# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Configuration must agree across every skill entry point."""

import json
import os

import pytest

import memflow.manager as manager_module
from memflow.claude_hook import default_manager_factory
from memflow.mcp_server import _create_manager
from memflow.skill_runtime import create_skill_manager


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for name in list(os.environ):
        if name.startswith(("MEMFLOW_", "QDRANT_", "VECTOR_EMBEDDING_", "LLM_")):
            monkeypatch.delenv(name)

    class FakeQdrantStore:
        def __init__(self):
            self.url = os.getenv("QDRANT_BASE_URL")
            self.collection = os.getenv("QDRANT_COLLECTION_NAME")

    def fail_llm(*_args, **_kwargs):
        raise AssertionError("skill operations must not initialize a generation LLM")

    monkeypatch.setattr(manager_module, "QdrantStore", FakeQdrantStore)
    monkeypatch.setattr(manager_module.LLMFactory, "create", fail_llm)


@pytest.fixture(params=["cli", "hook", "mcp"])
def factory(request, runtime):
    return {
        "cli": create_skill_manager,
        "hook": lambda path: default_manager_factory({"memflow": {"env_file": path}}),
        "mcp": _create_manager,
    }[request.param]


def test_defaults_to_qdrant_without_generation_llm(factory, tmp_path):
    (tmp_path / ".env").write_text("QDRANT_BASE_URL=http://default-file\n")

    manager = factory(None)

    assert manager.store.url == "http://default-file"
    with pytest.raises(RuntimeError, match="does not support LLM calls"):
        manager.llm.generate([])


def test_explicit_file_is_not_merged_with_cwd_dotenv(factory, tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("QDRANT_COLLECTION_NAME=wrong-collection\n")
    selected = tmp_path / "skills.env"
    selected.write_text("QDRANT_BASE_URL=http://selected-file\n")
    monkeypatch.setenv("QDRANT_BASE_URL", "http://shell-override")

    manager = factory(str(selected))

    assert manager.store.url == "http://shell-override"
    assert manager.store.collection is None


def test_missing_explicit_file_fails_before_store_initialization(factory, tmp_path):
    with pytest.raises(ValueError, match="environment file is not a file"):
        factory(str(tmp_path / "missing.env"))


def test_obsolete_backend_is_rejected(factory, monkeypatch):
    monkeypatch.setenv("MEMFLOW_BACKEND", "emulated")
    with pytest.raises(ValueError, match="unsupported skill backend"):
        factory(None)


def test_saved_environment_file_is_reused(factory, tmp_path):
    selected = tmp_path / "saved.env"
    selected.write_text("QDRANT_BASE_URL=http://saved-file\n")
    (tmp_path / ".env").write_text("QDRANT_COLLECTION_NAME=wrong-collection\n")
    config_path = tmp_path / ".memflow" / "claude-hook.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps({"memflow": {"env_file": str(selected)}}))

    manager = factory(None)

    assert manager.store.url == "http://saved-file"
    assert manager.store.collection is None


def test_explicit_env_file_overrides_saved_config(factory, tmp_path):
    selected = tmp_path / "explicit.env"
    selected.write_text("QDRANT_BASE_URL=http://explicit-file\n")
    config_path = tmp_path / ".memflow" / "claude-hook.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps({"memflow": {"env_file": "missing.env"}}))

    assert factory(str(selected)).store.url == "http://explicit-file"


def test_mcp_config_works_from_another_directory(runtime, tmp_path):
    selected = tmp_path / "saved.env"
    selected.write_text("QDRANT_BASE_URL=http://saved-file\n")
    config_path = tmp_path / "separate-project" / "hook.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps({"memflow": {"env_file": str(selected)}}))
    (tmp_path / ".env").write_text("QDRANT_COLLECTION_NAME=wrong-collection\n")

    manager = create_skill_manager(config_path=str(config_path))

    assert manager.store.url == "http://saved-file"
    assert manager.store.collection is None
