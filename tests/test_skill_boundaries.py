# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Public skill boundaries must reject input before any external work."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import memflow.manager as manager_module
from memflow import MemFlow, QdrantStore, SkillRecord
from tests.fakes import FakeSkillStore


@pytest.fixture
def bare_store():
    store = QdrantStore.__new__(QdrantStore)
    store._sparse_enabled = False
    store._client = MagicMock()
    store._compute_emb = MagicMock(return_value=[1.0])
    store._compute_emb_async = AsyncMock(return_value=[1.0])
    store._compute_embs_batch = MagicMock(return_value=[[1.0]])
    store._compute_embs_batch_async = AsyncMock(return_value=[[1.0]])
    store._upsert_point = MagicMock()
    return store


def invoke(target, method, *args, **kwargs):
    value = getattr(target, method)(*args, **kwargs)
    return asyncio.run(value) if method.endswith("_async") else value


@pytest.mark.parametrize("target_name", ["manager", "store"])
@pytest.mark.parametrize("method", ["add", "add_async"])
@pytest.mark.parametrize("invalid", ["procedure", "episodic", None, 42])
@pytest.mark.parametrize("batch", [False, True])
def test_add_rejects_before_sanitation_or_io(
    target_name, method, invalid, batch, bare_store, monkeypatch
):
    bad = (
        SkillRecord(title="bad", content="body", kind=invalid)
        if isinstance(invalid, str)
        else invalid
    )
    good = SkillRecord(title="good", content="nul\0body")
    value = [good, bad] if batch else bad
    sanitize = MagicMock(side_effect=AssertionError("sanitation started"))
    monkeypatch.setattr(bare_store, "_sanitize_content", sanitize)
    target = (
        MemFlow(store=bare_store, use_env=False)
        if target_name == "manager"
        else bare_store
    )
    with pytest.raises(ValueError if isinstance(invalid, str) else TypeError):
        invoke(target, method, value)
    sanitize.assert_not_called()
    bare_store._upsert_point.assert_not_called()
    assert not bare_store._client.mock_calls
    for name in [
        "_compute_emb",
        "_compute_emb_async",
        "_compute_embs_batch",
        "_compute_embs_batch_async",
    ]:
        getattr(bare_store, name).assert_not_called()
    assert good.content == "nul\0body"


@pytest.mark.parametrize("method", ["add", "add_async"])
def test_empty_add_is_noop(method, bare_store):
    assert invoke(bare_store, method, []) == 0
    fake = MagicMock()
    assert invoke(MemFlow(store=fake, use_env=False), method, []) == {
        "num_seeded": 0,
        "num_skipped": 0,
        "total": 0,
    }
    assert not fake.mock_calls
    assert not bare_store._client.mock_calls
    bare_store._upsert_point.assert_not_called()


@pytest.mark.parametrize("method", ["add", "add_async"])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("raw_metadata", [False, True])
@pytest.mark.parametrize("nul", [False, True])
def test_snapshot_sanitation_preserves_original(
    method, batch, raw_metadata, nul, bare_store
):
    body = "---\nname: original\n---\n\n  Keep spaces.  \n" + (
        "nul\0\n" if nul else "\n"
    )
    metadata = (
        {
            "skill": {"sha256": "original-source-hash"},
            "index": {"search_text_sha256": "original-index"},
        }
        if raw_metadata
        else {}
    )
    record = SkillRecord(
        title="original",
        content=body,
        metadata=metadata,
        source_path="/canonical/SKILL.md" if raw_metadata else None,
    )
    original = copy.deepcopy(record)
    assert invoke(bare_store, method, [record] if batch else record) == 1
    stored = bare_store._upsert_point.call_args.args[0]
    assert stored.content == body.replace("\0", "")
    assert stored.metadata == metadata
    assert record == original
    if nul:
        assert stored is not record
    from memflow.models import skill_search_text

    expected = skill_search_text(stored)
    embedding = getattr(bare_store, "_compute_embs_batch" if batch else "_compute_emb")
    if method.endswith("_async"):
        embedding = getattr(
            bare_store, "_compute_embs_batch_async" if batch else "_compute_emb_async"
        )
    assert embedding.call_args.args[0] == ([expected] if batch else expected)


@pytest.mark.parametrize(
    "target_name,method",
    [
        ("manager", "search"),
        ("manager", "search_async"),
        ("store", "search"),
        ("store", "search_async"),
        ("store", "search_hybrid"),
    ],
)
@pytest.mark.parametrize("kind", [None, "procedure", "semantic", ""])
@pytest.mark.parametrize("query", ["query", ["query", "another"]])
def test_search_rejects_before_routing(target_name, method, kind, query, bare_store):
    target = (
        MemFlow(store=bare_store, use_env=False)
        if target_name == "manager"
        else bare_store
    )
    with pytest.raises(ValueError, match="skill"):
        invoke(target, method, query, kind=kind)
    assert not bare_store._client.mock_calls
    bare_store._compute_emb.assert_not_called()
    bare_store._compute_emb_async.assert_not_called()
    bare_store._compute_embs_batch.assert_not_called()
    bare_store._compute_embs_batch_async.assert_not_called()


def test_sync_non_skill_rejected_before_source_access(monkeypatch):
    store = FakeSkillStore()
    record = SkillRecord(
        id="old",
        title="old",
        content="unchanged",
        kind="procedure",
        source_path="/should/not/be/read",
    )
    store.records[record.id] = record
    original = copy.deepcopy(record)
    manager = MemFlow(store=store, use_env=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("non-skill sync touched its source or metadata")

    monkeypatch.setattr(manager_module, "Path", forbidden)
    monkeypatch.setattr(manager_module, "_mark_stale_skill", forbidden)
    monkeypatch.setattr(store, "add", forbidden)
    with pytest.raises(ValueError, match="skill"):
        manager.sync_skill(record.id)
    assert record == original


@pytest.mark.parametrize("payload", [{}, {"kind": "procedure"}, {"kind": "episodic"}])
def test_decode_keeps_existing_record_kind(payload, bare_store):
    point = SimpleNamespace(id="point-id", payload={"id": "original-id", **payload})
    record = bare_store._point_to_skill(point)
    assert record.id == "original-id"
    assert record.kind == payload.get("kind", "skill")


def test_explicit_falsey_store_wins_over_environment(monkeypatch, tmp_path):
    class FalseyStore(FakeSkillStore):
        def __len__(self):
            return 0

    store = FalseyStore()
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("MEMFLOW_HYBRID_SEARCH_ENABLED=on\n")
    monkeypatch.delenv("MEMFLOW_HYBRID_SEARCH_ENABLED", raising=False)
    monkeypatch.setenv("MEMFLOW_BACKEND", "unsupported")
    qdrant = MagicMock(side_effect=AssertionError("external store initialized"))
    monkeypatch.setattr(manager_module, "QdrantStore", qdrant)
    manager = MemFlow(store=store)
    assert manager.store is store
    assert manager._hybrid_search_enabled is True
    assert MemFlow(store=store, use_env=False).store is store
    qdrant.assert_not_called()


@pytest.mark.parametrize(
    "backend", ["emulated", "file", "memmachine", "other", "", " "]
)
def test_invalid_backend_rejected_before_external_initialization(
    backend, clean_env, monkeypatch
):
    monkeypatch.setenv("MEMFLOW_BACKEND", backend)
    qdrant = MagicMock()
    monkeypatch.setattr(manager_module, "QdrantStore", qdrant)
    with pytest.raises(ValueError, match="unsupported backend"):
        MemFlow()
    qdrant.assert_not_called()


@pytest.mark.parametrize("backend", [None, "qdrant", " QDRANT "])
def test_environment_constructor_uses_qdrant_defaults(backend, clean_env, monkeypatch):
    if backend is not None:
        monkeypatch.setenv("MEMFLOW_BACKEND", backend)
    qdrant = MagicMock()
    monkeypatch.setattr(manager_module, "QdrantStore", qdrant)
    assert MemFlow().store is qdrant.return_value
    qdrant.assert_called_once_with()


def test_missing_store_and_retired_arguments_fail_without_connections(monkeypatch):
    qdrant = MagicMock()
    monkeypatch.setattr(manager_module, "QdrantStore", qdrant)
    with pytest.raises(ValueError, match="explicit store"):
        MemFlow(use_env=False)
    with pytest.raises(TypeError):
        MemFlow(object())
    manager = MemFlow(store=FakeSkillStore(), use_env=False)
    with pytest.raises(TypeError):
        manager.add(messages="extract this")
    with pytest.raises(TypeError):
        manager.add(procedure=SkillRecord(title="old keyword", content="body"))
    with pytest.raises(TypeError):
        asyncio.run(manager.add_async([], "old-user-id"))
    qdrant.assert_not_called()
