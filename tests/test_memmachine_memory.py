# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Explicit MemMachine reads stay lazy, scoped, and separate from skills."""

import importlib
import sys
from types import MappingProxyType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from memflow.memmachine_memory import MemMachineMemoryClient


@pytest.fixture
def sdk(monkeypatch):
    client = MagicMock()
    project = client.get_project.return_value
    memories = [MagicMock(), MagicMock()]
    project.memory.side_effect = memories
    responses = [
        SimpleNamespace(
            content={"episodic_memory": ["episode-a"], "semantic_memory": ["feature-a"]}
        ),
        SimpleNamespace(
            content={"episodic_memory": ["episode-b"], "semantic_memory": ["feature-b"]}
        ),
    ]
    for memory, response in zip(memories, responses):
        memory.search.return_value = response
    factory = MagicMock(return_value=client)
    monkeypatch.setitem(
        sys.modules, "memmachine_client", SimpleNamespace(MemMachineClient=factory)
    )
    return factory, client, project, memories, responses


def test_import_construction_and_env_are_lazy_without_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "memmachine_client", None)
    module = importlib.reload(sys.modules["memflow.memmachine_memory"])
    client = module.MemMachineMemoryClient()
    env_client = module.MemMachineMemoryClient.from_env()
    assert "memmachine_client" not in repr(client)
    with pytest.raises(ImportError, match=r"memflow\[memmachine\]"):
        env_client.search("query", user_id="user", set_metadata={"user": "user"})
    client.close()
    client.close()
    with pytest.raises(RuntimeError, match="closed"):
        client.search("query", user_id="user", set_metadata={"user": "user"})


def test_search_preserves_responses_and_separates_user_scopes(sdk):
    factory, client, project, memories, responses = sdk
    reader = MemMachineMemoryClient(
        base_url="http://memory", org_id="org", project_id="project", api_key="secret"
    )
    assert "secret" not in repr(reader)
    factory.assert_not_called()
    for index, user in enumerate([" alice ", "bob"]):
        metadata = MappingProxyType(
            {"owner": user, "nested": {"active": True, "values": [None, 3, 1.5]}}
        )
        assert (
            reader.search(
                " query ", user_id=user, set_metadata=metadata, limit=7, timeout=12
            )
            is responses[index]
        )
        memories[index].search.assert_called_once_with(
            query=" query ", limit=7, timeout=12, set_metadata=dict(metadata)
        )
        assert project.memory.call_args_list[index].kwargs == {
            "metadata": {"user_id": user}
        }
    factory.assert_called_once_with(base_url="http://memory", api_key="secret")
    client.get_project.assert_called_once_with(org_id="org", project_id="project")
    assert [call[0] for call in client.method_calls] == ["get_project"]
    assert [call[0] for call in project.method_calls] == ["memory", "memory"]
    for memory in memories:
        assert [call[0] for call in memory.method_calls] == ["search"]
    reader.close()
    reader.close()
    client.close.assert_called_once_with()
    with pytest.raises(RuntimeError, match="closed"):
        reader.search("query", user_id="user", set_metadata={"owner": "user"})


def test_direct_settings_and_explicit_env_defaults(monkeypatch, sdk, tmp_path):
    factory, client, *_ = sdk
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("MEMMACHINE_BASE_URL=http://must-not-load\n")
    for key in [
        "MEMMACHINE_BASE_URL",
        "MEMMACHINE_ORG_ID",
        "MEMMACHINE_PROJECT",
        "MEMMACHINE_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)
    reader = MemMachineMemoryClient.from_env()
    factory.assert_not_called()
    reader.search("q", user_id="u", set_metadata={"owner": "u"})
    factory.assert_called_once_with(base_url="http://localhost:8080")
    client.get_project.assert_called_once_with(org_id="default", project_id="memflow")
    reader.close()
    factory.reset_mock()
    client.reset_mock()
    monkeypatch.setenv("MEMMACHINE_BASE_URL", "http://env")
    monkeypatch.setenv("MEMMACHINE_ORG_ID", "env-org")
    monkeypatch.setenv("MEMMACHINE_PROJECT", "env-project")
    monkeypatch.setenv("MEMMACHINE_API_KEY", "env-secret")
    reader = MemMachineMemoryClient.from_env()
    assert (reader._base_url, reader._org_id, reader._project_id, reader._api_key) == (
        "http://env",
        "env-org",
        "env-project",
        "env-secret",
    )
    explicit = MemMachineMemoryClient(
        base_url="http://explicit", org_id="explicit-org", project_id="explicit-project"
    )
    explicit.search("q", user_id="u", set_metadata={"owner": "u"})
    factory.assert_called_once_with(base_url="http://explicit")
    client.get_project.assert_called_once_with(
        org_id="explicit-org", project_id="explicit-project"
    )
    explicit.close()


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("query", None, TypeError),
        ("query", 1, TypeError),
        ("query", " \n", ValueError),
        ("user_id", None, TypeError),
        ("user_id", 1, TypeError),
        ("user_id", "  ", ValueError),
        ("set_metadata", None, TypeError),
        ("set_metadata", [], TypeError),
        ("set_metadata", {}, ValueError),
        ("set_metadata", {1: "user"}, TypeError),
        ("set_metadata", {"a": {1: "user"}}, TypeError),
        ("set_metadata", {"a": object()}, TypeError),
        ("set_metadata", {"a": (1, 2)}, TypeError),
        ("set_metadata", {"a": float("nan")}, ValueError),
        ("set_metadata", {"a": float("inf")}, ValueError),
        ("limit", True, TypeError),
        ("limit", 1.5, TypeError),
        ("limit", "1", TypeError),
        ("limit", 0, ValueError),
        ("limit", -1, ValueError),
    ],
)
def test_invalid_scope_fails_before_sdk_initialization(sdk, field, value, error):
    factory, *_ = sdk
    kwargs = {"query": "q", "user_id": "u", "set_metadata": {"owner": "u"}, "limit": 10}
    kwargs[field] = value
    with pytest.raises(error):
        MemMachineMemoryClient().search(**kwargs)
    factory.assert_not_called()


def test_cycle_in_metadata_fails_without_sdk(sdk):
    factory, *_ = sdk
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="acyclic"):
        MemMachineMemoryClient().search("q", user_id="u", set_metadata=cyclic)
    factory.assert_not_called()


def test_initialization_failure_closes_client_and_can_retry(sdk):
    factory, client, project, memories, responses = sdk
    failure = RuntimeError("project unavailable")
    client.get_project.side_effect = [failure, project]
    reader = MemMachineMemoryClient()
    with pytest.raises(RuntimeError) as exc:
        reader.search("q", user_id="u", set_metadata={"owner": "u"})
    assert exc.value is failure
    client.close.assert_called_once_with()
    assert reader._client is reader._project is None
    assert reader.search("q", user_id="u", set_metadata={"owner": "u"}) is responses[0]
    assert factory.call_count == 2


def test_connection_and_search_errors_propagate(sdk):
    factory, client, project, memories, responses = sdk
    failure = RuntimeError("connection failed")
    factory.side_effect = failure
    reader = MemMachineMemoryClient()
    with pytest.raises(RuntimeError) as exc:
        reader.search("q", user_id="u", set_metadata={"owner": "u"})
    assert exc.value is failure
    factory.side_effect = None
    failure = RuntimeError("search failed")
    memories[0].search.side_effect = failure
    with pytest.raises(RuntimeError) as exc:
        reader.search("q", user_id="u", set_metadata={"owner": "u"})
    assert exc.value is failure
    reader.close()
