# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the user-facing skill management CLI."""

from __future__ import annotations

import io
import json
import os

import pytest

import memflow.cli as cli_module
import memflow.manager as manager_module
import memflow.skill_cli as skill_cli
from memflow.cli import build_parser, main
from memflow.models import Procedure


class FakeSkillManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.add_result: dict = {"event": "ADD_SKILL", "id": "skill:add"}
        self.sync_result: dict = {
            "event": "NOOP_SYNC_SKILL",
            "id": "skill:sync",
            "status": "noop",
        }
        self.list_result: list[Procedure] = []

    def add_skill(self, *args, **kwargs):
        self.calls.append(("add_skill", args, kwargs))
        return self.add_result

    def sync_skill(self, *args, **kwargs):
        self.calls.append(("sync_skill", args, kwargs))
        return self.sync_result

    def list_skills(self, *args, **kwargs):
        self.calls.append(("list_skills", args, kwargs))
        return self.list_result


def _invoke(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    rc = main(argv, stdout=stdout, stderr=stderr)
    return rc, stdout.getvalue(), stderr.getvalue()


def _patch_manager(monkeypatch, manager: FakeSkillManager) -> None:
    monkeypatch.setattr(skill_cli, "_create_skill_manager", lambda _env=None: manager)


def _skill(
    *,
    name: str,
    skill_id: str,
    content: str = "private skill body",
    user_id: str = "alice",
    trust_state: str = "trusted",
    stale: bool = False,
) -> Procedure:
    return Procedure(
        id=skill_id,
        title=name,
        content=content,
        user_id=user_id,
        source_path=f"/skills/{name}/SKILL.md",
        metadata={
            "skill": {
                "name": name,
                "description": f"Description for {name}",
                "sha256": f"sha-{skill_id}",
                "stale": stale,
            },
            "governance": {
                "trust_state": trust_state,
                "mode": "instruction" if trust_state == "trusted" else "data",
            },
        },
    )


def test_top_level_help_lists_skill_command():
    help_text = build_parser().format_help()

    assert "skill" in help_text
    assert "manage skills in Qdrant" in help_text


def test_skill_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["skill", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "add       register a skill" in help_text
    assert "sync      refresh a registered skill" in help_text
    assert "list      list registered skills" in help_text


@pytest.mark.parametrize(
    ("subcommand", "expected"),
    [
        ("add", "--trust-state"),
        ("sync", "PATH_OR_ID"),
        ("list", "--user-id"),
    ],
)
def test_skill_subcommand_help(subcommand, expected, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["skill", subcommand, "--help"])

    assert exc_info.value.code == 0
    assert expected in capsys.readouterr().out


def test_skill_routing_does_not_enter_legacy_chat_parser(monkeypatch):
    manager = FakeSkillManager()
    _patch_manager(monkeypatch, manager)

    def fail_legacy_parser():
        raise AssertionError("legacy chat parser was used")

    monkeypatch.setattr(cli_module, "_build_chat_parser", fail_legacy_parser)

    rc, stdout, stderr = _invoke(["skill", "list"])

    assert rc == 0
    assert json.loads(stdout) == {"count": 0, "skills": []}
    assert stderr == ""


def test_add_forwards_all_arguments_and_prints_stable_json(monkeypatch):
    manager = FakeSkillManager()
    manager.add_result = {"z": 2, "event": "ADD_SKILL", "a": 1}
    _patch_manager(monkeypatch, manager)

    rc, stdout, stderr = _invoke(
        [
            "skill",
            "add",
            "/skills/example/SKILL.md",
            "--user-id",
            "alice",
            "--source",
            "catalog",
            "--trust-state",
            "trusted",
        ]
    )

    assert rc == 0
    assert stderr == ""
    assert manager.calls == [
        (
            "add_skill",
            ("/skills/example/SKILL.md",),
            {
                "user_id": "alice",
                "source": "catalog",
                "trust_state": "trusted",
            },
        )
    ]
    assert stdout == ('{\n  "a": 1,\n  "event": "ADD_SKILL",\n  "z": 2\n}\n')


def test_add_forwards_omitted_trust_state_as_none(monkeypatch):
    manager = FakeSkillManager()
    _patch_manager(monkeypatch, manager)

    rc, _stdout, stderr = _invoke(["skill", "add", "/skills/example"])

    assert rc == 0
    assert stderr == ""
    assert manager.calls[0] == (
        "add_skill",
        ("/skills/example",),
        {"user_id": "default", "source": "local", "trust_state": None},
    )


@pytest.mark.parametrize("target", ["/skills/example", "skill:abc123"])
def test_sync_accepts_path_or_id_and_preserves_result(monkeypatch, target):
    manager = FakeSkillManager()
    manager.sync_result = {
        "event": "STALE_SYNC_SKILL",
        "id": "skill:abc123",
        "metadata_updated": True,
        "status": "stale",
    }
    _patch_manager(monkeypatch, manager)

    rc, stdout, stderr = _invoke(["skill", "sync", target])

    assert rc == 0
    assert stderr == ""
    assert manager.calls == [("sync_skill", (target,), {})]
    assert json.loads(stdout) == manager.sync_result


def test_list_filters_sorts_and_omits_content(monkeypatch):
    manager = FakeSkillManager()
    manager.list_result = [
        _skill(name="zeta", skill_id="skill:3", content="ZETA_SECRET"),
        _skill(name="alpha", skill_id="skill:2", content="ALPHA_SECRET"),
        _skill(
            name="alpha",
            skill_id="skill:1",
            content="OTHER_SECRET",
            stale=True,
        ),
    ]
    _patch_manager(monkeypatch, manager)

    rc, stdout, stderr = _invoke(
        [
            "skill",
            "list",
            "--user-id",
            "alice",
            "--trust-state",
            "trusted",
        ]
    )

    payload = json.loads(stdout)
    assert rc == 0
    assert stderr == ""
    assert manager.calls == [
        (
            "list_skills",
            (),
            {"user_id": "alice", "trust_state": "trusted"},
        )
    ]
    assert payload["count"] == 3
    assert [skill["id"] for skill in payload["skills"]] == [
        "skill:1",
        "skill:2",
        "skill:3",
    ]
    assert payload["skills"][0] == {
        "id": "skill:1",
        "name": "alpha",
        "description": "Description for alpha",
        "user_id": "alice",
        "source_path": "/skills/alpha/SKILL.md",
        "sha256": "sha-skill:1",
        "trust_state": "trusted",
        "mode": "instruction",
        "stale": True,
    }
    assert "content" not in stdout
    assert "SECRET" not in stdout


def test_explicit_env_file_loads_before_manager_and_preserves_environment(
    tmp_path, monkeypatch
):
    env_file = tmp_path / "backend.env"
    env_file.write_text(
        "MEMFLOW_BACKEND=memmachine\nQDRANT_BASE_URL=http://from-explicit-file\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMFLOW_BACKEND", "QDRANT")
    monkeypatch.delenv("QDRANT_BASE_URL", raising=False)
    observed = {}

    class DummyQdrantStore:
        def __init__(self):
            observed["qdrant_url"] = os.environ.get("QDRANT_BASE_URL")

    class DummyMemFlow:
        def __init__(self, *, llm, store, use_env):
            observed.update(
                {
                    "backend": os.environ.get("MEMFLOW_BACKEND"),
                    "llm": llm,
                    "store": store,
                    "use_env": use_env,
                }
            )

    monkeypatch.setattr(manager_module, "QdrantStore", DummyQdrantStore)
    monkeypatch.setattr(manager_module, "MemFlow", DummyMemFlow)

    result = skill_cli._create_skill_manager(str(env_file))

    assert isinstance(result, DummyMemFlow)
    assert observed["backend"] == "qdrant"
    assert observed["qdrant_url"] == "http://from-explicit-file"
    assert isinstance(observed["llm"], skill_cli._SkillManagementLLM)
    assert isinstance(observed["store"], DummyQdrantStore)
    assert observed["use_env"] is False


def test_explicit_env_file_does_not_merge_cwd_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / "backend.env"
    env_file.write_text("MEMFLOW_BACKEND=qdrant\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "QDRANT_BASE_URL=http://from-cwd-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MEMFLOW_BACKEND", raising=False)
    monkeypatch.delenv("QDRANT_BASE_URL", raising=False)
    observed = {}

    class DummyQdrantStore:
        def __init__(self):
            observed["qdrant_url"] = os.getenv(
                "QDRANT_BASE_URL", "http://localhost:6333"
            )

    class DummyMemFlow:
        def __init__(self, *, llm, store, use_env):
            observed["store"] = store
            observed["use_env"] = use_env

    monkeypatch.setattr(manager_module, "QdrantStore", DummyQdrantStore)
    monkeypatch.setattr(manager_module, "MemFlow", DummyMemFlow)

    result = skill_cli._create_skill_manager(str(env_file))

    assert isinstance(result, DummyMemFlow)
    assert observed["qdrant_url"] == "http://localhost:6333"
    assert isinstance(observed["store"], DummyQdrantStore)
    assert observed["use_env"] is False


def test_manager_initializes_without_optional_llm_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMFLOW_BACKEND", "QDRANT")

    class DummyQdrantStore:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fail_provider(*_args, **_kwargs):
        raise AssertionError("optional LLM provider was initialized")

    monkeypatch.setattr(manager_module, "QdrantStore", DummyQdrantStore)
    monkeypatch.setattr(manager_module.LLMFactory, "create", fail_provider)

    manager = skill_cli._create_skill_manager()

    assert isinstance(manager.llm, skill_cli._SkillManagementLLM)
    assert isinstance(manager.store, DummyQdrantStore)
    assert os.environ["MEMFLOW_BACKEND"] == "qdrant"


@pytest.mark.parametrize(
    "configured_backend",
    [
        "qdrant",
        "QDRANT",
        " qdrant ",
    ],
)
def test_qdrant_backend_is_normalized_before_manager(
    configured_backend, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMFLOW_BACKEND", configured_backend)
    observed = {}

    class DummyQdrantStore:
        pass

    class DummyMemFlow:
        def __init__(self, *, llm, store, use_env):
            observed["llm"] = llm
            observed["store"] = store
            observed["use_env"] = use_env
            observed["backend"] = os.environ["MEMFLOW_BACKEND"]

    monkeypatch.setattr(manager_module, "QdrantStore", DummyQdrantStore)
    monkeypatch.setattr(manager_module, "MemFlow", DummyMemFlow)

    manager = skill_cli._create_skill_manager()

    assert isinstance(manager, DummyMemFlow)
    assert isinstance(observed["llm"], skill_cli._SkillManagementLLM)
    assert isinstance(observed["store"], DummyQdrantStore)
    assert observed["use_env"] is False
    assert observed["backend"] == "qdrant"


@pytest.mark.parametrize(
    ("backend", "reason"),
    [
        ("emulated", "state is process-local"),
        ("file", "does not preserve complete skill metadata and source paths"),
        (
            "memmachine",
            "skill records are procedural memory, not episodic or semantic memory",
        ),
        ("pgvector", None),
        ("unsupported", None),
    ],
)
def test_skill_commands_reject_unsupported_backends(
    backend, reason, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMFLOW_BACKEND", backend)

    def fail_manager(*_args, **_kwargs):
        raise AssertionError("manager construction should not be attempted")

    monkeypatch.setattr(manager_module, "MemFlow", fail_manager)

    rc, stdout, stderr = _invoke(["skill", "list"])

    assert rc == 1
    assert stdout == ""
    assert f"unsupported skill CLI backend '{backend}'" in stderr
    if reason is not None:
        assert reason in stderr
    assert "set MEMFLOW_BACKEND to qdrant" in stderr


def test_skill_commands_reject_default_emulated_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MEMFLOW_BACKEND", raising=False)

    rc, stdout, stderr = _invoke(["skill", "list"])

    assert rc == 1
    assert stdout == ""
    assert "unsupported skill CLI backend 'emulated'" in stderr


def test_missing_explicit_env_file_returns_error_without_stdout(tmp_path):
    missing = tmp_path / "missing.env"

    rc, stdout, stderr = _invoke(["skill", "list", "--env-file", str(missing)])

    assert rc == 1
    assert stdout == ""
    assert stderr == f"error: environment file is not a file: {missing}\n"


def test_operation_error_returns_nonzero_and_uses_stderr(monkeypatch):
    manager = FakeSkillManager()

    def fail_add(*_args, **_kwargs):
        raise RuntimeError("backend unavailable")

    manager.add_skill = fail_add
    _patch_manager(monkeypatch, manager)

    rc, stdout, stderr = _invoke(["skill", "add", "/skills/example"])

    assert rc == 1
    assert stdout == ""
    assert stderr == "error: backend unavailable\n"
