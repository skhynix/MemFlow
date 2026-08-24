# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""User-facing commands for managing indexed MemFlow skills."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from memflow.llm import BaseLLM

if TYPE_CHECKING:
    from memflow.manager import MemFlow
    from memflow.models import Procedure

TRUST_STATES = ("trusted", "unknown", "blocked")
SUPPORTED_BACKEND = "qdrant"


class _SkillManagementLLM(BaseLLM):
    """LLM placeholder for commands that only use store-backed skill APIs."""

    def generate(self, messages: list[dict]) -> str:
        del messages
        raise RuntimeError("skill management commands do not support LLM calls")


def _stable_json(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _create_skill_manager(env_file: str | None = None) -> MemFlow:
    from memflow.manager import MemFlow, QdrantStore, _load_env_file

    if env_file is not None:
        path = Path(env_file).expanduser()
        if not path.is_file():
            raise ValueError(f"environment file is not a file: {path}")
        _load_env_file(str(path))
    else:
        _load_env_file()

    backend = os.getenv("MEMFLOW_BACKEND", "emulated").strip().lower()
    if backend == "emulated":
        raise ValueError(
            "unsupported skill CLI backend 'emulated': state is process-local; "
            "set MEMFLOW_BACKEND to qdrant"
        )
    if backend == "file":
        raise ValueError(
            "unsupported skill CLI backend 'file': FileStore does not preserve "
            "complete skill metadata and source paths; set MEMFLOW_BACKEND to qdrant"
        )
    if backend == "memmachine":
        raise ValueError(
            "unsupported skill CLI backend 'memmachine': skill records are "
            "procedural memory, not episodic or semantic memory; set "
            "MEMFLOW_BACKEND to qdrant"
        )
    if backend != SUPPORTED_BACKEND:
        raise ValueError(
            f"unsupported skill CLI backend {backend!r}; set MEMFLOW_BACKEND to qdrant"
        )

    os.environ["MEMFLOW_BACKEND"] = backend
    return MemFlow(
        llm=_SkillManagementLLM(),
        store=QdrantStore(),
        use_env=False,
    )


def _skill_summary(procedure: Procedure) -> dict[str, object]:
    skill = procedure.metadata.get("skill", {})
    if not isinstance(skill, dict):
        skill = {}
    governance = procedure.metadata.get("governance", {})
    if not isinstance(governance, dict):
        governance = {}

    return {
        "id": procedure.id,
        "name": str(skill.get("name") or procedure.title),
        "description": str(skill.get("description") or ""),
        "user_id": procedure.user_id,
        "source_path": procedure.source_path,
        "sha256": skill.get("sha256"),
        "trust_state": governance.get("trust_state"),
        "mode": governance.get("mode"),
        "stale": skill.get("stale") is True,
    }


def _run_operation(
    args: argparse.Namespace,
    operation: Callable[[MemFlow], object],
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    try:
        manager = _create_skill_manager(args.env_file)
        result = operation(manager)
    except Exception as exc:
        print(f"error: {exc}", file=stderr)
        return 1

    stdout.write(_stable_json(result))
    return 0


def _run_add(
    args: argparse.Namespace,
    *,
    stdout: TextIO,
    stderr: TextIO,
    **_: Any,
) -> int:
    return _run_operation(
        args,
        lambda manager: manager.add_skill(
            args.path,
            user_id=args.user_id,
            source=args.source,
            trust_state=args.trust_state,
        ),
        stdout=stdout,
        stderr=stderr,
    )


def _run_sync(
    args: argparse.Namespace,
    *,
    stdout: TextIO,
    stderr: TextIO,
    **_: Any,
) -> int:
    return _run_operation(
        args,
        lambda manager: manager.sync_skill(args.path_or_id),
        stdout=stdout,
        stderr=stderr,
    )


def _run_list(
    args: argparse.Namespace,
    *,
    stdout: TextIO,
    stderr: TextIO,
    **_: Any,
) -> int:
    def list_skills(manager: MemFlow) -> dict[str, object]:
        summaries = [
            _skill_summary(procedure)
            for procedure in manager.list_skills(
                user_id=args.user_id,
                trust_state=args.trust_state,
            )
        ]
        summaries.sort(key=lambda skill: (str(skill["name"]), str(skill["id"])))
        return {"count": len(summaries), "skills": summaries}

    return _run_operation(
        args,
        list_skills,
        stdout=stdout,
        stderr=stderr,
    )


def _add_env_file_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env-file",
        metavar="PATH",
        help="load Qdrant configuration from this environment file",
    )


def add_skill_subcommands(parser: argparse.ArgumentParser) -> None:
    """Register skill management subcommands on ``parser``."""
    subparsers = parser.add_subparsers(dest="skill_command", metavar="command")

    add = subparsers.add_parser("add", help="register a skill")
    add.add_argument("path", metavar="PATH", help="skill directory or SKILL.md path")
    add.add_argument("--user-id", default="default", help="skill owner user ID")
    add.add_argument("--source", default="local", help="skill source label")
    add.add_argument("--trust-state", choices=TRUST_STATES)
    _add_env_file_argument(add)
    add.set_defaults(handler=_run_add)

    sync = subparsers.add_parser("sync", help="refresh a registered skill")
    sync.add_argument(
        "path_or_id",
        metavar="PATH_OR_ID",
        help="skill directory, SKILL.md path, or existing skill ID",
    )
    _add_env_file_argument(sync)
    sync.set_defaults(handler=_run_sync)

    list_parser = subparsers.add_parser("list", help="list registered skills")
    list_parser.add_argument("--user-id", default="default", help="skill owner user ID")
    list_parser.add_argument("--trust-state", choices=TRUST_STATES)
    _add_env_file_argument(list_parser)
    list_parser.set_defaults(handler=_run_list)
