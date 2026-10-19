# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Command line tools for MemFlow skill retrieval and Claude Code setup."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from typing import TextIO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memflow",
        description="MemFlow command line tools.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    claude = subparsers.add_parser("claude", help="manage Claude Code integration")
    from memflow.claude_setup import add_claude_subcommands

    add_claude_subcommands(claude)

    skill = subparsers.add_parser("skill", help="manage skills in Qdrant")
    from memflow.skill_cli import add_skill_subcommands

    add_skill_subcommands(skill)
    return parser


def main(
    argv: Iterable[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = stdout or sys.stdout
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help(file=out)
        return 0
    return handler(args, stdout=out, stderr=stderr or sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
