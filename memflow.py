#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Run the experimental MemFlow interactive console.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_cli_main():
    root = Path(__file__).resolve().parent
    package_dir = root / "memflow"
    package = types.ModuleType("memflow")
    package.__path__ = [str(package_dir)]
    sys.modules["memflow"] = package

    spec = importlib.util.spec_from_file_location(
        "memflow.cli",
        package_dir / "cli.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the memflow CLI")

    module = importlib.util.module_from_spec(spec)
    sys.modules["memflow.cli"] = module
    spec.loader.exec_module(module)
    return module.main


def main() -> int:
    cli_main = _load_cli_main()
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
