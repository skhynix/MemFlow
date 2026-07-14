# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""MemFlow Benchmark Suite.

This package provides benchmark harnesses for evaluating MemFlow retrieval performance.

Available benchmarks:
    - proced_mem_bench: Procedural Memory Benchmark (Proced_mem_bench) based evaluation
    - wikihow_procedure_silver: WikiHow Procedure Silver v1 benchmark
    - skill_ret_bench: SkillRet benchmark for skill retrieval evaluation
"""

from importlib import import_module

__all__ = ["proced_mem_bench", "wikihow_procedure_silver", "skill_ret_bench"]


def __getattr__(name: str):
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
