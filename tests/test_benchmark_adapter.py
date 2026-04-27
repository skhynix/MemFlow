"""Unit tests for the procedural benchmark adapter."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def adapter_module(monkeypatch: pytest.MonkeyPatch):
    """Load the adapter module with minimal stubs for external dependencies."""

    memflow_mod = ModuleType("memflow")

    @dataclass
    class Procedure:
        title: str
        content: str
        id: str = ""
        user_id: str = "default"
        category: str = "general"
        tags: list[str] = field(default_factory=list)

    class MemFlowManager:
        pass

    memflow_mod.Procedure = Procedure
    memflow_mod.MemFlowManager = MemFlowManager

    benchmark_mod = ModuleType("procedural_memory_benchmark")

    class RetrievalSystem:
        pass

    @dataclass
    class RetrievedTrajectory:
        trajectory_id: str
        task_instance_id: str
        task_description: str
        similarity_score: float
        total_steps: int
        document_text: str

    benchmark_mod.RetrievalSystem = RetrievalSystem
    benchmark_mod.RetrievedTrajectory = RetrievedTrajectory

    agentinstruct_mod = ModuleType("procedural_memory_benchmark.agentinstruct")

    class AgentInstructCorpusLoader:
        pass

    agentinstruct_mod.AgentInstructCorpusLoader = AgentInstructCorpusLoader

    monkeypatch.setitem(sys.modules, "memflow", memflow_mod)
    monkeypatch.setitem(sys.modules, "procedural_memory_benchmark", benchmark_mod)
    monkeypatch.setitem(sys.modules, "procedural_memory_benchmark.agentinstruct", agentinstruct_mod)

    adapter_path = Path(__file__).resolve().parents[1] / "benchmark" / "proced_mem_bench" / "adapter.py"
    module_name = "tests._proced_mem_bench_adapter"
    spec = importlib.util.spec_from_file_location(module_name, adapter_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_trajectory_to_procedure_formats_steps_attribute(adapter_module) -> None:
    traj = SimpleNamespace(
        task_instance_id="alfworld_0",
        task_description="find two laptop and put them in bed.",
        source="agentinstruct",
        total_steps=2,
        steps=[
            {
                "step_id": 1,
                "state": "You are in the middle of a room.",
                "action": "go to diningtable 1",
            },
            {
                "step_id": 2,
                "state": "On the diningtable 1, you see a laptop 1.",
                "action": "take laptop 1 from diningtable 1",
            },
        ],
    )

    procedure = adapter_module.trajectory_to_procedure(traj, user_id="benchmark")

    assert procedure.title == "find two laptop and put them in bed."
    assert procedure.content == (
        "Steps:\n"
        "1. Action: go to diningtable 1\n"
        "2. Action: take laptop 1 from diningtable 1"
    )


def test_trajectory_to_procedure_formats_state_action_pairs_attribute(adapter_module) -> None:
    traj = SimpleNamespace(
        task_instance_id="alfworld_0",
        task_description="find two laptop and put them in bed.",
        source="agentinstruct",
        total_steps=2,
        state_action_pairs=[
            {
                "step_id": 1,
                "state": "You are in the middle of a room.",
                "action": "go to diningtable 1",
            },
            {
                "step_id": 2,
                "state": "On the diningtable 1, you see a laptop 1.",
                "action": "take laptop 1 from diningtable 1",
            },
        ],
    )

    procedure = adapter_module.trajectory_to_procedure(traj, user_id="benchmark")

    assert procedure.title == "find two laptop and put them in bed."
    assert procedure.content == (
        "Steps:\n"
        "1. Action: go to diningtable 1\n"
        "2. Action: take laptop 1 from diningtable 1"
    )
