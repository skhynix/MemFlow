# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from memflow import MemFlowManager, Procedure

from procedural_memory_benchmark import RetrievalSystem, RetrievedTrajectory
from procedural_memory_benchmark.agentinstruct import AgentInstructCorpusLoader


@dataclass
class AgentTrajectoryMeta:
    """Small normalized view of an AgentInstruct trajectory."""

    task_instance_id: str
    task_description: str
    source: str
    total_steps: int
    steps: list[tuple[int, str, str]]
    full_text: str


def _extract_steps(traj: Any) -> list[tuple[int, str, str]]:
    """Extract (step_id, state, action) tuples from benchmark trajectory objects."""
    parsed: list[tuple[int, str, str]] = []
    # AgentInstruct trajectory format can vary (dataclass-like objects vs dicts),
    # so we normalize both shapes into a stable tuple representation here.
    if hasattr(traj, "state_action_pairs"):
        raw_steps = list(getattr(traj, "state_action_pairs", []) or [])
    else:
        raw_steps = list(getattr(traj, "steps", []) or [])
    for idx, step in enumerate(raw_steps, start=1):
        if isinstance(step, dict):
            step_id = int(step.get("step_id", idx))
            state = str(step.get("state", "")).strip()
            action = str(step.get("action", "")).strip()
        else:
            step_id = int(getattr(step, "step_id", idx))
            state = str(getattr(step, "state", "")).strip()
            action = str(getattr(step, "action", "")).strip()
        parsed.append((step_id, state, action))
    return parsed


def _format_steps_only(steps: list[tuple[int, str, str]]) -> str:
    # IMPORTANT: benchmark requirement says Procedure.content should include
    # only the procedural trace (not repeating the task description).
    # Intentionally exclude state from benchmark content so retrieval focuses
    # on action-only procedural traces rather than environment descriptions.
    if not steps:
        return "Steps:\n"
    lines = ["Steps:"]
    lines.extend(f"{step_id}. Action: {action}" for step_id, _state, action in steps)
    return "\n".join(lines)


def _format_full_text(task_description: str, steps: list[tuple[int, str, str]]) -> str:
    # For retrieval output payloads, we expose a richer document text that
    # includes the natural-language task and procedural steps.
    steps_text = _format_steps_only(steps)
    return f"Task: {task_description}\n{steps_text}"


def _normalize_trajectory(traj: Any) -> AgentTrajectoryMeta:
    steps = _extract_steps(traj)
    task_description = str(getattr(traj, "task_description", "")).strip()
    task_instance_id = str(getattr(traj, "task_instance_id", "")).strip()
    source = str(getattr(traj, "source", "unknown")).strip() or "unknown"
    total_steps = int(getattr(traj, "total_steps", len(steps) if steps else 0))
    return AgentTrajectoryMeta(
        task_instance_id=task_instance_id,
        task_description=task_description,
        source=source,
        total_steps=total_steps,
        steps=steps,
        full_text=_format_full_text(task_description, steps),
    )


def trajectory_to_procedure(traj: Any, user_id: str) -> Procedure:
    """Convert an AgentInstruct trajectory into a MemFlow Procedure."""
    meta = _normalize_trajectory(traj)
    return Procedure(
        # Stable corpus ID is used to allow deterministic de-dup and gold label matching.
        id=meta.task_instance_id,
        # Title carries the natural-language task once.
        title=meta.task_description,
        # Content must carry only procedural trace.
        content=_format_steps_only(meta.steps),
        # user_id must be set before manager.add(procedure=...) because stores
        # persist the Procedure object as-is.
        user_id=user_id,
        category="alfworld",
        tags=["agentinstruct", f"source:{meta.source}", f"steps:{meta.total_steps}"],
    )


def seed_memflow_corpus(
    manager: MemFlowManager,
    user_id: str,
    corpus_path: str | None = None,
    clear_existing: bool = True,
) -> dict[str, Any]:
    """Load AgentInstruct corpus and seed it as direct Procedure objects."""
    print("\n=== Seeding AgentInstruct Corpus ===")

    try:
        loader = AgentInstructCorpusLoader(corpus_path=corpus_path) if corpus_path else AgentInstructCorpusLoader()
    except FileNotFoundError as e:
        raise RuntimeError(f"Corpus path not found: {corpus_path}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to initialize AgentInstructCorpusLoader: {e}") from e

    # procedural_memory_benchmark corpus loader API uses get_all_trajectories().
    # Keep a fallback for compatibility with older snapshots that exposed
    # load_trajectories().
    print("Loading trajectories from corpus...")
    try:
        if hasattr(loader, "get_all_trajectories"):
            trajectories = list(loader.get_all_trajectories())
        else:
            trajectories = list(loader.load_trajectories())
    except json.JSONDecodeError as e:
        raise RuntimeError("Invalid JSON in corpus file") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load trajectories from corpus: {e}") from e

    trajectory_map: dict[str, Any] = {
        str(getattr(tr, "task_instance_id", "")): tr for tr in trajectories
    }
    corpus_ids = {tid for tid in trajectory_map if tid}
    print(f"  Loaded {len(trajectory_map)} trajectories")

    if clear_existing:
        # Safe de-dup path: only remove records that are both:
        # (1) under this benchmark user_id scope and (2) in AgentInstruct corpus IDs.
        # This avoids deleting unrelated procedures from the same store.
        print("Clearing existing procedures...")
        try:
            existing = manager.store.list_all(user_id=user_id)
            deleted_count = 0
            for proc in existing:
                if proc.id in corpus_ids:
                    manager.store.delete(proc.id)
                    deleted_count += 1
            print(f"  Deleted {deleted_count} existing procedures")
        except Exception as e:
            raise RuntimeError(f"Failed to clear existing procedures: {e}") from e

    print("Seeding procedures...")
    for i, (tid, traj) in enumerate(trajectory_map.items(), 1):
        if not tid:
            continue
        proc = trajectory_to_procedure(traj, user_id=user_id)
        manager.add(procedure=proc)
        if i % 10 == 0 or i == len(trajectory_map):
            print(f"\r  Progress: {i}/{len(trajectory_map)} procedures", end="", flush=True)

    print()  # newline after loop
    print(f"Corpus seeding complete: {len(trajectory_map)} procedures\n")
    return trajectory_map


class MemFlowRetrievalAdapter(RetrievalSystem):
    """RetrievalSystem adapter that uses MemFlowManager.search()."""

    def __init__(
        self,
        manager: MemFlowManager,
        user_id: str,
        trajectory_map: dict[str, Any],
        backend: str,
        llm_provider: str,
        llm_model: str,
    ) -> None:
        self.manager = manager
        self.user_id = user_id
        self.trajectory_map = trajectory_map
        self.backend = backend
        self.llm_provider = llm_provider
        self.llm_model = llm_model

    def get_system_name(self) -> str:
        return "memflow_manager_search"

    def get_system_info(self) -> dict[str, Any]:
        return {
            "method": "memflow_manager.search",
            "backend": self.backend,
            "user_id": self.user_id,
            "corpus_size": len(self.trajectory_map),
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
        }

    def _to_retrieved_trajectory(self, traj_id: str, score: float) -> RetrievedTrajectory:
        traj = self.trajectory_map.get(traj_id)
        if traj is None:
            # Log warning for missing trajectory ID
            import warnings
            warnings.warn(f"Retrieved trajectory ID '{traj_id}' not found in trajectory map", RuntimeWarning, stacklevel=2)
            # Keep output shape consistent even if a store returns an ID that
            # is missing from the seeded trajectory map.
            return RetrievedTrajectory(
                trajectory_id=traj_id,
                task_instance_id=traj_id,
                task_description="",
                similarity_score=float(score),
                total_steps=0,
                document_text="",
            )

        meta = _normalize_trajectory(traj)
        return RetrievedTrajectory(
            trajectory_id=meta.task_instance_id,
            task_instance_id=meta.task_instance_id,
            task_description=meta.task_description,
            similarity_score=float(score),
            total_steps=meta.total_steps,
            document_text=meta.full_text,
        )

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedTrajectory]:
        # Required benchmark path: use MemFlowManager.search directly.
        search_results = self.manager.search(query, user_id=self.user_id, top_k=k)
        return [
            self._to_retrieved_trajectory(
                result.procedure.id,
                float(result.score) if result.score is not None else 0.0,
            )
            for result in search_results
        ]
