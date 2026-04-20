#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
08. Plan — decompose a task into steps using stored procedures as context.

plan(task) is the first half of the Phase 3 pipeline:
  1. Retrieve relevant procedures from the store (context for the planner).
  2. Call the LLM with PLANNING_PROMPT to decompose the task into steps.
  3. Return a TaskPlan — no execution yet.

This example shows two scenarios side by side:
  A. Cold start — no procedures stored, planner works from scratch.
  B. Warm start — a relevant SOP is in the store, planner reuses it.

Comparing the two plans makes the Retrieve → Planner back-edge visible.

Run:
  uv run ./examples/08_plan.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memflow import MemFlowManager, Procedure
from memflow.llm import LLMFactory

llm = LLMFactory.create("ollama", model="llama3.2")
manager = MemFlowManager(llm=llm)

TASK = "Plan a relaxing day off at home"


def show_plan(label: str) -> None:
    task_plan = manager.plan(TASK)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Task: {task_plan.task}")
    print(f"{'='*60}")
    if task_plan.context:
        print(f"  Context used: yes ({len(task_plan.context)} chars)")
    else:
        print("  Context used: none (cold start)")
    print(f"  Steps ({len(task_plan.steps)}):")
    for i, step in enumerate(task_plan.steps, 1):
        print(f"    {i}. [{step.tool_name or 'llm'}] {step.goal}")
        for k, v in step.args.items():
            preview = str(v)[:70] + "..." if len(str(v)) > 70 else str(v)
            print(f"         {k}: {preview}")


# ---------------------------------------------------------------------------
# A. Cold start — no relevant procedure in store
# ---------------------------------------------------------------------------

show_plan("A. Cold start — no SOP in store")

# ---------------------------------------------------------------------------
# B. Warm start — store a relevant SOP, then plan again
# ---------------------------------------------------------------------------

manager.add(procedure=Procedure(
    title="How to Have a Perfect Rest Day",
    content=(
        "1. Start with a light breakfast — fruit, yogurt, or toast.\n"
        "2. Pick one enjoyable activity: reading, a movie, or a gentle walk.\n"
        "3. Prepare your favorite snack or warm drink in the afternoon.\n"
        "4. Take a short nap if you feel tired — 20 to 30 minutes is enough.\n"
        "5. End with a relaxing bath or shower before bed."
    ),
    category="lifestyle",
))

show_plan("B. Warm start — SOP stored, planner has context")
