#!/usr/bin/env python3
"""
09. Execute — plan a task, then execute step by step (without auto-learn).

This example demonstrates the plan() + execute() pipeline manually,
allowing you to inspect each stage before moving to the full run() pipeline.

Scenarios:
  A. Cold start — no procedures stored, planner works from scratch.
  B. Warm start — a relevant SOP is in the store, planner reuses it.

After planning, each Job is executed manually using the ToolRegistry.
No AutoLearn step is performed — this is for incremental testing.

Run:
  ./examples/09_execute.py
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
    """Display the generated plan."""
    task_plan = manager.plan(TASK)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Task: {task_plan.task}")
    print(f"{'='*60}")
    if task_plan.context:
        print(f"  Context used: yes ({len(task_plan.context)} chars)")
    else:
        print("  Context used: none (cold start)")
    print(f"  Jobs ({len(task_plan.jobs)}):")
    for i, job in enumerate(task_plan.jobs, 1):
        print(f"    {i}. [{job.tool}] {job.description}")
        for k, v in job.args.items():
            preview = str(v)[:70] + "..." if len(str(v)) > 70 else str(v)
            print(f"         {k}: {preview}")
    return task_plan


def execute_plan(task_plan) -> list:
    """Execute each Job in the plan and display results."""
    print(f"\n{'='*60}")
    print("  Executing Jobs")
    print(f"{'='*60}")

    if not task_plan.jobs:
        print("  No jobs to execute (empty plan).")
        return []

    # Use manager.execute() to run all jobs
    job_results = manager.execute(task_plan)

    results = []
    for i, (job, job_result) in enumerate(zip(task_plan.jobs, job_results), 1):
        print(f"\n  [{i}/{len(task_plan.jobs)}] Executing: {job.description}")
        if job_result.success:
            status = "OK"
            output_preview = job_result.output.replace("\n", " ")[:100]
            print(f"       → {status}: {output_preview}")
            results.append({"job": job, "success": True, "output": job_result.output})
        else:
            status = "ERR"
            error_preview = job_result.error[:100]
            print(f"       → {status}: {error_preview}")
            results.append({"job": job, "success": False, "error": job_result.error})
    return results


# ---------------------------------------------------------------------------
# A. Cold start — no relevant procedure in store
# ---------------------------------------------------------------------------

task_plan_cold = show_plan("A. Cold start — no SOP in store")

# Execute the cold start plan
_ = execute_plan(task_plan_cold)

# ---------------------------------------------------------------------------
# B. Warm start — store a relevant SOP, then plan + execute again
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

task_plan_warm = show_plan("B. Warm start — SOP stored, planner has context")

# Execute the warm start plan
_ = execute_plan(task_plan_warm)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("  Summary")
print(f"{'='*60}")
print("  This example demonstrated plan() + execute() without AutoLearn.")
print("  For the full pipeline (plan + execute + learn), see 10_run.py")
