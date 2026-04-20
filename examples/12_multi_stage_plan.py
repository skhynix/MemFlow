#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
12. Multi-Stage Planning — Reflect-and-Refine loop.

This example demonstrates multi-stage planning:
  - Plan 1 step at a time
  - Execute and verify each step
  - Adapt based on results
  - Learn reusable procedure for future

Run:
  ./examples/12_multi_stage_plan.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memflow import MemFlowManager
from memflow.llm import LLMFactory
from utils import Colors, print_header, print_labeled_text, print_success

# NOTE: This example might not be executed as expected.
#       Better model is recommended.
llm = LLMFactory.create("ollama", model="llama3.2")

# 4-step task for multi-stage planning demo
TASK = """
    1. Find all .py files in memflow/ directory
    2. Count lines in each file
    3. Create report.txt with top 3 files by line count
    4. Create backup/ directory and copy report.txt into it
"""

# Clean up
for f in ["report.txt"]:
    try:
        os.remove(f)
    except FileNotFoundError:
        pass
try:
    os.rmdir("backup")
except (FileNotFoundError, OSError):
    pass

print_header("Multi-Stage Planning with Reflect-and-Refine")
print_labeled_text("Task:", TASK)
print_labeled_text("Config:", "max_steps_per_iteration=1 (one step at a time)\n", Colors.YELLOW)

def run(task, manager, max_iterations=8):
    """
    Run task with multi-stage planning, showing each iteration.

    This wrapper demonstrates the Reflect-and-Refine loop by showing
    each planning cycle separately. Uses only public APIs.
    """
    executed_steps = []
    all_results = []
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print_labeled_text(f"  Iteration {iteration}/{max_iterations}:", "", Colors.CYAN)

        # Plan next step(s) using public plan() API with multi_stage=True
        if executed_steps:
            plan = manager.plan(task, multi_stage=True, executed_steps=executed_steps)
        else:
            # Initial planning
            plan = manager.plan(task, multi_stage=True)

        if not plan.steps:
            print_success(f"    Task complete (no more steps)\n")
            break

        for step in plan.steps:
            print(f"    [{step.tool_name or 'llm'}] {step.goal}")

        # Execute using public execute() API
        results = manager.execute(plan)
        executed_steps.extend(plan.steps)
        all_results.extend(results)

        for r in results:
            status = f"{Colors.GREEN}✓{Colors.RESET} Done" if r.success else f"{Colors.RED}✗{Colors.RESET} Failed"
            print(f"    {status}")
            if not r.success and r.error:
                print(f"        {Colors.RED}Error: {r.error[:50]}{Colors.RESET}")

        # Check completion
        if manager._is_task_complete(task, executed_steps):
            print_success(f"    Task verified complete\n")
            break
        print()

    # Learn from execution
    if manager._learner is None:
        from memflow.learner import Learner
        manager._learner = Learner(manager.llm)

    learned = manager._learner.extract(task, executed_steps, user_id="default")
    if learned:
        manager.store.add(learned)

    from memflow.models import TaskPlan, RunResult
    merged = TaskPlan(task=task, steps=executed_steps, context="")
    return RunResult(plan=merged, step_results=all_results, learned=learned)

# ---------------------------------------------------------------------------
# Run: Multi-stage planning with learning
# ---------------------------------------------------------------------------

print(f"{Colors.BOLD}>>> Executing with multi-stage planning{Colors.RESET}\n")

# Configure manager for 1 step per iteration to show each planning cycle
manager = MemFlowManager(
    llm=llm,
    max_steps_per_iteration=1,  # Plan 1 step at a time
    max_plan_iterations=8,       # Max 8 planning iterations
)

result = run(TASK, manager, max_iterations=8)
