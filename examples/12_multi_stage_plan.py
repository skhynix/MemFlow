#!/usr/bin/env python3
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
from memflow.planner import LLMPlanner
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
    all_plans = []
    all_results = []
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print_labeled_text(f"  Iteration {iteration}/{max_iterations}:", "", Colors.CYAN)

        # Plan next step(s) using public plan() API with multi_stage=True
        if all_results:
            # Replanning with execution history - convert StepResult to JobResult
            from memflow.models import Job, JobResult
            executed_results = []
            for i, sr in enumerate(all_results):
                # Find the step from the last plan
                if i < len(all_plans[-1].steps):
                    step = all_plans[-1].steps[i]
                    job = Job(id=sr.step_id, tool=step.tool_name or "llm", description=step.goal, args=step.args)
                    executed_results.append(JobResult(job=job, success=sr.success, output=sr.output, error=sr.error))
            plan = manager.plan(task, multi_stage=True, executed_results=executed_results)
        else:
            # Initial planning
            plan = manager.plan(task, multi_stage=True)

        if not plan.steps:
            print_success(f"    Task complete (no more steps)\n")
            break

        all_plans.append(plan)
        for step in plan.steps:
            print(f"    [{step.tool_name or 'llm'}] {step.goal}")

        # Execute using public execute() API
        results = manager.execute(plan)
        all_results.extend(results)

        for r in results:
            status = f"{Colors.GREEN}✓{Colors.RESET} Done" if r.success else f"{Colors.RED}✗{Colors.RESET} Failed"
            print(f"    {status}")
            if not r.success and r.error:
                print(f"        {Colors.RED}Error: {r.error[:50]}{Colors.RESET}")

        # Check completion
        if manager._is_task_complete(task, all_results):
            print_success(f"    Task verified complete\n")
            break
        print()

    # Learn from execution
    if manager._learner is None:
        from memflow.learner import Learner
        manager._learner = Learner(manager.llm)
    # Convert StepResult to JobResult for learner
    from memflow.models import Job, JobResult
    job_results = []
    for i, sr in enumerate(all_results):
        step = plan.steps[i] if i < len(plan.steps) else None
        if step:
            job = Job(id=sr.step_id, tool=step.tool_name or "llm", description=step.goal, args=step.args)
            job_results.append(JobResult(job=job, success=sr.success, output=sr.output, error=sr.error))

    learned = manager._learner.extract(task, job_results, user_id="default")
    if learned:
        manager.store.add(learned)

    from memflow.models import TaskPlan, RunResult
    merged = TaskPlan(task=task, steps=[s for p in all_plans for s in p.steps], context="")
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
