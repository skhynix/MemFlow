#!/usr/bin/env python3
"""
10. Coding Agent — write scripts, run them, and learn the workflow.

A minimal coding agent that:
  1. Plans tasks using two domain-specific tools (write_file, run_script).
  2. Executes each step autonomously via the bash tool.
  3. Learns the coding workflow as a reusable SOP after each successful run.
  4. Reuses that SOP to plan the next task more effectively.

Custom tools registered for this agent:
  write_file(filename, content)  — create a Python script in the work directory
  run_script(filename)           — execute the script with python3

Run 1  Cold start — no SOP stored yet.
       Task: write and run a script that counts from 1 to 5.
       Agent figures out the steps from scratch, then learns the pattern.

Run 2  Warm start — SOP from Run 1 is in the store.
       Task: write and run a script that prints a simple multiplication table.
       The planner reuses the stored workflow — plan is noticeably more direct.

Run:
  ./examples/10_coding_agent.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memflow import MemFlowManager, LLMPlanner
from memflow.llm import LLMFactory

# ---------------------------------------------------------------------------
# Work directory — all scripts are created here
# ---------------------------------------------------------------------------

WORK_DIR = Path(tempfile.mkdtemp(prefix="memflow_agent_"))

llm = LLMFactory.create("ollama", model="llama3.2")
manager = MemFlowManager(llm=llm)

# ---------------------------------------------------------------------------
# Custom tools
# ---------------------------------------------------------------------------

def write_file(filename: str, content: str) -> str:
    """Create a Python source file in the work directory."""
    path = WORK_DIR / filename
    path.write_text(content, encoding="utf-8")
    return f"Created {path} ({len(content)} chars)"


def run_script(filename: str) -> str:
    """Run a Python script that was written with write_file."""
    path = WORK_DIR / filename
    result = subprocess.run(
        ["python3", str(path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    output = (result.stdout + result.stderr).strip()
    return output or "(no output)"


CUSTOM_TOOLS = {"write_file": write_file, "run_script": run_script}

# Register the custom tools with the planner so the LLM knows to use them.
manager._planner = LLMPlanner(
    llm,
    extra_tools=[
        {
            "name": "write_file",
            "description": (
                "Create a Python source file in the working directory. "
                'Args: {"filename": "script.py", "content": "print(\'hello\')"}'
            ),
        },
        {
            "name": "run_script",
            "description": (
                "Execute a Python script that was created with write_file. "
                'Args: {"filename": "script.py"}'
            ),
        },
    ],
)

# ---------------------------------------------------------------------------
# Helper: display a run result
# ---------------------------------------------------------------------------

def show_run(label: str, task: str, result) -> None:
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"  Task: {task}")
    print(f"{'='*62}")

    if result.plan.context:
        print(f"  Stored SOP used: yes ({len(result.plan.context)} chars)\n")
    else:
        print("  Stored SOP used: no (cold start)\n")

    print(f"  Plan ({len(result.plan.jobs)} steps):")
    for i, job in enumerate(result.plan.jobs, 1):
        print(f"    {i}. [{job.tool}] {job.description}")

    ok = sum(1 for r in result.job_results if r.success)
    print(f"\n  Execution ({ok}/{len(result.job_results)} succeeded):")
    for r in result.job_results:
        mark = "OK " if r.success else "ERR"
        print(f"    [{mark}] {r.job.description}")
        if r.success and r.output:
            for line in r.output.splitlines()[:4]:
                print(f"           {line}")
        elif not r.success:
            print(f"           ! {r.error[:80]}")

    if result.learned:
        print(f"\n  Learned → stored: [{result.learned.category}] {result.learned.title}")
    else:
        print("\n  Learned → nothing stored")


# ---------------------------------------------------------------------------
# Run 1 — cold start
# ---------------------------------------------------------------------------

result1 = manager.run(
    "Write a Python script called count.py that prints numbers 1 to 5, "
    "one per line, then run it.",
    tools=CUSTOM_TOOLS,
)
show_run("Run 1 — cold start", result1.plan.task, result1)

# ---------------------------------------------------------------------------
# Run 2 — warm start (SOP from Run 1 is now in the store)
# ---------------------------------------------------------------------------

result2 = manager.run(
    "Write a Python script called times_table.py that prints the "
    "multiplication table for 3 (3×1 through 3×10), then run it.",
    tools=CUSTOM_TOOLS,
)
show_run("Run 2 — warm start", result2.plan.task, result2)

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

print(f"\n{'='*62}")
print("  Summary")
print(f"{'='*62}")
procs = manager.store.list_all()
print(f"  Procedures stored: {len(procs)}")
for p in procs:
    print(f"    [{p.category}] {p.title}")

scripts = sorted(WORK_DIR.glob("*.py"))
print(f"\n  Scripts created in {WORK_DIR}:")
for s in scripts:
    lines = s.read_text().splitlines()
    print(f"    {s.name} ({len(lines)} lines)")
