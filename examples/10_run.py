#!/usr/bin/env python3
"""
10. Run — plan, execute, and learn from a task automatically.

run(task) is the full Phase 3 pipeline:
  1. plan()   — retrieve context + LLM decomposition into steps
  2. execute  — run each step through built-in tools (llm / bash)
  3. learn    — extract a reusable SOP from successful steps and store it

Built-in tools used in this example:
  bash — gets today's date so the evening plan is time-aware
  llm  — asks the language model for activity ideas and tips

After run() completes, the learned procedure is retrieved via search() to
demonstrate the Learn → Retrieve back-edge closing the loop.

Run:
  ./examples/10_run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memflow import MemFlowManager
from memflow.llm import LLMFactory

llm = LLMFactory.create("ollama", model="llama3.2")
manager = MemFlowManager(llm=llm)

TASK = "Help me wind down and have a comfortable evening at home tonight"

# ---------------------------------------------------------------------------
# Run the task
# ---------------------------------------------------------------------------

print(f"Task: {TASK}\n")
result = manager.run(TASK)

# ---------------------------------------------------------------------------
# Show the plan
# ---------------------------------------------------------------------------

print(f"{'='*60}")
print(f"  Plan ({len(result.plan.steps)} steps)")
print(f"{'='*60}")
for i, step in enumerate(result.plan.steps, 1):
    print(f"  {i}. [{step.tool_name or 'llm'}] {step.goal}")

# ---------------------------------------------------------------------------
# Show execution results
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("  Execution Results")
print(f"{'='*60}")
success_count = sum(1 for r in result.step_results if r.success)
print(f"  {success_count}/{len(result.step_results)} steps succeeded\n")

step_by_id = {step.id: step for step in result.plan.steps}
for r in result.step_results:
    status = "OK " if r.success else "ERR"
    step = step_by_id.get(r.step_id)
    description = step.goal if step else f"Step {r.step_id[:8]}"
    print(f"  [{status}] {description}")
    if r.success and r.output:
        preview = r.output.replace("\n", " ")[:120]
        print(f"       → {preview}")
    elif not r.success:
        print(f"       ! {r.error[:120]}")

# ---------------------------------------------------------------------------
# Show what was learned
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("  Auto-Learn Result")
print(f"{'='*60}")
if result.learned:
    print(f"  Stored: [{result.learned.category}] {result.learned.title}")
    print()
    for line in result.learned.content.splitlines():
        print(f"  {line}")
else:
    print("  Nothing new was stored (steps not reusable or all failed).")

# ---------------------------------------------------------------------------
# Verify the Learn → Retrieve back-edge
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("  Learn → Retrieve back-edge")
print(f"{'='*60}")
hits = manager.search("how to relax in the evening")
if hits:
    print(f"  search('how to relax in the evening') → {len(hits)} result(s)")
    for h in hits:
        print(f"    score={h.score:.2f}  {h.procedure.title}")
else:
    print("  No match found (procedure may have been classified as non-reusable).")
