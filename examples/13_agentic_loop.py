#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
13. Agentic Loop Example — MemFlow Plan/Execute/Learn.

This example demonstrates the full agentic loop with chat() API:
  1. Step-based execution (TOOL type with bash/http/llm)
  2. chat() API with multi-intent classification (SEARCH/ADD/EXECUTE)
  3. Multi-stage planning with Reflect-and-Refine
  4. Automatic learning from successful execution

Run:
  uv run ./examples/13_agentic_loop.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memflow import MemFlowManager
from memflow.models import Step, StepType
from utils import Colors, print_header, print_section, print_labeled_text, print_label, print_step, print_success

# NOTE: This example might not be executed as expected.
#       Better model is recommended.
# LLM and store are loaded from .env file automatically
manager = MemFlowManager()

print_header("Agentic Loop using chat API()")

# ---------------------------------------------------------------------------
# Feature 1: Step-based Data Structure
# ---------------------------------------------------------------------------

print_section("Feature 1: Step-based Execution")

print(f"""
{Colors.CYAN}Design:{Colors.RESET} Step has two types — PLAN (sub-plan recursion) and TOOL (external call)
{Colors.CYAN}Status:{Colors.RESET} pending → done (or failed)
""")

print_labeled_text("Sample Step:", "")

# Create a sample step
step = Step(
    id="step-001",
    goal="List Python files in memflow directory",
    type=StepType.TOOL,
    tool_name="bash",
    args={"command": "ls memflow/*.py 2>/dev/null || echo 'No .py files'"},
)

print(f"  ID:     {step.id}")
print(f"  Goal:   {step.goal}")
print(f"  Type:   {step.type}")
print(f"  Tool:   {step.tool_name}")
print(f"  Status: {step.status}")

# Execute the step
result = manager.execute(type('TaskPlan', (), {'steps': [step]})())
if result:
    r = result[0]
    step.status = "done" if r.success else "failed"
    print(f"\n{Colors.CYAN}Execution Result:{Colors.RESET}")
    print(f"  Status: {step.status}")
    print(f"  Output: {r.output[:60]}..." if r.output else f"  Error: {r.error}")

# ---------------------------------------------------------------------------
# Feature 2: chat() API with Intent Classification
# ---------------------------------------------------------------------------

print_section("Feature 2: chat() API with Intent Classification")

print(f"""
{Colors.CYAN}Design:{Colors.RESET} chat() is the primary interface, classifying intent automatically
{Colors.CYAN}Intents:{Colors.RESET} SEARCH, ADD, EXECUTE, CONVERSATION
""")

# Test CONVERSATION intent
print_label("Test 1: CONVERSATION Intent")
print_labeled_text("User:", "Hello! How are you?")
result = manager.chat("Hello! How are you?")
print(f"  {Colors.CYAN}Intents:{Colors.RESET} {result.get('intents', [result.get('intent', 'N/A')])}")
print(f"  {Colors.CYAN}Primary:{Colors.RESET} {result.get('primary_intent', result.get('intent', 'N/A'))}")
print(f"  {Colors.CYAN}Response:{Colors.RESET} {result['response']}")

# Test ADD intent
print(f"\n")
print_label("Test 2: ADD Intent (storing a procedure)")
procedure_text = """
Remember this procedure:
How to restart a service:
1. Stop the service: sudo systemctl stop myservice
2. Wait 5 seconds: sleep 5
3. Start the service: sudo systemctl start myservice
4. Verify status: sudo systemctl status myservice
"""
print_labeled_text("User:", "")
for line in procedure_text.strip().split('\n'):
    print(f"    {line}")

result = manager.chat(procedure_text, user_id="demo")
print(f"  {Colors.CYAN}Intents:{Colors.RESET} {result.get('intents', [result.get('intent', 'N/A')])}")
print(f"  {Colors.CYAN}Primary:{Colors.RESET} {result.get('primary_intent', result.get('intent', 'N/A'))}")
print(f"  {Colors.CYAN}Response:{Colors.RESET} {result['response']}")

# Test SEARCH intent
print(f"\n")
print_label("Test 3: SEARCH Intent")
print_labeled_text("User:", "How do I restart a service?")
result = manager.chat("How do I restart a service?")
print(f"  {Colors.CYAN}Intents:{Colors.RESET} {result.get('intents', [result.get('intent', 'N/A')])}")
print(f"  {Colors.CYAN}Primary:{Colors.RESET} {result.get('primary_intent', result.get('intent', 'N/A'))}")
print(f"  {Colors.CYAN}Response:{Colors.RESET}")
# Print the actual response text
for line in result['response'].split('\n'):
    print(f"    {line}")

# Test EXECUTE intent without confirmation
print(f"\n")
print_label("Test 4: EXECUTE Intent (with Tool Calling Details)")
print_labeled_text("User:", "Show me the current date")
result = manager.chat("Show me the current date", allow_execute=True)
print(f"  {Colors.CYAN}Intents:{Colors.RESET} {result.get('intents', [result.get('intent', 'N/A')])}")
print(f"  {Colors.CYAN}Primary:{Colors.RESET} {result.get('primary_intent', result.get('intent', 'N/A'))}")
print(f"  {Colors.CYAN}Response:{Colors.RESET}")
print(f"  {Colors.CYAN}Full Response:{Colors.RESET} {result['response'][:200] if result.get('response') else 'N/A'}...")

# Access execute result from handler_results
if 'handler_results' in result and 'EXECUTE' in result['handler_results']:
    exec_handler = result['handler_results']['EXECUTE']
    if 'data' in exec_handler and 'result' in exec_handler['data']:
        exec_result = exec_handler['data']['result']
        success = sum(1 for r in exec_result.step_results if r.success)

        # Show tool calling details
        print(f"\n{Colors.CYAN}Tool Calling Details:{Colors.RESET}")
        for i, (step, r) in enumerate(zip(exec_result.plan.steps, exec_result.step_results), 1):
            status = f"{Colors.GREEN}✓{Colors.RESET}" if r.success else f"{Colors.RED}✗{Colors.RESET}"
            print(f"\n    {Colors.CYAN}Step {i}:{Colors.RESET} {step.goal}")
            print(f"      {Colors.YELLOW}Tool:{Colors.RESET} [{step.type}] {step.tool_name or 'llm'}")
            if step.args:
                print(f"      {Colors.YELLOW}Args:{Colors.RESET}")
                for key, value in step.args.items():
                    print(f"        {key}: {value}")
            print(f"      {Colors.YELLOW}Result:{Colors.RESET} {status}")
            if r.output:
                output_preview = r.output.replace('\n', ' ')[:70]
                print(f"      {Colors.YELLOW}Output:{Colors.RESET} {output_preview}")
            if r.error and not r.success:
                print(f"      {Colors.YELLOW}Error:{Colors.RESET} {r.error}")

        print(f"\n    {Colors.GREEN}Summary:{Colors.RESET} {success}/{len(exec_result.step_results)} steps succeeded")

# ---------------------------------------------------------------------------
# Feature 3: Multi-stage Planning
# ---------------------------------------------------------------------------

print_section("Feature 3: Multi-stage Planning (Reflect-and-Refine)")

print(f"""
{Colors.CYAN}Design:{Colors.RESET} Plan 1 step at a time, execute, then replan based on results
{Colors.CYAN}Benefit:{Colors.RESET} Adapts to failures and unexpected outputs
""")

TASK = "Create a file named 'demo.txt' with 'Hello World', then verify it exists"

print_labeled_text("Task:", TASK)
print(f"\n{Colors.CYAN}Executing with multi-stage planning...{Colors.RESET}\n")

result = manager.run(TASK, user_id="demo", multi_stage=True)

print_success(f"\nExecution Complete!")
print(f"  Total steps: {len(result.plan.steps)}")
print(f"  Success:     {sum(1 for r in result.step_results if r.success)}/{len(result.step_results)}")

if result.learned:
    print_success(f"  Learned: [{result.learned.category}] {result.learned.title}")

# Show step details
print(f"\n{Colors.CYAN}Step Details:{Colors.RESET}")
for i, (step, r) in enumerate(zip(result.plan.steps, result.step_results), 1):
    print_step(i, f"[{step.tool_name or 'llm'}] {step.goal}", r.success, r.output if r.success else None, r.error if not r.success else None)

