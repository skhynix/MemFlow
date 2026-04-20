#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
14. Multi-Intent Chat Demo — Handle multiple intents in a single message.

This example demonstrates the multi-intent classification feature:
  - A single message can contain multiple intents (SEARCH, ADD, EXECUTE, CONVERSATION)
  - Intents are processed in order
  - Each intent handler is called sequentially

Run:
  uv run ./examples/14_multi_intent.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memflow import MemFlowManager
from memflow.llm import LLMFactory
from utils import Colors, print_header, print_labeled_text

# NOTE: This example might not be executed as expected.
#       Better model is recommended.
llm = LLMFactory.create("ollama", model="llama3.2")
manager = MemFlowManager(llm=llm)

print_header("Multi-Intent Chat Demo")

# ---------------------------------------------------------------------------
# Test 1: Single Intent (SEARCH)
# ---------------------------------------------------------------------------

print_header("Test 1: Single Intent (SEARCH)")

message = "How do I restart a service in ubuntu?"
print_labeled_text(f"\nUser:", message)

result = manager.chat(message, user_id="demo")
print(f"\n{Colors.YELLOW}Intents:{Colors.RESET} {result.get('intents', [result.get('intent', 'N/A')])}")
print(f"{Colors.YELLOW}Primary:{Colors.RESET} {result.get('primary_intent', result.get('intent', 'N/A'))}")
print(f"{Colors.CYAN}Response:{Colors.RESET}")
for i, line in enumerate(result['response'].split('\n')):
    if i < 10:
        print(f"    {line}")
    else:
        print("    ...")
        break

# ---------------------------------------------------------------------------
# Test 2: Single Intent (ADD)
# ---------------------------------------------------------------------------

print_header("Test 2: Single Intent (ADD)")

procedure = """
Remember this procedure:
How to check disk usage:
1. Run: df -h to see overall disk usage
2. Run: du -sh * to see directory sizes
3. Run: du -ah | sort -rh | head -20 to find largest files
"""
print_labeled_text(f"\nUser:", "")
for line in procedure.strip().split('\n'):
    print(f"    {line}")

result = manager.chat(procedure, user_id="demo")
print(f"\n{Colors.YELLOW}Intents:{Colors.RESET} {result.get('intents', [result.get('intent', 'N/A')])}")
print(f"{Colors.YELLOW}Primary:{Colors.RESET} {result.get('primary_intent', result.get('intent', 'N/A'))}")
print(f"{Colors.CYAN}Response:{Colors.RESET}")
for line in result['response'].split('\n'):
    print(f"    {line}")


# ---------------------------------------------------------------------------
# Test 3: Multi-Intent (SEARCH + EXECUTE with confirmation)
# ---------------------------------------------------------------------------

print_header("Test 3: Multi-Intent (SEARCH + EXECUTE, requires confirmation)")

message = "Find the disk usage procedure and run it for me"
print_labeled_text(f"\nUser:", message)

# Without allow_execute, should ask for confirmation
result = manager.chat(message, user_id="demo", allow_execute=False)
print(f"\n{Colors.YELLOW}Intents:{Colors.RESET} {result.get('intents', [result.get('intent', 'N/A')])}")
print(f"{Colors.YELLOW}Requires Confirmation:{Colors.RESET} {result.get('requires_confirmation', False)}")
print(f"{Colors.CYAN}Response:{Colors.RESET}")
for line in result['response'].split('\n'):
    print(f"    {line}")

# Now with allow_execute=True
print(f"\n{Colors.GREEN}User confirms: proceed with execution{Colors.RESET}")
result = manager.chat(message, user_id="demo", allow_execute=True)
print(f"\n{Colors.YELLOW}Intents:{Colors.RESET} {result.get('intents', [result.get('intent', 'N/A')])}")
print(f"{Colors.YELLOW}Primary:{Colors.RESET} {result.get('primary_intent', result.get('intent', 'N/A'))}")

# Show execution details
if 'handler_results' in result and 'EXECUTE' in result['handler_results']:
    exec_result = result['handler_results']['EXECUTE']
    if 'data' in exec_result and 'result' in exec_result['data']:
        run_result = exec_result['data']['result']
        print(f"\n{Colors.CYAN}Execution Details:{Colors.RESET}")
        for i, (step, r) in enumerate(zip(run_result.plan.steps, run_result.step_results), 1):
            status = f"{Colors.GREEN}✓{Colors.RESET}" if r.success else f"{Colors.RED}✗{Colors.RESET}"
            print(f"  {i}. {status} [{step.tool_name or 'llm'}] {step.goal}")
            if r.output:
                output_preview = r.output.replace('\n', ' ')[:60]
                print(f"       Output: {output_preview}")

print(f"{Colors.CYAN}Response:{Colors.RESET}")
for line in result['response'].split('\n'):
    print(f"    {line}")
