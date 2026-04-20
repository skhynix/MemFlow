#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
03. Explicit Extraction — extract procedure from conversation via LLM.

Shows how MemFlow extracts procedural knowledge from a conversation
and stores it as a Procedure.

  Path 1: Explicit Store — pass a ready-made Procedure object.
  Path 2: Explicit Extraction — pass conversation text, LLM extracts.

Run:
  uv run ./examples/03_explicit_extraction.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memflow import MemFlowManager, Procedure
from memflow.llm import LLMFactory

llm = LLMFactory.create("ollama", model="llama3.2")
manager = MemFlowManager(llm=llm)

print("=== 1. Explicit Store ===")
print("Pass a ready-made Procedure object directly.\n")

proc = Procedure(
    title="How to Make Coffee",
    content="1. Boil water. 2. Put coffee in a cup. 3. Pour hot water. 4. Stir and drink.",
    user_id="alice",
    category="cooking",
)
manager.add(procedure=proc, user_id="alice")
print(f"  [Stored] {proc.title}\n")

print("=== 2. Explicit Extraction ===")
print("Pass conversation text, LLM extracts procedure.\n")

conversation = """
User: How do I water a plant?

Assistant: Here is the step by step guide:
Step 1: Check the soil.
Step 2: Pour water slowly.
Step 3: Stop when water comes out.
Step 4: Water weekly.
"""

result = manager.add(messages=conversation, user_id="alice")
if result.get("results"):
    for r in result["results"]:
        print(f"  [Extracted] {r['title']}")
elif result.get("skipped"):
    print(f"  Skipped: {result['skipped']}")
else:
    print("  LLM found no procedure in the conversation")

print()
print(f"Total stored: {len(manager.store.list_all(user_id='alice'))}")
