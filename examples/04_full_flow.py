#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
04. Full Flow — explicit add, LLM extraction, search, and chat combined.

Brings everything together:
  1. Explicit add: store two procedures directly.
  2. LLM extraction: extract a procedure from a conversation.
  3. Search: find the right procedure with direct and indirect queries.
  4. Chat: answer questions grounded in stored procedures.

Run:
  ./examples/04_full_flow.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memflow import MemFlowManager, Procedure
from memflow.llm import LLMFactory

llm = LLMFactory.create("ollama", model="llama3.2")
manager = MemFlowManager(llm=llm)

# ---------------------------------------------------------------------------
# 1. Explicit add
# ---------------------------------------------------------------------------

print("=== 1. Explicit Add ===")

procedures = [
    Procedure(
        title="How to Make Coffee",
        content=(
            "1. Boil water.\n"
            "2. Put coffee in a cup.\n"
            "3. Pour hot water into the cup.\n"
            "4. Stir and drink."
        ),
        user_id="alice",
        category="cooking",
    ),
    Procedure(
        title="How to Wash Clothes",
        content=(
            "1. Put clothes in the washing machine.\n"
            "2. Add soap.\n"
            "3. Press start.\n"
            "4. Take clothes out and dry them."
        ),
        user_id="alice",
        category="household",
    ),
]

for proc in procedures:
    manager.add(procedure=proc, user_id="alice")
    print(f"  [Stored] {proc.title}")
    for line in proc.content.splitlines():
        print(f"           {line}")
print()

# ---------------------------------------------------------------------------
# 2. LLM extraction
# ---------------------------------------------------------------------------

print("=== 2. LLM Extraction ===")

conversation = """
User: How do I water a plant?

Assistant: Here is the step by step guide:
Step 1: Check the soil — if it feels dry, it needs water.
Step 2: Pour water slowly at the base of the plant.
Step 3: Stop when water comes out of the bottom hole.
Step 4: Do this once or twice a week.
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

# ---------------------------------------------------------------------------
# 3. Search
# ---------------------------------------------------------------------------

print("=== 3. Search ===")

queries = [
    ("Direct",   "how to make coffee"),
    ("Indirect", "I want something warm to drink"),
    ("Direct",   "how to wash clothes"),
    ("Indirect", "my clothes are dirty"),
    ("Direct",   "how to water a plant"),
    ("Indirect", "my plant needs water"),
]

for label, q in queries:
    results = manager.search(q, user_id="alice")
    top = results[0] if results else None
    hit = f"score={top.score:.2f}  →  {top.procedure.title}" if top else "no result"
    print(f"  {label:8s}  '{q}'")
    print(f"           {hit}")
print()

# ---------------------------------------------------------------------------
# 4. Chat
# ---------------------------------------------------------------------------

print("=== 4. Chat ===")

questions = [
    ("Direct",   "How do I make coffee?"),
    ("Indirect", "I feel sleepy. What can I make quickly at home?"),
    ("Direct",   "How do I wash clothes?"),
    ("Indirect", "I have a lot of dirty clothes. What should I do?"),
]

for label, q in questions:
    print(f"  {label:8s}  Q: {q}")
    answer = manager.chat(q, user_id="alice")["response"]
    print(f"             A: {answer[:200]}{'...' if len(answer) > 200 else ''}\n")

time.sleep(2)
print(f"Total stored procedures: {len(manager.store.list_all(user_id='alice'))}")
