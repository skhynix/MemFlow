#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
02. Search — direct vs indirect queries.

Stores two procedures and shows how search behaves:
  - Direct query: keywords match the stored title closely.
  - Indirect query: same intent but different words.

Run:
  uv run ./examples/02_search.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memflow import MemFlowManager, Procedure
from memflow.llm import LLMFactory

llm = LLMFactory.create("ollama", model="llama3.2")
manager = MemFlowManager(llm=llm)

# --- Store two procedures ---
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
    print(f"[Stored] {proc.title}")
print()

# --- Search: direct and indirect ---
queries = [
    ("Direct",   "how to make coffee"),
    ("Indirect", "I want something warm to drink"),
    ("Direct",   "how to wash clothes"),
    ("Indirect", "my clothes are dirty"),
]

print("[Search]")
for label, q in queries:
    results = manager.search(q, user_id="alice")
    top = results[0] if results else None
    hit = f"score={top.score:.2f}  →  {top.procedure.title}" if top else "no result"
    print(f"  {label:8s}  '{q}'")
    print(f"           {hit}")
