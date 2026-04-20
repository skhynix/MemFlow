#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
01. Quickstart — store a procedure, then chat.

The simplest possible MemFlow example:
  1. Store one procedure explicitly.
  2. Ask a question and get an answer grounded in that procedure.

Run:
  ./examples/01_quickstart.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memflow import MemFlowManager, Procedure
from memflow.llm import LLMFactory

llm = LLMFactory.create("ollama", model="llama3.2")
manager = MemFlowManager(llm=llm)

# --- Store ---
proc = Procedure(
    title="How to Make Coffee",
    content=(
        "1. Boil water.\n"
        "2. Put coffee in a cup.\n"
        "3. Pour hot water into the cup.\n"
        "4. Stir and drink."
    ),
    user_id="alice",
    category="cooking",
)
manager.add(procedure=proc, user_id="alice")
print(f"[Stored] {proc.title}")
print(f"{proc.content}\n")

# --- Chat ---
question = "How do I make coffee?"
print(f"[Q] {question}")
answer = manager.chat(question, user_id="alice")
print(f"[A] {answer}")
