#!/usr/bin/env python3
"""
05. Memory Classification — LLM routes content to the right memory type.

Phase 2 adds a Stage 2 LLM classifier between the keyword heuristic (Stage 1)
and the extraction step (Stage 3).  The classifier identifies the memory type
and decides what to do with the content:

  Stage 1  Keyword heuristic — fast, no LLM call
           No keywords found → skip immediately (saves LLM cost)

  Stage 2  LLM classification — accurate
           procedural → extract and store in procedure store
           semantic   → skip (or forward to MemMachine bypass if configured)
           episodic   → skip (or forward to MemMachine bypass if configured)
           none       → discard

This example feeds four inputs — one of each type — through the pipeline and
shows which stage filtered each one and why.

Run:
  ./examples/05_classification.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memflow import MemFlowManager
from memflow.llm import LLMFactory

llm = LLMFactory.create("ollama", model="llama3.2")
manager = MemFlowManager(llm=llm)

# ---------------------------------------------------------------------------
# Four inputs covering all four memory types (and one Stage-1 block)
# ---------------------------------------------------------------------------

inputs = [
    (
        "procedural",
        "How to make instant noodles:\n"
        "Step 1. Boil 500 ml of water.\n"
        "Step 2. Put noodles in the boiling water.\n"
        "Step 3. Cook for 3 minutes, then add the seasoning.\n"
        "Step 4. Pour into a bowl and eat.",
    ),
    (
        "semantic",
        "The first law of thermodynamics states that energy cannot be created "
        "or destroyed, only converted from one form to another.",
    ),
    (
        "episodic",
        "Yesterday I ran the deployment script and it failed because the "
        "database connection was not configured correctly.",
    ),
    (
        "filler (no keywords)",
        "Thanks so much! That was really helpful. Chat with you later.",
    ),
]

print("=" * 68)
print("  3-Stage Classification Pipeline")
print("=" * 68)
print(f"  {'Type':<22}  {'Stage 1':<14}  {'Stage 2':<12}  Action")
print(f"  {'-'*22}  {'-'*14}  {'-'*12}  {'-'*20}")

for label, content in inputs:
    result = manager.add(messages=content, user_id="demo")

    skipped = result.get("skipped", "")
    routed = result.get("routed_to", "")
    stored = result.get("results", [])

    if "no procedural keywords" in skipped:
        stage1 = "BLOCKED"
        stage2 = "—"
        action = "discarded (Stage 1)"
    elif stored:
        stage1 = "passed"
        stage2 = "procedural"
        titles = [r["title"] for r in stored]
        action = f"stored → {titles[0][:30]}"
    elif routed:
        stage1 = "passed"
        stage2 = result.get("type", "?")
        action = f"→ bypass ({result['type']})"
    else:
        stage1 = "passed"
        stage2 = skipped.replace("classified as ", "") if skipped else "?"
        action = f"discarded (Stage 2)"

    print(f"  {label:<22}  {stage1:<14}  {stage2:<12}  {action}")

print()
print(f"Procedures stored: {len(manager.store.list_all())}")
