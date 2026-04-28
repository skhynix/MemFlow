#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
05. Memory Classification — LLM routes content to the right memory type.

2-stage pipeline (classification + extraction):

  Stage 1  LLM classification — identifies memory type
           procedural → extract and store in procedure store
           semantic   → discard (or forward to MemMachine bypass if configured)
           episodic   → discard (or forward to MemMachine bypass if configured)
           none       → discard

  Stage 2  LLM extraction — extracts structured procedure from content

This example feeds four inputs — one of each type (procedural, semantic,
episodic, none) — through the pipeline and shows how each one is classified
and handled.

Run:
  uv run ./examples/05_classification.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memflow import MemFlowManager

# LLM and store are loaded from .env file automatically
manager = MemFlowManager()

# ---------------------------------------------------------------------------
# Four inputs covering all memory types (including 'none' for conversational filler)
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
        "none",
        "Thanks so much! That was really helpful. Chat with you later.",
    ),
]

print("=" * 68)
print("  2-Stage Pipeline (Classification + Extraction)")
print("=" * 68)
print(f"  {'Type':<22}  {'Stage 1':<12}  Action")
print(f"  {'-'*22}  {'-'*12}  {'-'*30}")

for label, content in inputs:
    result = manager.add(messages=content, user_id="demo")

    skipped = result.get("skipped", "")
    routed = result.get("routed_to", "")
    stored = result.get("results", [])

    if stored:
        stage1 = "procedural"
        titles = [r["title"] for r in stored]
        action = f"stored → {titles[0][:30]}"
    elif routed:
        stage1 = result.get("type", "?")
        action = f"→ bypass ({result['type']})"
    else:
        stage1 = skipped.replace("classified as ", "") if skipped else "?"
        action = "discarded"

    print(f"  {label:<22}  {stage1:<12}  {action}")

print()
print(f"Procedures stored: {len(manager.store.list_all())}")
