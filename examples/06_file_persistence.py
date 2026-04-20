#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
06. File Persistence — procedures survive between sessions.

EmulatedStore keeps everything in RAM and loses data when the process exits.
FileStore writes each procedure as a Markdown file so data persists across
multiple MemFlowManager instances and process restarts.

This example runs two independent sessions against the same data directory:

  Session 1  Create a MemFlowManager backed by FileStore.
             Store three procedures.  Exit the manager (object goes out of scope).

  Session 2  Create a brand-new MemFlowManager pointing at the same directory.
             list_all() and search() both find the procedures from Session 1.

The data directory is cleaned up at the end of the example.

Run:
  uv run ./examples/06_file_persistence.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memflow import FileStore, MemFlowManager, Procedure
from memflow.llm import LLMFactory

llm = LLMFactory.create("ollama", model="llama3.2")
data_dir = tempfile.mkdtemp(prefix="memflow_demo_")

try:
    # -----------------------------------------------------------------------
    # Session 1 — store procedures
    # -----------------------------------------------------------------------

    print(f"Data directory: {data_dir}\n")
    print("=== Session 1 — store three procedures ===")

    session1 = MemFlowManager(llm=llm, store=FileStore(data_dir=data_dir))

    procedures = [
        Procedure(
            title="How to Make Coffee",
            content=(
                "1. Boil water.\n"
                "2. Put coffee in a cup.\n"
                "3. Pour hot water into the cup.\n"
                "4. Stir and drink."
            ),
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
            category="household",
        ),
        Procedure(
            title="How to Water a Plant",
            content=(
                "1. Check the soil — if dry, it needs water.\n"
                "2. Pour water slowly at the base.\n"
                "3. Stop when water drains from the bottom.\n"
                "4. Water once or twice a week."
            ),
            category="gardening",
        ),
    ]

    for proc in procedures:
        session1.add(procedure=proc)
        print(f"  [Stored] {proc.title}")

    files = sorted(Path(data_dir).glob("*.md"))
    print(f"\n  Files written to disk: {len(files)}")
    for f in files:
        print(f"    {f.name}")

    # Discard the manager object — data lives only on disk from here on.
    del session1
    print("\n  Manager discarded (data lives on disk only)\n")

    # -----------------------------------------------------------------------
    # Session 2 — new manager, same directory
    # -----------------------------------------------------------------------

    print("=== Session 2 — new manager, same directory ===")

    session2 = MemFlowManager(llm=llm, store=FileStore(data_dir=data_dir))

    all_procs = session2.store.list_all()
    print(f"\n  list_all() found {len(all_procs)} procedure(s):")
    for p in all_procs:
        print(f"    [{p.category}] {p.title}")

    print()
    queries = [
        "I want something warm to drink",
        "my clothes are dirty",
        "my plant looks sad and dry",
    ]
    for q in queries:
        results = session2.search(q)
        top = results[0] if results else None
        hit = f"score={top.score:.2f}  →  {top.procedure.title}" if top else "no match"
        print(f"  Q: {q}")
        print(f"     {hit}")

finally:
    # shutil.rmtree(data_dir)
    print(f"\nData directory preserved for inspection: {data_dir}")
