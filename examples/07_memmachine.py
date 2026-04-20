#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
07. MemMachine Backend — production-grade vector search with bypass routing.

With MEMFLOW_BACKEND=memmachine, MemFlow uses MemMachine as the primary store:

  - Procedures are stored in MemMachine VectorDB with real semantic search.
  - Semantic and episodic content is routed to MemMachineBypass, where
    MemMachine stores it in the appropriate backend (VectorDB / GraphDB).

The example is skipped gracefully if a MemMachine server is not available.

Requirements:
  - MemMachine server (default: http://localhost:8080)
  - OPENAI_API_KEY or another embedding provider configured in MemMachine

Run:
  MEMFLOW_BACKEND=memmachine ./examples/07_memmachine.py

  Or with a custom server:
  MEMFLOW_BACKEND=memmachine \\
    MEMMACHINE_BASE_URL=http://myserver:8080 \\
    ./examples/07_memmachine.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memflow import MemFlowManager, MemMachineBypass, MemMachineStore, Procedure
from memflow.llm import LLMFactory

# ---------------------------------------------------------------------------
# Check backend selection
# ---------------------------------------------------------------------------

backend = os.getenv("MEMFLOW_BACKEND", "emulated")
if backend != "memmachine":
    print("This example requires MEMFLOW_BACKEND=memmachine.")
    print("Run with:  MEMFLOW_BACKEND=memmachine ./examples/07_memmachine.py")
    sys.exit(0)

mm_url = os.getenv("MEMMACHINE_BASE_URL", "http://localhost:8080")
mm_org = os.getenv("MEMMACHINE_ORG_ID", "default")
mm_proj = os.getenv("MEMMACHINE_PROJECT", "memflow")
mm_key = os.getenv("MEMMACHINE_API_KEY")

# ---------------------------------------------------------------------------
# Verify MemMachine is reachable before running the demo
# ---------------------------------------------------------------------------

store = MemMachineStore(
    base_url=mm_url, org_id=mm_org, project_id=mm_proj, api_key=mm_key
)
bypass = MemMachineBypass(
    base_url=mm_url, org_id=mm_org, project_id=mm_proj, api_key=mm_key
)

print(f"MemMachine server: {mm_url}")
try:
    store._get_memory()
    print("Connection: OK\n")
except Exception as e:
    print(f"Connection failed: {e}")
    print("\nStart MemMachine with:  docker compose -f docker-compose.test.yml up -d")
    sys.exit(1)

llm = LLMFactory.create("ollama", model="llama3.2")
manager = MemFlowManager(llm=llm, store=store, bypass=bypass)

# ---------------------------------------------------------------------------
# 1. Store procedures (semantic vector search)
# ---------------------------------------------------------------------------

print("=== 1. Store Procedures ===")

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
        title="How to Restart a Service",
        content=(
            "1. Open a terminal.\n"
            "2. Run: sudo systemctl restart <service-name>\n"
            "3. Check status: sudo systemctl status <service-name>\n"
            "4. Look at logs if it fails: journalctl -u <service-name> -n 50"
        ),
        category="operations",
    ),
]

for proc in procedures:
    manager.add(procedure=proc)
    print(f"  [Stored in MemMachine] {proc.title}")

print()

# ---------------------------------------------------------------------------
# 2. Semantic search (vector similarity, not word overlap)
# ---------------------------------------------------------------------------

print("=== 2. Semantic Search ===")

queries = [
    "I need a hot beverage",           # similar to "coffee" by meaning
    "the application is not responding",  # similar to "restart a service"
]

for q in queries:
    results = manager.search(q, top_k=1)
    top = results[0] if results else None
    hit = f"score={top.score:.3f}  →  {top.procedure.title}" if top else "no match"
    print(f"  Q: {q}")
    print(f"     {hit}")

print()

# ---------------------------------------------------------------------------
# 3. Bypass routing for non-procedural content
# ---------------------------------------------------------------------------

print("=== 3. Bypass Routing ===")

inputs = [
    (
        "procedural",
        "How to water a plant:\n"
        "Step 1. Check the soil — if dry, water it.\n"
        "Step 2. Pour water slowly at the base.\n"
        "Step 3. Stop when water drains from the bottom.",
    ),
    (
        "semantic",
        "The first law of thermodynamics states that energy cannot be created "
        "or destroyed.",
    ),
    (
        "episodic",
        "Yesterday I ran the deployment and it failed because the config was wrong.",
    ),
]

for label, content in inputs:
    result = manager.add(messages=content)
    stored = result.get("results", [])
    skipped = result.get("skipped", "")
    routed = result.get("routed_to", "")

    if stored:
        action = f"stored in MemMachineStore → {stored[0]['title'][:40]}"
    elif routed:
        action = f"forwarded to MemMachineBypass (type={result.get('type')})"
    else:
        action = f"discarded ({skipped})"

    print(f"  [{label}]")
    print(f"    {action}")

print()

# ---------------------------------------------------------------------------
# 4. Chat using retrieved procedures
# ---------------------------------------------------------------------------

print("=== 4. Chat ===")

q = "How do I fix a service that stopped working?"
print(f"  Q: {q}")
answer = manager.chat(q, enable_auto_learn=False)
print(f"  A: {answer[:300]}{'...' if len(answer) > 300 else ''}")
