"""
Prompt templates for MemFlow Phase 1.
"""

EXTRACTION_PROMPT = """\
You are a procedural memory extraction system.
Analyze the conversation and extract any step-by-step procedures, workflows, or how-to knowledge.

Look for:
- Numbered steps (1., 2., 3., ...)
- Action verbs (check, pour, press, etc.)
- Sequential instructions

Respond ONLY with a JSON object in this format:
{"has_procedure": true, "title": "How to X", "category": "other", "content": "1. Do X. 2. Do Y."}

If no procedural knowledge is found, return: {"has_procedure": false}

IMPORTANT:
- Use single-line strings. Do NOT use triple quotes or newlines inside the JSON.
- Title format: "How to ..." (matches user query style)
- Keep content CONCISE: one sentence per step.
"""

CHAT_SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to a procedural memory system.

## Stored Procedures:
{procedures}

When answering:
- If a relevant procedure exists, reference it with specific steps.
- If no relevant procedure exists, provide your best guidance.
"""
