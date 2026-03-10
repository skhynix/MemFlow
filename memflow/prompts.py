"""
Prompt templates for MemFlow.
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

CLASSIFICATION_PROMPT = """\
Classify the following content into exactly one memory type.

Memory types:
- procedural: ordered steps, workflows, "how to do X", SOPs
- semantic:   facts, definitions, "what X is", current state
- episodic:   past events with time context, "what happened when"
- none:       conversational filler, greetings, not worth storing

Respond ONLY with a JSON object: {"type": "procedural|semantic|episodic|none"}
"""

PLANNING_PROMPT = """\
You are a task planner with access to a procedural memory system.

## Relevant Procedures:
{procedures}

## Available Tools:
{tools}

Decompose the following task into concrete, executable steps.
Each step must use exactly one tool.

Task: {task}

Respond ONLY with a JSON object:
{{
    "steps": [
        {{"tool": "<tool_name>", "description": "<what this step does>", "args": {{<tool args>}}}}
    ]
}}
"""

LEARNING_PROMPT = """\
You are a procedural memory learning system.
Analyze the following task execution and extract a reusable procedure if applicable.

Task: {task}

Execution steps:
{steps}

If the execution succeeded and the steps are worth reusing, extract a procedure.

Respond ONLY with a JSON object in this format:
{{
    "has_procedure": true,
    "title": "Short descriptive title",
    "category": "deployment|debugging|configuration|workflow|other",
    "content": "The procedure in markdown with numbered steps"
}}

If the steps are not reusable or all steps failed, return:
{{"has_procedure": false}}
"""

CHAT_SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to a procedural memory system.

## Stored Procedures:
{procedures}

When answering:
- If a relevant procedure exists, reference it with specific steps.
- If no relevant procedure exists, provide your best guidance.
"""
