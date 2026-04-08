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

Plan only the first {max_steps} step(s) — you will replan after execution.

IMPORTANT:
1. Each step must produce MEASURABLE OUTPUT:
   - bash: a command that produces output (e.g., "date", "ls", "cat file.txt")
   - llm: a question that the LLM can answer from its knowledge
   - http: a valid URL that returns data
2. Each step's "args" must match the tool's expected parameters exactly:
   - For "bash" tool: only use {{"command": "your shell command"}}
   - For "llm" tool: only use {{"prompt": "your question"}}
   - For "http" tool: use {{"url": "...", "method": "GET", "body": ""}}
   - Do NOT invent new argument names like "output", "src", "filename", etc.
3. Think about WHAT EACH STEP ACCOMPLISHES:
   - "Get current date" → bash: "date" (produces actual date)
   - NOT: llm: "what is the date?" (LLM doesn't know real-time info)
4. For file operations, use commands that PRODUCE CONSOLE OUTPUT:
   - Create file with timestamp: "date | tee file.txt" (outputs AND saves)
   - Create file with content: "echo 'hello' > file.txt && echo 'Created file.txt'"
   - Read file: "cat file.txt"
   - Copy file AND verify: "cp src.txt dst.txt && ls -la dst.txt"
   - NEVER use redirection alone (e.g., "date > file.txt") — this produces NO console output

Respond ONLY with a JSON object:
{{
    "steps": [
        {{"tool": "<tool_name>", "description": "<what this step does>", "args": {{<tool args>}}}}
    ]
}}
"""

REPLAN_PROMPT = """\
You are a task planner with access to a procedural memory system.
The task is partially complete. Review the execution history and plan the next steps.

## Relevant Procedures:
{procedures}

## Available Tools:
{tools}

## Task:
{task}

## Execution History:
{history}

Based on the results above, plan the next {max_steps} step(s) to complete the task.

IMPORTANT:
1. DO NOT repeat steps that already SUCCEEDED with meaningful output.
2. Evaluate if each step ACTUALLY ACHIEVED ITS GOAL:
   - Success with EMPTY output → step didn't really accomplish anything
   - Success with MEANINGFUL output → step worked correctly (DO NOT REPEAT)
   - Failure → try a different approach
3. If ALL previous steps FAILED with the same error, do NOT repeat them — try a completely different approach.
4. If steps SUCCEEDED but produced EMPTY/MEANINGLESS output, the approach is wrong — try different commands.
5. If the task appears complete (e.g., file created, output generated), return an empty steps list.
6. Each step's "args" must match the tool's expected parameters exactly:
   - For "bash" tool: only use {{"command": "your shell command"}}
   - For "llm" tool: only use {{"prompt": "your question"}}
   - For "http" tool: use {{"url": "...", "method": "GET", "body": ""}}
   - Do NOT invent new argument names like "output", "src", "filename", etc.
7. For file operations, use commands that PRODUCE CONSOLE OUTPUT:
   - Create file with timestamp: "date | tee file.txt" (outputs AND saves)
   - Create file with content: "echo 'hello' > file.txt && echo 'Created file.txt'"
   - Read file: "cat file.txt"
   - Copy file AND verify: "cp src.txt dst.txt && ls -la dst.txt"
   - NEVER use redirection alone (e.g., "date > file.txt") — this produces NO console output

If the task is already complete or no productive next step exists, return:
{{"steps": []}}

Respond ONLY with a JSON object:
{{
    "steps": [
        {{"tool": "<tool_name>", "description": "<what this step does>", "args": {{<tool args>}}}}
    ]
}}
"""

LEARNING_PROMPT = """\
You are a procedural memory learning system.
Analyze the following task execution and extract a reusable procedure.

Task: {task}

Execution steps:
{steps}

IMPORTANT:
- If the task succeeded (at least one step produced meaningful output), ALWAYS extract a procedure.
- Even simple tasks are worth learning — they become building blocks for complex tasks.
- Focus on WHAT worked, not whether it's "interesting" or "complex".
- Format the procedure as clear, numbered steps that anyone can follow.

Respond ONLY with a JSON object in this format:
{{
    "has_procedure": true,
    "title": "Short descriptive title (e.g., 'How to {task}')",
    "category": "deployment|debugging|configuration|workflow|file-operations|system|other",
    "content": "The procedure in markdown with numbered steps"
}}

Only return {{"has_procedure": false}} if ALL steps failed or produced no output.
"""

CHAT_SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to a procedural memory system.

## Stored Procedures:
{procedures}

When answering:
- If a relevant procedure exists, reference it with specific steps.
- If no relevant procedure exists, provide your best guidance.
"""

INTENT_CLASSIFICATION_PROMPT = """\
Classify the user's intent into one or more of these categories.

- SEARCH: User is asking about how to do something, looking for existing knowledge
  (e.g., "How do I...?", "What's the procedure for...?", "Do you know how to...?")

- ADD: User wants to save/store new procedural knowledge
  (e.g., "Remember this...", "Save this procedure...", "Don't forget...")

- EXECUTE: User wants to perform an action or task
  (e.g., "Run this...", "Execute...", "Do this for me...")

- CONVERSATION: General chat, questions, or anything that doesn't fit above
  (e.g., greetings, "How are you?", factual questions, opinions)

IMPORTANT:
- A single message may contain MULTIPLE intents (e.g., "Find the deploy procedure and run it")
- Classify ALL intents that apply, in the order they should be processed
- If no specific intent applies, use only CONVERSATION

Respond ONLY with a JSON object:
{"intents": ["SEARCH", "EXECUTE"], "primary": "SEARCH"}

Where:
- "intents": list of intent types in processing order
- "primary": the main intent (for response formatting)
"""
