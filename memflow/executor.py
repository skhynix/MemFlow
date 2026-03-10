"""
Tool executor for MemFlow Phase 3.

ToolRegistry holds callable tools and executes Jobs one at a time,
returning a JobResult for each.

Built-in tools
--------------
  llm   — delegates to the LLM (requires llm= at init time)
  bash  — runs a shell command via subprocess (30 s timeout)
  http  — makes an HTTP request via urllib (no extra dependencies)

Custom tools can be registered with ToolRegistry.register(name, fn).
Each tool function receives the Job.args dict as keyword arguments and
must return a string.
"""

from __future__ import annotations

import subprocess
import urllib.request
from typing import Callable

from memflow.llm import BaseLLM
from memflow.models import Job, JobResult


class ToolRegistry:
    """Registry of named tools that execute Jobs."""

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self._tools: dict[str, Callable[..., str]] = {}
        if llm is not None:
            self.register("llm", _make_llm_tool(llm))
        self.register("bash", _bash_tool)
        self.register("http", _http_tool)

    def register(self, name: str, fn: Callable[..., str]) -> None:
        """Register a custom tool.  Overwrites any existing tool with the same name."""
        self._tools[name] = fn

    def available_tools(self) -> list[str]:
        return list(self._tools)

    def execute(self, job: Job) -> JobResult:
        """Execute a single Job and return a JobResult."""
        fn = self._tools.get(job.tool)
        if fn is None:
            return JobResult(
                job=job,
                success=False,
                output="",
                error=f"Unknown tool {job.tool!r}. Available: {self.available_tools()}",
            )
        try:
            output = fn(**job.args)
            return JobResult(job=job, success=True, output=str(output))
        except Exception as e:
            return JobResult(job=job, success=False, output="", error=str(e))


# ---------------------------------------------------------------------------
# Built-in tool implementations
# ---------------------------------------------------------------------------

def _make_llm_tool(llm: BaseLLM) -> Callable[..., str]:
    def _llm_tool(prompt: str) -> str:
        return llm.generate([{"role": "user", "content": prompt}])
    return _llm_tool


def _bash_tool(command: str) -> str:
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout
    if result.stderr:
        output += ("\n" if output else "") + result.stderr
    return output.strip()


def _http_tool(url: str, method: str = "GET", body: str = "") -> str:
    req = urllib.request.Request(url, method=method.upper())
    req.add_header("User-Agent", "MemFlow/1.0")
    if body:
        req.data = body.encode()
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read().decode(errors="replace")
