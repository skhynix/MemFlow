# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code UserPromptSubmit hook integration for MemFlow skills."""

from __future__ import annotations

import argparse
import copy
import json
import signal
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

from memflow.llm import BaseLLM
from memflow.skill_context import (
    AuditLogger,
    ContextRenderer,
    SkillContextRequest,
    SkillContextResponse,
    SkillContextSelector,
    selected_skill_metadata,
)

ADAPTER_NAME = "claude-code-user-prompt-submit"
DEFAULT_CONFIG_PATH = ".memflow/claude-hook.json"
DEFAULT_RETRIEVAL_TIMEOUT_MS = 2000

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": "memflow.claude_hook.v1",
    "memflow": {
        "env_file": ".env",
        "reuse_existing_config": True,
        "store": "PgVectorStore",
        "user_id": "default",
    },
    "claude": {
        "native_catalog_mode": "hidden_or_minimized",
    },
    "retrieval": {
        "top_k": 3,
        "max_top_k": 5,
        "candidate_k": 20,
        "min_score": 0.2,
        "include_cwd_in_query": True,
        "timeout_ms": DEFAULT_RETRIEVAL_TIMEOUT_MS,
    },
    "rendering": {
        "max_chars": 6000,
        "hard_max_chars": 10000,
        "max_chars_per_skill": 3000,
        "format": "selected_skills_xml_v1",
    },
    "logging": {
        "path": ".memflow/logs/skill_context_hook.jsonl",
        "record_raw_prompt": False,
        "record_skill_body": False,
    },
}


@dataclass(frozen=True)
class HookInput:
    session_id: str
    transcript_path: str
    cwd: str
    hook_event_name: str
    prompt: str


ManagerFactory = Callable[[dict[str, Any]], Any]


class RetrievalTimeoutError(TimeoutError):
    """Raised when MemFlow skill retrieval exceeds the hook timeout."""


class _HookRetrievalOnlyLLM(BaseLLM):
    """LLM placeholder for hook paths that only need store-backed retrieval."""

    def generate(self, messages: list[dict]) -> str:
        del messages
        raise RuntimeError("Claude hook skill retrieval does not support LLM calls")


def _deep_merge(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _as_int(value: Any, default: int, *, minimum: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, number)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_hook_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load hook config, using defaults when the config file is absent."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path:
        path = Path(config_path).expanduser()
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("Claude hook config must be a JSON object")
            config = _deep_merge(config, loaded)

    retrieval = config.setdefault("retrieval", {})
    default_retrieval = DEFAULT_CONFIG["retrieval"]
    max_top_k = _as_int(
        retrieval.get("max_top_k"),
        default_retrieval["max_top_k"],
        minimum=1,
    )
    top_k = _as_int(retrieval.get("top_k"), default_retrieval["top_k"], minimum=0)
    top_k = min(top_k, max_top_k)
    candidate_k = _as_int(
        retrieval.get("candidate_k"),
        default_retrieval["candidate_k"],
        minimum=0,
    )
    retrieval["max_top_k"] = max_top_k
    retrieval["top_k"] = top_k
    retrieval["candidate_k"] = max(candidate_k, top_k)
    retrieval["min_score"] = _as_float(
        retrieval.get("min_score"), default_retrieval["min_score"]
    )
    retrieval["include_cwd_in_query"] = bool(retrieval.get("include_cwd_in_query"))
    retrieval["timeout_ms"] = _as_int(
        retrieval.get("timeout_ms"),
        DEFAULT_RETRIEVAL_TIMEOUT_MS,
        minimum=0,
    )

    rendering = config.setdefault("rendering", {})
    default_rendering = DEFAULT_CONFIG["rendering"]
    rendering["max_chars"] = _as_int(
        rendering.get("max_chars"), default_rendering["max_chars"], minimum=0
    )
    rendering["hard_max_chars"] = _as_int(
        rendering.get("hard_max_chars"),
        default_rendering["hard_max_chars"],
        minimum=0,
    )
    rendering["max_chars_per_skill"] = _as_int(
        rendering.get("max_chars_per_skill"),
        default_rendering["max_chars_per_skill"],
        minimum=0,
    )

    config.setdefault("memflow", copy.deepcopy(DEFAULT_CONFIG["memflow"]))
    config.setdefault("claude", copy.deepcopy(DEFAULT_CONFIG["claude"]))
    config.setdefault("logging", copy.deepcopy(DEFAULT_CONFIG["logging"]))
    return config


def default_manager_factory(config: dict[str, Any]) -> Any:
    """Build MemFlow from the current environment and optional config env file."""
    from memflow.manager import MemFlow, _load_env_file

    memflow_config = config.get("memflow", {})
    if isinstance(memflow_config, dict):
        env_file = memflow_config.get("env_file")
        if env_file:
            _load_env_file(str(env_file))
    return MemFlow(llm=_HookRetrievalOnlyLLM(), use_env=True)


def parse_hook_input(stdin_text: str) -> HookInput:
    payload = json.loads(stdin_text)
    if not isinstance(payload, dict):
        raise ValueError("Claude hook stdin must be a JSON object")
    return HookInput(
        session_id=str(payload.get("session_id") or ""),
        transcript_path=str(payload.get("transcript_path") or ""),
        cwd=str(payload.get("cwd") or ""),
        hook_event_name=str(payload.get("hook_event_name") or ""),
        prompt=str(payload.get("prompt") or ""),
    )


def build_skill_context_request(
    hook_input: HookInput,
    config: dict[str, Any],
) -> SkillContextRequest:
    memflow_config = config.get("memflow", {})
    if not isinstance(memflow_config, dict):
        memflow_config = {}
    return SkillContextRequest(
        prompt=hook_input.prompt,
        cwd=hook_input.cwd,
        agent="claude-code",
        adapter=ADAPTER_NAME,
        session_id=hook_input.session_id,
        transcript_path=hook_input.transcript_path,
        user_id=str(memflow_config.get("user_id") or "default"),
        project_scope=str(memflow_config.get("project_scope") or hook_input.cwd),
    )


@contextmanager
def retrieval_timeout(timeout_ms: int):
    """Raise RetrievalTimeoutError when retrieval exceeds timeout_ms.

    Claude hooks run on the prompt path, so the runtime CLI uses SIGALRM on
    Unix to fail open instead of waiting indefinitely on a store call.
    """
    if (
        timeout_ms <= 0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def raise_timeout(_signum, _frame):
        raise RetrievalTimeoutError("MemFlow skill retrieval timed out")

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_ms / 1000)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(
                signal.ITIMER_REAL,
                previous_timer[0],
                previous_timer[1],
            )


def run_hook(
    stdin_text: str,
    *,
    config_path: str | Path | None = DEFAULT_CONFIG_PATH,
    manager_factory: ManagerFactory | None = None,
) -> str:
    """Run the Claude hook and return stdout content."""
    started = time.perf_counter()
    trace_id = uuid.uuid4().hex
    config: dict[str, Any]
    hook_input: HookInput | None = None
    context_request: SkillContextRequest | None = None
    prompt = ""

    try:
        config = load_hook_config(config_path)
    except Exception:
        return ""

    def latency_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    audit_logger = AuditLogger(config, adapter=ADAPTER_NAME)
    selector = SkillContextSelector(config)
    renderer = ContextRenderer(config)

    try:
        hook_input = parse_hook_input(stdin_text)
        context_request = build_skill_context_request(hook_input, config)
        prompt = context_request.prompt
        if hook_input.hook_event_name != "UserPromptSubmit":
            record = audit_logger.base_record(
                trace_id=trace_id,
                request=context_request,
                hook_event=hook_input.hook_event_name,
                prompt=prompt,
                status="fail_open",
                latency_ms=latency_ms(),
                warnings=["unsupported_hook_event"],
            )
            audit_logger.write_or_fail(record)
            return ""

        query = selector.build_query(context_request)
        if not query.strip():
            context_response = SkillContextResponse(
                trace_id=trace_id,
                selected_skills=(),
                rendered_context="",
                warnings=("empty_query",),
                status="no_results",
                latency_ms=latency_ms(),
            )
            record = audit_logger.base_record(
                trace_id=context_response.trace_id,
                request=context_request,
                hook_event=hook_input.hook_event_name,
                prompt=prompt,
                status=context_response.status,
                latency_ms=context_response.latency_ms,
                warnings=context_response.warnings,
            )
            if not audit_logger.write_or_fail(record):
                return ""
            return ""

        retrieval_config = config.get("retrieval", {})
        timeout_ms = int(retrieval_config.get("timeout_ms", 0))
        with retrieval_timeout(timeout_ms):
            factory = manager_factory or default_manager_factory
            manager = factory(config)
            candidates, selection_warnings = selector.select(manager, context_request)
        render_result = renderer.render(candidates, trace_id=trace_id)
        selected_skills = tuple(
            selected_skill_metadata(rendered) for rendered in render_result.skills
        )
        warnings = (*selection_warnings, *render_result.warnings)
        status = "injected" if render_result.xml else "no_results"
        context_response = SkillContextResponse(
            trace_id=trace_id,
            selected_skills=selected_skills,
            rendered_context=render_result.xml,
            warnings=warnings,
            status=status,
            latency_ms=latency_ms(),
        )
        record = audit_logger.base_record(
            trace_id=context_response.trace_id,
            request=context_request,
            hook_event=hook_input.hook_event_name,
            prompt=prompt,
            status=context_response.status,
            latency_ms=context_response.latency_ms,
            warnings=context_response.warnings,
            selected_skills=list(context_response.selected_skills),
        )
        if not audit_logger.write_or_fail(record):
            return ""
        if not context_response.rendered_context:
            return ""
        response = {
            "suppressOutput": True,
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": context_response.rendered_context,
            },
        }
        return json.dumps(response)
    except Exception as exc:
        record = audit_logger.base_record(
            trace_id=trace_id,
            request=context_request,
            hook_event=hook_input.hook_event_name if hook_input else None,
            prompt=prompt,
            status="fail_open",
            latency_ms=latency_ms(),
            warnings=[f"{type(exc).__name__}"],
        )
        audit_logger.write_or_fail(record)
        return ""


def _default_config_text() -> str:
    return json.dumps(DEFAULT_CONFIG, indent=2, sort_keys=True) + "\n"


def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    manager_factory: ManagerFactory | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--print-default-config",
        action="store_true",
        help="print the default hook config JSON and exit",
    )
    args = parser.parse_args(argv)

    out = stdout or sys.stdout
    if args.print_default_config:
        out.write(_default_config_text())
        return 0

    input_stream = stdin or sys.stdin
    output = run_hook(
        input_stream.read(),
        config_path=args.config,
        manager_factory=manager_factory,
    )
    if output:
        out.write(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
