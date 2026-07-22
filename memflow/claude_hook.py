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
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

from memflow.claude_catalog import normalize_native_catalog_mode
from memflow.claude_session_state import (
    ActivationUpdate,
    ClaudeSessionStateStore,
    StateOperationResult,
    resolve_state_dir,
)
from memflow.llm import BaseLLM
from memflow.skill_context import (
    AuditLogger,
    ContextRenderer,
    RenderedSkill,
    SkillContextRequest,
    SkillContextResponse,
    SkillContextSelector,
    selected_skill_metadata,
)

ADAPTER_NAME = "claude-code-user-prompt-submit"
DEFAULT_CONFIG_PATH = ".memflow/claude-hook.json"
DEFAULT_RETRIEVAL_TIMEOUT_MS = 2000
DEFAULT_SESSION_DEDUPE_ROLLOUT = "off"
DEFAULT_SESSION_DEDUPE_POLICY = "on_hash_change"
DEFAULT_SESSION_DEDUPE_STATE_DIR = "claude-sessions"
SUPPORTED_SESSION_DEDUPE_ROLLOUTS = {"off", "shadow"}

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
        "session_dedupe": {
            "rollout": DEFAULT_SESSION_DEDUPE_ROLLOUT,
            "policy": DEFAULT_SESSION_DEDUPE_POLICY,
            "state_dir": DEFAULT_SESSION_DEDUPE_STATE_DIR,
        },
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


@dataclass(frozen=True)
class _SessionDedupePlan:
    """One fail-open dedupe decision over an already-rendered baseline."""

    requested_rollout: str | None
    rollout: str
    configuration_status: str
    policy: str
    output_skills: tuple[RenderedSkill, ...]
    record_skills: tuple[RenderedSkill, ...]
    reused_skills: tuple[str, ...]
    would_reuse_skills: tuple[str, ...]
    state_load_status: str
    context_generation: int | None
    warnings: tuple[str, ...] = ()
    store: ClaudeSessionStateStore | None = None


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


def _normalize_session_dedupe_config(
    config: dict[str, Any],
    *,
    config_path: str | Path | None,
) -> dict[str, Any]:
    """Return effective fail-closed-to-off session dedupe configuration."""
    warnings: list[str] = []
    claude_config = config.get("claude")
    raw_config = (
        claude_config.get("session_dedupe") if isinstance(claude_config, dict) else None
    )
    if not isinstance(raw_config, dict):
        warnings.append("invalid_session_dedupe_config")
        raw_config = {}

    rollout = raw_config.get("rollout", DEFAULT_SESSION_DEDUPE_ROLLOUT)
    if not isinstance(rollout, str) or rollout not in SUPPORTED_SESSION_DEDUPE_ROLLOUTS:
        warnings.append("invalid_session_dedupe_rollout")

    policy = raw_config.get("policy", DEFAULT_SESSION_DEDUPE_POLICY)
    if policy != DEFAULT_SESSION_DEDUPE_POLICY:
        warnings.append("invalid_session_dedupe_policy")

    state_dir = raw_config.get("state_dir", DEFAULT_SESSION_DEDUPE_STATE_DIR)
    if not isinstance(state_dir, str) or not state_dir.strip():
        warnings.append("invalid_session_dedupe_state_dir")
        state_dir = DEFAULT_SESSION_DEDUPE_STATE_DIR

    try:
        resolved_state_dir = resolve_state_dir(
            state_dir,
            config_path=config_path or DEFAULT_CONFIG_PATH,
        )
    except (OSError, TypeError, ValueError):
        warnings.append("invalid_session_dedupe_state_dir")
        resolved_state_dir = resolve_state_dir(
            DEFAULT_SESSION_DEDUPE_STATE_DIR,
            config_path=config_path or DEFAULT_CONFIG_PATH,
        )

    requested_rollout = rollout if isinstance(rollout, str) else None
    effective_rollout = (
        rollout
        if not warnings and isinstance(rollout, str)
        else DEFAULT_SESSION_DEDUPE_ROLLOUT
    )
    return {
        "requested_rollout": requested_rollout,
        "rollout": effective_rollout,
        "configuration_status": ("valid" if not warnings else "invalid_fallback_off"),
        "policy": DEFAULT_SESSION_DEDUPE_POLICY,
        "state_dir": str(resolved_state_dir),
        "warnings": list(dict.fromkeys(warnings)),
    }


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
    catalog_mode = normalize_native_catalog_mode(config)
    config["_memflow_catalog_mode"] = {
        "raw": catalog_mode.raw_mode,
        "effective": catalog_mode.effective_mode,
        "warnings": list(catalog_mode.warnings),
    }
    config["_memflow_session_dedupe"] = _normalize_session_dedupe_config(
        config,
        config_path=config_path,
    )
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


def _session_dedupe_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("_memflow_session_dedupe", {})
    if isinstance(settings, dict):
        return settings
    return {
        "requested_rollout": DEFAULT_SESSION_DEDUPE_ROLLOUT,
        "rollout": DEFAULT_SESSION_DEDUPE_ROLLOUT,
        "configuration_status": "invalid_fallback_off",
        "policy": DEFAULT_SESSION_DEDUPE_POLICY,
        "state_dir": DEFAULT_SESSION_DEDUPE_STATE_DIR,
        "warnings": ["invalid_session_dedupe_config"],
    }


def _baseline_dedupe_plan(
    config: dict[str, Any],
    baseline_skills: Sequence[RenderedSkill] = (),
    *,
    state_load_status: str,
    warnings: Sequence[str] = (),
) -> _SessionDedupePlan:
    settings = _session_dedupe_settings(config)
    return _SessionDedupePlan(
        requested_rollout=settings.get("requested_rollout"),
        rollout=str(settings.get("rollout") or DEFAULT_SESSION_DEDUPE_ROLLOUT),
        configuration_status=str(
            settings.get("configuration_status") or "invalid_fallback_off"
        ),
        policy=str(settings.get("policy") or DEFAULT_SESSION_DEDUPE_POLICY),
        output_skills=tuple(baseline_skills),
        record_skills=(),
        reused_skills=(),
        would_reuse_skills=(),
        state_load_status=state_load_status,
        context_generation=None,
        warnings=tuple(warnings),
    )


def _matching_activations(
    baseline_skills: Sequence[RenderedSkill],
    stored_activations: dict[str, Any],
) -> tuple[RenderedSkill, ...]:
    matches: list[RenderedSkill] = []
    for rendered in baseline_skills:
        identity = rendered.identity
        fingerprint = rendered.activation_fingerprint
        if not identity or not fingerprint:
            continue
        stored = stored_activations.get(identity)
        if stored is not None and stored.activation_fingerprint == fingerprint:
            matches.append(rendered)
    return tuple(matches)


def _plan_session_dedupe(
    config: dict[str, Any],
    baseline_skills: Sequence[RenderedSkill],
    *,
    session_id: str,
) -> _SessionDedupePlan:
    """Compare only the completed baseline and fail open on state errors."""
    settings = _session_dedupe_settings(config)
    rollout = str(settings.get("rollout") or DEFAULT_SESSION_DEDUPE_ROLLOUT)
    policy = str(settings.get("policy") or DEFAULT_SESSION_DEDUPE_POLICY)
    baseline = tuple(baseline_skills)

    if rollout == "off":
        return _baseline_dedupe_plan(
            config,
            baseline,
            state_load_status="off",
        )
    if not baseline:
        return _baseline_dedupe_plan(
            config,
            state_load_status="not_needed",
        )
    if not session_id:
        return _baseline_dedupe_plan(
            config,
            baseline,
            state_load_status="missing_session_id",
            warnings=("session_dedupe_missing_session_id",),
        )

    try:
        store = ClaudeSessionStateStore(str(settings["state_dir"]))
        loaded = store.load(session_id)
    except Exception:
        return _baseline_dedupe_plan(
            config,
            baseline,
            state_load_status="load_failed",
            warnings=("session_dedupe_state_load_failed",),
        )

    if not loaded.available:
        return _baseline_dedupe_plan(
            config,
            baseline,
            state_load_status=loaded.status,
            warnings=loaded.warnings,
        )

    assert loaded.state is not None
    try:
        matching = _matching_activations(baseline, loaded.state.activations)
        matching_identities = tuple(
            rendered.identity for rendered in matching if rendered.identity
        )
    except Exception:
        return _baseline_dedupe_plan(
            config,
            baseline,
            state_load_status="compare_failed",
            warnings=("session_dedupe_compare_failed",),
        )

    if rollout == "shadow":
        return _SessionDedupePlan(
            requested_rollout=settings.get("requested_rollout"),
            rollout=rollout,
            configuration_status=str(
                settings.get("configuration_status") or "invalid_fallback_off"
            ),
            policy=policy,
            output_skills=baseline,
            record_skills=baseline,
            reused_skills=(),
            would_reuse_skills=matching_identities,
            state_load_status=loaded.status,
            context_generation=loaded.state.context_generation,
            store=store,
        )
    return _baseline_dedupe_plan(
        config,
        baseline,
        state_load_status=loaded.status,
    )


def _activation_identities(
    rendered_skills: Sequence[RenderedSkill],
) -> list[str]:
    return [
        rendered.identity
        for rendered in rendered_skills
        if isinstance(rendered.identity, str) and rendered.identity
    ]


def _rendered_skill_name(rendered: RenderedSkill) -> str:
    procedure = rendered.candidate.procedure
    skill = procedure.metadata.get("skill", {})
    if not isinstance(skill, dict):
        skill = {}
    frontmatter = skill.get("frontmatter", {})
    if not isinstance(frontmatter, dict):
        frontmatter = {}
    return str(skill.get("name") or frontmatter.get("name") or procedure.title)


def _add_session_dedupe_audit(
    record: dict[str, Any],
    plan: _SessionDedupePlan,
    *,
    dedupe_saved_chars: int,
) -> None:
    record.update(
        {
            "injected_skills": _activation_identities(plan.output_skills),
            "reused_skills": list(plan.reused_skills),
            "would_reuse_skills": list(plan.would_reuse_skills),
            "skipped_skills": [
                {
                    "identity": identity,
                    "reason": "duplicate_activation_fingerprint",
                }
                for identity in plan.reused_skills
            ],
            "session_dedupe": {
                "requested_rollout": plan.requested_rollout,
                "rollout": plan.rollout,
                "configuration_status": plan.configuration_status,
                "policy": plan.policy,
                "state_load_status": plan.state_load_status,
                "context_generation": plan.context_generation,
                "lifecycle_tracking": "user_prompt_submit_only",
                "dedupe_saved_chars": dedupe_saved_chars,
            },
        }
    )


def _record_session_activations(
    plan: _SessionDedupePlan,
    *,
    session_id: str,
) -> StateOperationResult | None:
    """Record payloads prepared for output, without claiming delivery."""
    if plan.store is None or plan.context_generation is None:
        return None

    try:
        updates = []
        invalidated_identities = []
        for rendered in plan.record_skills:
            identity = rendered.identity
            if not identity:
                continue
            fingerprint = rendered.activation_fingerprint
            if not fingerprint:
                invalidated_identities.append(identity)
                continue
            updates.append(
                ActivationUpdate(
                    identity=identity,
                    activation_fingerprint=fingerprint,
                    name=_rendered_skill_name(rendered),
                    emitted_chars=rendered.rendered_chars,
                )
            )
        if updates or invalidated_identities:
            return plan.store.record_activations(
                session_id,
                updates,
                expected_generation=plan.context_generation,
                invalidated_identities=invalidated_identities,
            )
        return StateOperationResult("unchanged")
    except Exception:
        # The already-audited stdout plan must survive best-effort state failure.
        return StateOperationResult(
            "record_failed",
            warnings=("session_dedupe_state_record_failed",),
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
    catalog_mode = config.get("_memflow_catalog_mode", {})
    if not isinstance(catalog_mode, dict):
        catalog_mode = {}
    catalog_warnings = tuple(str(item) for item in catalog_mode.get("warnings", ()))
    session_dedupe_settings = _session_dedupe_settings(config)
    session_dedupe_config_warnings = tuple(
        str(item) for item in session_dedupe_settings.get("warnings", ())
    )

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
                warnings=[*catalog_warnings, "unsupported_hook_event"],
            )
            audit_logger.write_or_fail(record)
            return ""

        query = selector.build_query(context_request)
        if not query.strip():
            dedupe_plan = _plan_session_dedupe(
                config,
                (),
                session_id=context_request.session_id,
            )
            context_response = SkillContextResponse(
                trace_id=trace_id,
                selected_skills=(),
                rendered_context="",
                warnings=(
                    *catalog_warnings,
                    *session_dedupe_config_warnings,
                    *dedupe_plan.warnings,
                    "empty_query",
                ),
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
            _add_session_dedupe_audit(
                record,
                dedupe_plan,
                dedupe_saved_chars=0,
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
        baseline = renderer.render(candidates, trace_id=trace_id)
        dedupe_plan = _plan_session_dedupe(
            config,
            baseline.skills,
            session_id=context_request.session_id,
        )
        output_context = baseline.xml
        selected_skills = tuple(
            selected_skill_metadata(rendered) for rendered in baseline.skills
        )
        warnings = (
            *catalog_warnings,
            *session_dedupe_config_warnings,
            *selection_warnings,
            *baseline.warnings,
            *dedupe_plan.warnings,
        )
        if not baseline.xml:
            status = "no_results"
        else:
            status = "injected"
        context_response = SkillContextResponse(
            trace_id=trace_id,
            selected_skills=selected_skills,
            rendered_context=output_context,
            warnings=warnings,
            status=status,
            latency_ms=latency_ms(),
        )
        planned_stdout = ""
        if context_response.rendered_context:
            response = {
                "suppressOutput": True,
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": context_response.rendered_context,
                },
            }
            planned_stdout = json.dumps(response)
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
        dedupe_saved_chars = 0
        _add_session_dedupe_audit(
            record,
            dedupe_plan,
            dedupe_saved_chars=dedupe_saved_chars,
        )
        if not audit_logger.write_or_fail(record):
            return ""
        state_record_result = _record_session_activations(
            dedupe_plan,
            session_id=context_request.session_id,
        )
        if state_record_result is not None and not state_record_result.succeeded:
            state_record_warnings = state_record_result.warnings or (
                f"claude_session_state_{state_record_result.status}",
            )
            state_record_failure = audit_logger.base_record(
                trace_id=context_response.trace_id,
                request=context_request,
                hook_event=hook_input.hook_event_name,
                prompt=prompt,
                status="state_record_failed",
                latency_ms=latency_ms(),
                warnings=state_record_warnings,
            )
            state_record_failure["audit_event"] = "session_dedupe_state_record"
            state_record_failure["session_dedupe"] = {
                "requested_rollout": dedupe_plan.requested_rollout,
                "rollout": dedupe_plan.rollout,
                "configuration_status": dedupe_plan.configuration_status,
                "policy": dedupe_plan.policy,
                "state_load_status": dedupe_plan.state_load_status,
                "state_record_status": state_record_result.status,
                "context_generation": dedupe_plan.context_generation,
                "lifecycle_tracking": "user_prompt_submit_only",
            }
            audit_logger.write_or_fail(state_record_failure)
        return planned_stdout
    except Exception as exc:
        is_user_prompt_submit = (
            hook_input is not None and hook_input.hook_event_name == "UserPromptSubmit"
        )
        warnings = [*catalog_warnings]
        if is_user_prompt_submit:
            warnings.extend(session_dedupe_config_warnings)
        warnings.append(f"{type(exc).__name__}")
        record = audit_logger.base_record(
            trace_id=trace_id,
            request=context_request,
            hook_event=hook_input.hook_event_name if hook_input else None,
            prompt=prompt,
            status="fail_open",
            latency_ms=latency_ms(),
            warnings=warnings,
        )
        if is_user_prompt_submit:
            fail_open_plan = _plan_session_dedupe(
                config,
                (),
                session_id=hook_input.session_id,
            )
            _add_session_dedupe_audit(
                record,
                fail_open_plan,
                dedupe_saved_chars=0,
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
