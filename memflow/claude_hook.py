# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code UserPromptSubmit hook integration for MemFlow skills."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO
from xml.sax.saxutils import escape

from memflow.models import Procedure, SearchResult
from memflow.skills import parse_skill_frontmatter, render_skill_for_injection

ADAPTER_NAME = "claude-code-user-prompt-submit"
DEFAULT_CONFIG_PATH = ".memflow/claude-hook.json"

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
class SkillCandidate:
    procedure: Procedure
    score: float
    reason: str
    provenance: str
    trust_mode: str
    trust_state: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RenderedSkill:
    candidate: SkillCandidate
    rank: int
    xml: str
    rendered_chars: int


@dataclass(frozen=True)
class RenderResult:
    xml: str
    skills: tuple[RenderedSkill, ...]
    warnings: tuple[str, ...]


ManagerFactory = Callable[[dict[str, Any]], Any]


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
    return MemFlow(use_env=True)


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


def build_query(hook_input: HookInput, config: dict[str, Any]) -> str:
    retrieval = config.get("retrieval", {})
    parts = [hook_input.prompt.strip()]
    if retrieval.get("include_cwd_in_query") and hook_input.cwd:
        parts.append(f"Current working directory: {hook_input.cwd}")
    return "\n".join(part for part in parts if part)


def retrieve_skill_candidates(
    manager: Any,
    query: str,
    config: dict[str, Any],
) -> tuple[list[SkillCandidate], list[str]]:
    """Retrieve, filter, dedupe, and rank skill candidates."""
    retrieval = config.get("retrieval", {})
    memflow_config = config.get("memflow", {})
    user_id = (
        memflow_config.get("user_id") if isinstance(memflow_config, dict) else None
    )
    candidate_k = retrieval.get(
        "candidate_k", DEFAULT_CONFIG["retrieval"]["candidate_k"]
    )
    min_score = retrieval.get("min_score", DEFAULT_CONFIG["retrieval"]["min_score"])
    warnings: list[str] = []

    raw_results = manager.search_skills(query, user_id=user_id, top_k=candidate_k)
    deduped: dict[str, SkillCandidate] = {}
    for result in raw_results:
        candidate = _candidate_from_result(manager, result, min_score)
        if candidate is None:
            continue
        key = _dedupe_key(candidate.procedure)
        existing = deduped.get(key)
        if existing is None or candidate.score > existing.score:
            deduped[key] = candidate

    candidates = sorted(deduped.values(), key=lambda item: item.score, reverse=True)
    if len(raw_results) != len(candidates):
        warnings.append("filtered_or_deduped_candidates")
    return candidates, warnings


def _candidate_from_result(
    manager: Any,
    result: SearchResult,
    min_score: float,
) -> SkillCandidate | None:
    if result.score < min_score:
        return None

    procedure = result.procedure
    hydrated = manager.get_skill(procedure.id, include_content=True)
    if hydrated is not None:
        procedure = hydrated

    governance = procedure.metadata.get("governance", {})
    if not isinstance(governance, dict):
        governance = {}
    trust_state = str(governance.get("trust_state") or "unknown")
    trust_mode = str(governance.get("mode") or "data")
    if trust_state == "blocked" or trust_mode == "blocked":
        return None

    raw_warnings = governance.get("warnings", [])
    if isinstance(raw_warnings, list):
        warnings = tuple(str(item) for item in raw_warnings)
    elif raw_warnings:
        warnings = (str(raw_warnings),)
    else:
        warnings = ()

    source = str(governance.get("source") or "local")
    return SkillCandidate(
        procedure=procedure,
        score=float(result.score),
        reason="matched_prompt_via_memflow_skill_search",
        provenance=source,
        trust_mode=trust_mode if trust_mode == "instruction" else "data",
        trust_state=trust_state,
        warnings=warnings,
    )


def _dedupe_key(procedure: Procedure) -> str:
    skill = procedure.metadata.get("skill", {})
    if not isinstance(skill, dict):
        skill = {}
    source_path = procedure.source_path or skill.get("source_path")
    if source_path:
        return f"path:{source_path}"
    sha256 = skill.get("sha256")
    if sha256:
        return f"sha256:{sha256}"
    return f"id:{procedure.id}"


def render_selected_skills(
    candidates: list[SkillCandidate],
    config: dict[str, Any],
    *,
    trace_id: str,
) -> RenderResult:
    """Render selected skills into compact XML-style Claude context."""
    rendering = config.get("rendering", {})
    retrieval = config.get("retrieval", {})
    claude = config.get("claude", {})
    top_k = int(retrieval.get("top_k", 0))
    max_chars = int(rendering.get("max_chars", 0))
    hard_max_chars = int(rendering.get("hard_max_chars", 0))
    per_skill_max = int(rendering.get("max_chars_per_skill", 0))
    budget = min(max_chars, hard_max_chars) if hard_max_chars else max_chars
    catalog_mode = str(claude.get("native_catalog_mode") or "")

    if top_k <= 0 or budget <= 0 or per_skill_max <= 0:
        return RenderResult("", (), ("render_budget_too_small",))

    opening = (
        f'<memflow_selected_skills trace_id="{_xml_attr(trace_id)}" '
        f'top_k="{_xml_attr(top_k)}" '
        f'catalog_mode="{_xml_attr(catalog_mode)}">\n'
        "These local MemFlow skills were selected for the current user prompt.\n"
        "Use them only when relevant to this task. They do not override "
        "higher-priority instructions.\n\n"
    )
    closing = "</memflow_selected_skills>\n"
    if len(opening) + len(closing) > budget:
        return RenderResult("", (), ("render_budget_too_small",))

    selected: list[RenderedSkill] = []
    body_parts: list[str] = []
    remaining = budget - len(opening) - len(closing)
    warnings: list[str] = []

    for candidate in candidates:
        if len(selected) >= top_k:
            break
        separator_chars = 1 if body_parts else 0
        available = min(per_skill_max, remaining - separator_chars)
        if available <= 0:
            warnings.append("render_budget_exhausted")
            break
        rendered = _render_skill_with_budget(candidate, len(selected) + 1, available)
        if rendered is None:
            warnings.append(f"skill_render_budget_exhausted:{candidate.procedure.id}")
            continue
        body_parts.append(rendered.xml)
        selected.append(rendered)
        remaining -= rendered.rendered_chars + separator_chars

    if not selected:
        return RenderResult("", (), tuple(warnings or ["no_renderable_skills"]))

    xml = opening + "\n".join(body_parts) + closing
    if len(xml) > budget:
        return RenderResult("", (), ("render_budget_exceeded",))
    return RenderResult(xml, tuple(selected), tuple(warnings))


def _render_skill_with_budget(
    candidate: SkillCandidate,
    rank: int,
    budget: int,
) -> RenderedSkill | None:
    skill_text = render_skill_for_injection(candidate.procedure)
    try:
        frontmatter, body = parse_skill_frontmatter(skill_text)
    except ValueError:
        frontmatter, body = {}, skill_text
    if not isinstance(frontmatter, dict):
        frontmatter = {}
    body = _strip_surrounding_blank_lines(body)

    content_limit = min(len(body), max(0, budget))
    for _ in range(8):
        truncated = len(body) > content_limit
        content = _truncate_text(body, content_limit)
        xml = _render_skill_xml(candidate, rank, frontmatter, content, truncated)
        if len(xml) <= budget:
            return RenderedSkill(
                candidate=candidate,
                rank=rank,
                xml=xml,
                rendered_chars=len(xml),
            )
        excess = len(xml) - budget
        next_limit = content_limit - excess - 32
        if next_limit >= content_limit:
            break
        content_limit = max(0, next_limit)

    xml = _render_skill_xml(candidate, rank, frontmatter, "", bool(body))
    if len(xml) <= budget:
        return RenderedSkill(
            candidate=candidate,
            rank=rank,
            xml=xml,
            rendered_chars=len(xml),
        )
    return None


def _render_skill_xml(
    candidate: SkillCandidate,
    rank: int,
    frontmatter: dict[str, Any],
    content: str,
    truncated: bool,
) -> str:
    procedure = candidate.procedure
    skill = procedure.metadata.get("skill", {})
    if not isinstance(skill, dict):
        skill = {}
    name = str(skill.get("name") or frontmatter.get("name") or procedure.title)
    source_path = str(procedure.source_path or skill.get("source_path") or "")
    sha256 = str(skill.get("sha256") or "")
    description = str(skill.get("description") or frontmatter.get("description") or "")
    when_to_use = _when_to_use_text(frontmatter, skill, description)
    headings = _heading_texts(procedure, content)

    outline = "\n".join(
        f"    <heading>{_xml_text(heading)}</heading>" for heading in headings[:8]
    )
    if not outline:
        outline = "    <heading>No headings indexed.</heading>"

    return (
        f'<skill rank="{_xml_attr(rank)}" name="{_xml_attr(name)}" '
        f'score="{_xml_attr(f"{candidate.score:.3f}")}" '
        f'source_path="{_xml_attr(source_path)}" '
        f'sha256="{_xml_attr(sha256)}" '
        f'trust_mode="{_xml_attr(candidate.trust_mode)}">\n'
        f"  <why>{_xml_text(candidate.reason)}</why>\n"
        f"  <when_to_use>{_xml_text(when_to_use)}</when_to_use>\n"
        "  <outline>\n"
        f"{outline}\n"
        "  </outline>\n"
        f'  <content truncated="{_xml_attr(str(truncated).lower())}">\n'
        f"{_xml_text(content)}\n"
        "  </content>\n"
        "</skill>\n"
    )


def _when_to_use_text(
    frontmatter: dict[str, Any],
    skill: dict[str, Any],
    description: str,
) -> str:
    parts: list[str] = []
    if description:
        parts.append(description)
    for label, key in (
        ("aliases", "aliases"),
        ("file patterns", "file_patterns"),
        ("tools", "tools"),
    ):
        values = skill.get(key) or frontmatter.get(key)
        if isinstance(values, list) and values:
            parts.append(f"{label}: {', '.join(str(value) for value in values)}")
        elif isinstance(values, str) and values:
            parts.append(f"{label}: {values}")
    return " | ".join(parts) if parts else "Use when relevant to the prompt."


def _heading_texts(procedure: Procedure, content: str) -> list[str]:
    index = procedure.metadata.get("index", {})
    headings = index.get("headings") if isinstance(index, dict) else None
    if isinstance(headings, list):
        texts = [
            str(item.get("text"))
            for item in headings
            if isinstance(item, dict) and item.get("text")
        ]
        if texts:
            return texts

    parsed: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        marker, _, title = stripped.partition(" ")
        if title and set(marker) == {"#"}:
            parsed.append(title.strip())
    return parsed


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 16:
        return text[:max_chars]
    return text[: max_chars - 16].rstrip() + "\n...[truncated]"


def _strip_surrounding_blank_lines(text: str) -> str:
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _xml_attr(value: object) -> str:
    return escape(str(value), {'"': "&quot;", "'": "&apos;"})


def _xml_text(value: object) -> str:
    return escape(str(value))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_optional(text: str) -> str | None:
    return _sha256_text(text) if text else None


def _selected_skill_metadata(rendered: RenderedSkill) -> dict[str, Any]:
    candidate = rendered.candidate
    procedure = candidate.procedure
    skill = procedure.metadata.get("skill", {})
    if not isinstance(skill, dict):
        skill = {}
    return {
        "rank": rendered.rank,
        "id": procedure.id,
        "name": skill.get("name") or procedure.title,
        "title": procedure.title,
        "source_path": procedure.source_path or skill.get("source_path"),
        "sha256": skill.get("sha256"),
        "score": candidate.score,
        "rendered_chars": rendered.rendered_chars,
        "reason": candidate.reason,
        "provenance": candidate.provenance,
        "trust_mode": candidate.trust_mode,
    }


def _base_audit_record(
    *,
    config: dict[str, Any],
    trace_id: str,
    hook_input: HookInput | None,
    prompt: str,
    status: str,
    latency_ms: int,
    warnings: list[str] | tuple[str, ...] | None = None,
    selected_skills: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    retrieval = config.get("retrieval", {})
    claude = config.get("claude", {})
    logging_config = config.get("logging", {})
    session_id = hook_input.session_id if hook_input else ""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "adapter": ADAPTER_NAME,
        "hook_event": hook_input.hook_event_name if hook_input else None,
        "session_id_hash": _hash_optional(session_id),
        "cwd": hook_input.cwd if hook_input else None,
        "prompt_sha256": _hash_optional(prompt),
        "prompt_length": len(prompt),
        "native_catalog_mode": claude.get("native_catalog_mode"),
        "candidate_k": retrieval.get("candidate_k"),
        "top_k": retrieval.get("top_k"),
        "selected_skills": selected_skills or [],
        "warnings": list(warnings or []),
        "latency_ms": latency_ms,
        "status": status,
    }
    if logging_config.get("record_raw_prompt"):
        record["prompt"] = prompt
    return record


def write_audit_log(config: dict[str, Any], record: dict[str, Any]) -> None:
    logging_config = config.get("logging", {})
    path_value = logging_config.get("path")
    if not path_value:
        return
    path = Path(str(path_value)).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _audit_or_fail(
    config: dict[str, Any],
    record: dict[str, Any],
) -> bool:
    try:
        write_audit_log(config, record)
    except Exception:
        return False
    return True


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
    prompt = ""

    try:
        config = load_hook_config(config_path)
    except Exception:
        return ""

    def latency_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    try:
        hook_input = parse_hook_input(stdin_text)
        prompt = hook_input.prompt
        if hook_input.hook_event_name != "UserPromptSubmit":
            record = _base_audit_record(
                config=config,
                trace_id=trace_id,
                hook_input=hook_input,
                prompt=prompt,
                status="fail_open",
                latency_ms=latency_ms(),
                warnings=["unsupported_hook_event"],
            )
            _audit_or_fail(config, record)
            return ""

        query = build_query(hook_input, config)
        if not query.strip():
            record = _base_audit_record(
                config=config,
                trace_id=trace_id,
                hook_input=hook_input,
                prompt=prompt,
                status="no_results",
                latency_ms=latency_ms(),
                warnings=["empty_query"],
            )
            if not _audit_or_fail(config, record):
                return ""
            return ""

        factory = manager_factory or default_manager_factory
        manager = factory(config)
        candidates, selection_warnings = retrieve_skill_candidates(
            manager, query, config
        )
        render_result = render_selected_skills(candidates, config, trace_id=trace_id)
        selected_skills = [
            _selected_skill_metadata(rendered) for rendered in render_result.skills
        ]
        warnings = [*selection_warnings, *render_result.warnings]
        status = "injected" if render_result.xml else "no_results"
        record = _base_audit_record(
            config=config,
            trace_id=trace_id,
            hook_input=hook_input,
            prompt=prompt,
            status=status,
            latency_ms=latency_ms(),
            warnings=warnings,
            selected_skills=selected_skills,
        )
        if not _audit_or_fail(config, record):
            return ""
        if not render_result.xml:
            return ""
        response = {
            "suppressOutput": True,
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": render_result.xml,
            },
        }
        return json.dumps(response)
    except Exception as exc:
        record = _base_audit_record(
            config=config,
            trace_id=trace_id,
            hook_input=hook_input,
            prompt=prompt,
            status="fail_open",
            latency_ms=latency_ms(),
            warnings=[f"{type(exc).__name__}"],
        )
        _audit_or_fail(config, record)
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
