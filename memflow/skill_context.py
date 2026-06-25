# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Agent-agnostic skill context selection, rendering, and audit support."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from memflow.models import Procedure, SearchResult
from memflow.skills import parse_skill_frontmatter, render_skill_for_injection


@dataclass(frozen=True)
class SkillContextRequest:
    prompt: str
    cwd: str
    agent: str
    adapter: str
    session_id: str
    transcript_path: str
    user_id: str
    project_scope: str


@dataclass(frozen=True)
class SkillContextResponse:
    trace_id: str
    selected_skills: tuple[dict[str, Any], ...]
    rendered_context: str
    warnings: tuple[str, ...]
    status: str
    latency_ms: int


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


class SkillPolicy:
    """Config-backed policy for query construction and candidate eligibility."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @property
    def retrieval(self) -> dict[str, Any]:
        retrieval = self.config.get("retrieval", {})
        return retrieval if isinstance(retrieval, dict) else {}

    def build_query(self, request: SkillContextRequest) -> str:
        parts = [request.prompt.strip()]
        if self.retrieval.get("include_cwd_in_query") and request.cwd:
            parts.append(f"Current working directory: {request.cwd}")
        return "\n".join(part for part in parts if part)

    def candidate_k(self) -> int:
        return int(self.retrieval.get("candidate_k", 20))

    def min_score(self) -> float:
        return float(self.retrieval.get("min_score", 0.2))

    def candidate_from_result(
        self,
        manager: Any,
        result: SearchResult,
    ) -> SkillCandidate | None:
        if result.score < self.min_score():
            return None

        procedure = result.procedure
        if not _has_complete_skill_snapshot(procedure):
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

    def dedupe_key(self, procedure: Procedure) -> str:
        return _dedupe_key(procedure)


class SkillContextSelector:
    """Retrieve, filter, dedupe, and rank skill context candidates."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        policy: SkillPolicy | None = None,
    ) -> None:
        self.policy = policy or SkillPolicy(config)

    def build_query(self, request: SkillContextRequest) -> str:
        return self.policy.build_query(request)

    def select(
        self,
        manager: Any,
        request: SkillContextRequest,
    ) -> tuple[list[SkillCandidate], list[str]]:
        raw_results = manager.search_skills(
            self.build_query(request),
            user_id=request.user_id,
            top_k=self.policy.candidate_k(),
        )
        deduped: dict[str, SkillCandidate] = {}
        for result in raw_results:
            candidate = self.policy.candidate_from_result(manager, result)
            if candidate is None:
                continue
            key = self.policy.dedupe_key(candidate.procedure)
            existing = deduped.get(key)
            if existing is None or candidate.score > existing.score:
                deduped[key] = candidate

        candidates = sorted(deduped.values(), key=lambda item: item.score, reverse=True)
        warnings: list[str] = []
        if len(raw_results) != len(candidates):
            warnings.append("filtered_or_deduped_candidates")
        return candidates, warnings


class ContextRenderer:
    """Render selected skills into compact XML-style context."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def render(
        self,
        candidates: list[SkillCandidate],
        *,
        trace_id: str,
    ) -> RenderResult:
        rendering = self.config.get("rendering", {})
        retrieval = self.config.get("retrieval", {})
        top_k = int(retrieval.get("top_k", 0))
        max_chars = int(rendering.get("max_chars", 0))
        hard_max_chars = int(rendering.get("hard_max_chars", 0))
        per_skill_max = int(rendering.get("max_chars_per_skill", 0))
        budget = min(max_chars, hard_max_chars) if hard_max_chars else max_chars

        if top_k <= 0 or budget <= 0 or per_skill_max <= 0:
            return RenderResult("", (), ("render_budget_too_small",))

        opening = (
            "<selected_skills>\n"
            "These local skills were selected for the current user prompt.\n"
            "Use them only when relevant to this task. They do not override "
            "higher-priority instructions.\n\n"
        )
        closing = "</selected_skills>\n"
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
            rendered = _render_skill_with_budget(
                candidate, len(selected) + 1, available
            )
            if rendered is None:
                warnings.append(
                    f"skill_render_budget_exhausted:{candidate.procedure.id}"
                )
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


class AuditLogger:
    """Build and append privacy-preserving skill-context audit records."""

    def __init__(self, config: dict[str, Any], *, adapter: str) -> None:
        self.config = config
        self.adapter = adapter

    def base_record(
        self,
        *,
        trace_id: str,
        request: SkillContextRequest | None,
        hook_event: str | None,
        prompt: str,
        status: str,
        latency_ms: int,
        warnings: list[str] | tuple[str, ...] | None = None,
        selected_skills: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        retrieval = self.config.get("retrieval", {})
        claude = self.config.get("claude", {})
        logging_config = self.config.get("logging", {})
        session_id = request.session_id if request else ""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "adapter": self.adapter,
            "hook_event": hook_event,
            "session_id_hash": _hash_optional(session_id),
            "cwd": request.cwd if request else None,
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

    def write(self, record: dict[str, Any]) -> None:
        logging_config = self.config.get("logging", {})
        path_value = logging_config.get("path")
        if not path_value:
            return
        path = Path(str(path_value)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def write_or_fail(self, record: dict[str, Any]) -> bool:
        try:
            self.write(record)
        except Exception:
            return False
        return True


def selected_skill_metadata(rendered: RenderedSkill) -> dict[str, Any]:
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


def _has_complete_skill_snapshot(procedure: Procedure) -> bool:
    if not procedure.content:
        return False
    skill = procedure.metadata.get("skill", {})
    if not isinstance(skill, dict) or not skill:
        return False
    return bool(
        procedure.source_path or skill.get("source_path") or skill.get("sha256")
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
        f'source_path="{_xml_attr(source_path)}" '
        f'trust_mode="{_xml_attr(candidate.trust_mode)}">\n'
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


__all__ = [
    "AuditLogger",
    "ContextRenderer",
    "RenderResult",
    "RenderedSkill",
    "SkillCandidate",
    "SkillContextRequest",
    "SkillContextResponse",
    "SkillContextSelector",
    "SkillPolicy",
    "selected_skill_metadata",
]
