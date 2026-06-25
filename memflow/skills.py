# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for loading and indexing Codex-style SKILL.md files."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from memflow.models import Procedure, procedure_search_text

SUPPORTED_TRUST_STATES = {"trusted", "unknown", "blocked"}
_LIST_FRONTMATTER_FIELDS = {"tags", "aliases", "file_patterns", "tools"}


def _now_iso() -> str:
    return datetime.now().isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _normalize_frontmatter(frontmatter: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(frontmatter)
    for field in _LIST_FRONTMATTER_FIELDS:
        normalized[field] = _normalize_list(normalized.get(field))
    return normalized


def _resolve_skill_path(path: str | Path) -> Path:
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "SKILL.md"
    if source.name != "SKILL.md":
        raise ValueError(f"Expected a skill directory or SKILL.md path: {path}")
    if not source.exists():
        raise FileNotFoundError(f"SKILL.md not found: {source}")
    return source.resolve()


def parse_skill_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse leading YAML frontmatter from a SKILL.md document."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text

    closing_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        return {}, text

    raw_frontmatter = "".join(lines[1:closing_index])
    body = "".join(lines[closing_index + 1 :])
    try:
        data = yaml.safe_load(raw_frontmatter) if raw_frontmatter.strip() else {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid SKILL.md frontmatter: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("SKILL.md frontmatter must be a mapping")
    return data, body


def skill_id(source_path: Path, name: str = "") -> str:
    """Return the deterministic id for a skill source path."""
    del name
    normalized = str(source_path.expanduser().resolve())
    return f"skill:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


def _relative_skill_path(root_path: Path, source_path: Path) -> str:
    try:
        return source_path.relative_to(root_path.parent).as_posix()
    except ValueError:
        return source_path.name


def _heading_index(text: str) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            continue
        marker, _, title = stripped.partition(" ")
        if not title.strip() or set(marker) != {"#"}:
            continue
        headings.append(
            {
                "text": title.strip(),
                "level": len(marker),
                "line": line_number,
            }
        )
    return headings


def _body_index(text: str) -> dict[str, Any]:
    lines = text.splitlines(keepends=True)
    body_offset = 0
    body_start_line = 1
    frontmatter_present = False

    if lines and lines[0].strip() == "---":
        offset = len(lines[0])
        for index, line in enumerate(lines[1:], start=1):
            offset += len(line)
            if line.strip() == "---":
                body_offset = offset
                body_start_line = index + 2
                frontmatter_present = True
                break

    body = text[body_offset:]
    return {
        "frontmatter_present": frontmatter_present,
        "body_offset": body_offset,
        "body_start_line": body_start_line,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }


def build_resource_manifest(root_path: Path) -> dict[str, Any]:
    """Build manifests for scripts, references, and assets under a skill root."""
    root = root_path.expanduser().resolve()
    manifest: dict[str, Any] = {
        "has_aux_files": False,
        "scripts": [],
        "references": [],
        "assets": [],
    }

    for section in ("scripts", "references", "assets"):
        section_path = root / section
        if not section_path.exists() or not section_path.is_dir():
            continue
        entries = []
        for path in sorted(section_path.rglob("*")):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if not resolved.is_file() or not resolved.is_relative_to(root):
                continue
            stat = resolved.stat()
            entries.append(
                {
                    "path": resolved.relative_to(root).as_posix(),
                    "sha256": _sha256_file(resolved),
                    "size": stat.st_size,
                }
            )
        manifest[section] = entries

    manifest["has_aux_files"] = any(
        manifest[section] for section in ("scripts", "references", "assets")
    )
    return manifest


def _trusted_roots() -> list[Path]:
    configured = os.getenv("MEMFLOW_TRUSTED_SKILL_ROOTS")
    roots: list[Path] = []
    if configured:
        candidates = [item for item in configured.split(os.pathsep) if item]
    else:
        candidates = []
        codex_home = os.getenv("CODEX_HOME")
        if codex_home:
            candidates.append(str(Path(codex_home) / "skills"))
        home = os.getenv("HOME")
        if home:
            candidates.append(str(Path(home) / ".codex" / "skills"))

    for candidate in candidates:
        try:
            roots.append(Path(candidate).expanduser().resolve())
        except OSError:
            continue
    return roots


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_trust_state(
    source_path: Path,
    *,
    source: str = "local",
    trust_state: str | None = None,
) -> str:
    """Compute the stored trust state for a skill source path."""
    if trust_state is not None:
        if trust_state not in SUPPORTED_TRUST_STATES:
            raise ValueError(f"Unsupported trust_state: {trust_state}")
        return trust_state

    resolved_source = source_path.expanduser().resolve()
    if any(_is_relative_to(resolved_source, root) for root in _trusted_roots()):
        return "trusted"
    if source == "project":
        return "unknown"
    try:
        cwd = Path.cwd().resolve()
        if _is_relative_to(resolved_source, cwd):
            return "unknown"
    except OSError:
        pass
    return "unknown"


def _governance_metadata(trust_state: str, source: str) -> dict[str, Any]:
    if trust_state == "trusted":
        mode = "instruction"
        warnings: list[str] = []
    elif trust_state == "blocked":
        mode = "blocked"
        warnings = ["Skill is blocked and must not be injected."]
    else:
        mode = "data"
        warnings = ["Skill trust is unknown and must be treated as data."]

    return {
        "source": source,
        "trust_state": trust_state,
        "mode": mode,
        "warnings": warnings,
        "policy_version": "skill-policy-v1",
    }


def build_skill_metadata(
    *,
    root_path: Path,
    source_path: Path,
    raw_text: str,
    frontmatter: dict[str, Any],
    source: str = "local",
    trust_state: str | None = None,
    indexed_at: str | None = None,
) -> dict[str, Any]:
    """Build MemFlow metadata for a parsed skill."""
    indexed_at = indexed_at or _now_iso()
    normalized_frontmatter = _normalize_frontmatter(frontmatter)
    name = str(normalized_frontmatter.get("name") or root_path.name)
    description = str(normalized_frontmatter.get("description") or "")
    category = str(normalized_frontmatter.get("category") or "skill")
    tags = _normalize_list(normalized_frontmatter.get("tags"))
    content_bytes = raw_text.encode("utf-8")
    stat = source_path.stat()
    resolved_trust_state = resolve_trust_state(
        source_path, source=source, trust_state=trust_state
    )

    metadata = {
        "skill": {
            "name": name,
            "description": description,
            "root_path": str(root_path),
            "source_path": str(source_path),
            "relative_path": _relative_skill_path(root_path, source_path),
            "sha256": _sha256_bytes(content_bytes),
            "size": len(content_bytes),
            "mtime": stat.st_mtime,
            "frontmatter": normalized_frontmatter,
            "aliases": _normalize_list(normalized_frontmatter.get("aliases")),
            "file_patterns": _normalize_list(
                normalized_frontmatter.get("file_patterns")
            ),
            "tools": _normalize_list(normalized_frontmatter.get("tools")),
            "resources": build_resource_manifest(root_path),
        },
        "governance": _governance_metadata(resolved_trust_state, source),
        "index": {
            "indexed_at": indexed_at,
            "headings": _heading_index(raw_text),
            **_body_index(raw_text),
        },
    }
    metadata["index"]["search_text_sha256"] = hashlib.sha256(
        procedure_search_text(
            Procedure(
                title=name,
                content=raw_text,
                category=category,
                tags=tags,
                metadata=metadata,
            )
        ).encode("utf-8")
    ).hexdigest()
    metadata["index"]["token_estimate"] = len(raw_text.split())
    return metadata


def load_skill(
    path: str | Path,
    user_id: str = "default",
    *,
    source: str = "local",
    trust_state: str | None = None,
) -> Procedure:
    """Load a skill directory or direct SKILL.md path into a Procedure."""
    source_path = _resolve_skill_path(path)
    root_path = source_path.parent
    raw_text = source_path.read_text(encoding="utf-8")
    frontmatter, _body = parse_skill_frontmatter(raw_text)
    normalized = _normalize_frontmatter(frontmatter)
    name = str(normalized.get("name") or root_path.name)
    category = str(normalized.get("category") or "skill")
    tags = _normalize_list(normalized.get("tags"))
    now = _now_iso()
    metadata = build_skill_metadata(
        root_path=root_path,
        source_path=source_path,
        raw_text=raw_text,
        frontmatter=frontmatter,
        source=source,
        trust_state=trust_state,
        indexed_at=now,
    )
    return Procedure(
        id=skill_id(source_path, name),
        title=name,
        content=raw_text,
        user_id=user_id,
        category=category,
        tags=tags,
        kind="skill",
        source_path=str(source_path),
        metadata=metadata,
        created_at=now,
        updated_at=now,
    )


def indexed_skill_render_parts(
    procedure: Procedure,
) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    """Return frontmatter and body using indexed metadata only.

    The prompt/render path must not reparse SKILL.md frontmatter. New skill
    records carry a body offset under metadata["index"]. Legacy records fall
    back to the stored content so body markers are still injected.
    """
    skill = procedure.metadata.get("skill", {})
    if not isinstance(skill, dict):
        skill = {}
    frontmatter = skill.get("frontmatter", {})
    if not isinstance(frontmatter, dict):
        frontmatter = {}

    index = procedure.metadata.get("index", {})
    if not isinstance(index, dict):
        index = {}
    body_offset = index.get("body_offset")
    if isinstance(body_offset, int) and 0 <= body_offset <= len(procedure.content):
        return dict(frontmatter), procedure.content[body_offset:], ()

    return (
        dict(frontmatter),
        procedure.content,
        (f"legacy_skill_render_metadata_missing:{procedure.id}",),
    )


def render_skill_for_injection(procedure: Procedure) -> str:
    """Return the indexed skill body for injection without reparsing frontmatter."""
    _frontmatter, body, _warnings = indexed_skill_render_parts(procedure)
    return body
