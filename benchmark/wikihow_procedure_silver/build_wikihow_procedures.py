#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SOURCE_DATASET = "paolop/human-instructions-dataset-updated-json-files"
SOURCE_LICENSE = "CC-BY-NC-SA-4.0"
SOURCE_NAME = "human-instructions-dataset-updated-json-files"
EXPECTED_PROCEDURES_SHA256 = (
    "d4364ff7dd35ceac71d0f7f86b5c1f2cae2de6120403fdcfc5ca36ec0bad0f8f"
)
EXPECTED_PROCEDURES_RECORDS = 132_157
CATEGORY_ALIAS_PROVENANCE_PATH = "data/normalization/category_aliases.json"
DEFAULT_CATEGORY_ALIASES = (
    Path(__file__).resolve().parent / "assets/category_aliases.json"
)
NON_WORD_RE = re.compile(r"[^a-z0-9]+")
SHARD_RE = re.compile(r"wikiHow(\d+)\.json$")


class CorpusBuildError(RuntimeError):
    """Raised when the local WikiHow corpus build cannot be verified."""


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _slug(text: str) -> str:
    return NON_WORD_RE.sub("-", text.lower()).strip("-")[:72]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_id(shard_name: str, row_index: int, url: str, title: str) -> str:
    key = f"{shard_name}:{row_index}:{url}:{title}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    slug = _slug(title)
    return f"wikihow-{slug}-{digest}" if slug else f"wikihow-row-{row_index}-{digest}"


def _shard_sort_key(path: Path) -> tuple[int, str]:
    match = SHARD_RE.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10_000_000, path.name


def _raw_shards(input_dir: Path, pattern: str) -> list[Path]:
    return sorted(input_dir.glob(pattern), key=_shard_sort_key)


def _load_category_normalization(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing category normalization asset: {path}")

    asset = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(asset, dict):
        raise SystemExit(f"{path} must contain a JSON object")

    alias_map = asset.get("alias_map")
    if not isinstance(alias_map, dict):
        raise SystemExit(f"{path} must contain an alias_map object")
    for raw_label, canonical_label in alias_map.items():
        if not isinstance(raw_label, str) or not isinstance(canonical_label, str):
            raise SystemExit(f"{path} alias_map keys and values must be strings")
        if not raw_label.strip() or not canonical_label.strip():
            raise SystemExit(f"{path} alias_map keys and values must be non-empty")

    unresolved = asset.get("unresolved", [])
    if unresolved:
        raise SystemExit(f"{path} contains unresolved normalization decisions")

    return {
        "alias_count": len(alias_map),
        "alias_map": alias_map,
        "path": CATEGORY_ALIAS_PROVENANCE_PATH,
        "schema_version": asset.get("schema_version"),
        "sha256": _sha256(path),
        "unresolved_count": len(unresolved),
    }


def _step_groups(
    article: dict[str, Any],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    direct_steps = article.get("Steps") or []
    if direct_steps:
        groups.append(("steps", "", direct_steps))

    for method in article.get("Methods") or []:
        groups.append(
            ("method", _clean(method.get("MethodName")), method.get("Steps") or [])
        )

    for part in article.get("Parts") or []:
        groups.append(("part", _clean(part.get("PartName")), part.get("steps") or []))

    return groups


def _extract_steps(article: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for section_type, section_name, raw_steps in _step_groups(article):
        for raw_step in raw_steps:
            action = _clean(raw_step.get("Headline"))
            if not action:
                continue
            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "action": action,
                    "state": _clean(raw_step.get("Description")),
                    "section_type": section_type,
                    "section_name": section_name,
                }
            )
    return steps


def _raw_category_path(article: dict[str, Any]) -> list[str]:
    categories = [_clean(item) for item in article.get("Categories") or []]
    return [item for item in categories if item]


def _leaf_category(categories: list[str]) -> str:
    return categories[-1] if categories else "wikihow"


def _normalize_categories(
    categories: list[str], alias_map: dict[str, str]
) -> tuple[list[str], list[dict[str, str]]]:
    normalized_categories = []
    aliases_applied = []
    for category in categories:
        normalized_category = alias_map.get(category, category)
        normalized_categories.append(normalized_category)
        if normalized_category != category:
            aliases_applied.append(
                {
                    "canonical_label": normalized_category,
                    "raw_label": category,
                }
            )
    return normalized_categories, aliases_applied


def _content(title: str, summary: str, steps: list[dict[str, Any]]) -> str:
    lines = []
    if summary:
        lines.extend([f"Summary: {summary}", ""])
    lines.append("Steps:")
    for step in steps:
        line = f"{step['step_id']}. {step['action']}"
        if step["state"]:
            line += f" {step['state']}"
        lines.append(line)
    return "\n".join(lines)


def _metadata(
    article: dict[str, Any],
    shard_name: str,
    row_index: int,
    summary: str,
    category_normalization: dict[str, Any],
) -> dict[str, Any]:
    raw_categories = article.get("Categories") or []
    raw_category_path = _raw_category_path(article)
    normalized_categories, aliases_applied = _normalize_categories(
        raw_category_path,
        category_normalization["alias_map"],
    )
    return {
        "authors_count": article.get("AuthorsCount"),
        "categories": raw_categories,
        "category_normalization": {
            "alias_asset": category_normalization["path"],
            "alias_asset_sha256": category_normalization["sha256"],
            "aliases_applied": aliases_applied,
            "applied": bool(aliases_applied),
            "schema_version": category_normalization["schema_version"],
        },
        "main_task_summary": summary,
        "normalized_categories": normalized_categories,
        "normalized_category": _leaf_category(normalized_categories),
        "raw_category": _leaf_category(raw_category_path),
        "source": f"kaggle:{SOURCE_DATASET}",
        "source_dataset": SOURCE_DATASET,
        "source_license": SOURCE_LICENSE,
        "source_row_index": row_index,
        "source_shard": shard_name,
        "time": article.get("Time"),
        "url": _clean(article.get("URL")),
        "views": article.get("Views"),
    }


def _record_for_article(
    article: dict[str, Any],
    shard_name: str,
    row_index: int,
    min_steps: int,
    category_normalization: dict[str, Any],
) -> dict[str, Any] | None:
    title = _clean(article.get("MainTask"))
    if not title:
        return None

    steps = _extract_steps(article)
    if len(steps) < min_steps:
        return None

    summary = _clean(article.get("MainTaskSummary"))
    url = _clean(article.get("URL"))
    procedure_id = _stable_id(shard_name, row_index, url, title)
    content = _content(title=title, summary=summary, steps=steps)
    metadata = _metadata(
        article=article,
        shard_name=shard_name,
        row_index=row_index,
        summary=summary,
        category_normalization=category_normalization,
    )
    category = metadata["normalized_category"]
    tags = [
        "wikihow",
        "human-instructions",
        f"category:{category}",
        f"steps:{len(steps)}",
    ]

    procedure = {
        "category": category,
        "content": content,
        "id": procedure_id,
        "metadata": metadata,
        "tags": tags,
        "title": title,
    }
    return procedure


def _write_jsonl_record(handle: Any, record: dict[str, Any]) -> int:
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    handle.write(line + "\n")
    return len(line.encode("utf-8")) + 1


def _output_info(path: Path, records: int) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "records": records,
        "sha256": _sha256(path),
    }


def build(
    input_dir: Path,
    output_dir: Path,
    shard_pattern: str,
    min_steps: int,
    limit: int | None,
    category_aliases: Path,
) -> dict[str, Any]:
    shards = _raw_shards(input_dir, shard_pattern)
    if not shards:
        raise SystemExit(
            f"No input shards found in {input_dir} matching {shard_pattern}"
        )

    category_normalization = _load_category_normalization(category_aliases)

    output_dir.mkdir(parents=True, exist_ok=True)
    procedures_path = output_dir / "wikihow_procedures.jsonl"
    manifest_path = output_dir / "MANIFEST.json"

    raw_files = []
    counts = {
        "articles_read": 0,
        "category_alias_applications": 0,
        "procedures": 0,
        "records_with_category_aliases": 0,
        "skipped_missing_title": 0,
        "skipped_too_few_steps": 0,
        "total_steps": 0,
    }
    seen_ids: set[str] = set()

    with procedures_path.open("w", encoding="utf-8") as procedures_file:
        for shard in shards:
            articles = json.loads(shard.read_text(encoding="utf-8"))
            if not isinstance(articles, list):
                raise SystemExit(f"{shard} must contain a JSON array")

            shard_counts = {
                "articles_read": 0,
                "category_alias_applications": 0,
                "procedures": 0,
                "records_with_category_aliases": 0,
                "skipped_missing_title": 0,
                "skipped_too_few_steps": 0,
            }
            for row_index, article in enumerate(articles):
                if limit is not None and counts["procedures"] >= limit:
                    break
                if not isinstance(article, dict):
                    continue

                counts["articles_read"] += 1
                shard_counts["articles_read"] += 1

                title = _clean(article.get("MainTask"))
                if not title:
                    counts["skipped_missing_title"] += 1
                    shard_counts["skipped_missing_title"] += 1
                    continue

                procedure = _record_for_article(
                    article=article,
                    shard_name=shard.name,
                    row_index=row_index,
                    min_steps=min_steps,
                    category_normalization=category_normalization,
                )
                if procedure is None:
                    counts["skipped_too_few_steps"] += 1
                    shard_counts["skipped_too_few_steps"] += 1
                    continue

                procedure_id = procedure["id"]
                if procedure_id in seen_ids:
                    raise SystemExit(
                        f"Duplicate generated procedure id: {procedure_id}"
                    )
                seen_ids.add(procedure_id)

                _write_jsonl_record(procedures_file, procedure)
                aliases_applied = procedure["metadata"]["category_normalization"][
                    "aliases_applied"
                ]
                counts["category_alias_applications"] += len(aliases_applied)
                shard_counts["category_alias_applications"] += len(aliases_applied)
                if aliases_applied:
                    counts["records_with_category_aliases"] += 1
                    shard_counts["records_with_category_aliases"] += 1
                counts["procedures"] += 1
                step_tag = next(
                    (
                        tag.removeprefix("steps:")
                        for tag in procedure["tags"]
                        if str(tag).startswith("steps:")
                    ),
                    "0",
                )
                counts["total_steps"] += int(step_tag)
                shard_counts["procedures"] += 1

            raw_files.append(
                {
                    "articles_read": shard_counts["articles_read"],
                    "bytes": shard.stat().st_size,
                    "category_alias_applications": shard_counts[
                        "category_alias_applications"
                    ],
                    "name": shard.name,
                    "procedures": shard_counts["procedures"],
                    "records_with_category_aliases": shard_counts[
                        "records_with_category_aliases"
                    ],
                    "sha256": _sha256(shard),
                    "skipped_missing_title": shard_counts["skipped_missing_title"],
                    "skipped_too_few_steps": shard_counts["skipped_too_few_steps"],
                }
            )
            if limit is not None and counts["procedures"] >= limit:
                break

    manifest = {
        "builder_version": 1,
        "counts": counts,
        "deterministic": True,
        "normalization": {
            "assets": {
                "category_aliases": {
                    "alias_count": category_normalization["alias_count"],
                    "path": category_normalization["path"],
                    "schema_version": category_normalization["schema_version"],
                    "sha256": category_normalization["sha256"],
                    "unresolved_count": category_normalization["unresolved_count"],
                }
            }
        },
        "outputs": {
            "procedures": _output_info(procedures_path, counts["procedures"]),
        },
        "raw_files": raw_files,
        "settings": {
            "limit": limit,
            "min_steps": min_steps,
            "shard_pattern": shard_pattern,
        },
        "source": {
            "dataset": SOURCE_DATASET,
            "license": SOURCE_LICENSE,
            "name": SOURCE_NAME,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest"] = {
        "bytes": manifest_path.stat().st_size,
        "sha256": _sha256(manifest_path),
    }
    return manifest


def verify_expected_sha256(path: Path, expected_sha256: str | None) -> None:
    if not expected_sha256:
        return
    actual = _sha256(path)
    if actual != expected_sha256:
        raise SystemExit(
            "wikihow_procedures.jsonl checksum mismatch: "
            f"expected {expected_sha256}, got {actual}"
        )


def build_wikihow_procedures(
    input_dir: Path,
    output_dir: Path,
    category_aliases: Path = DEFAULT_CATEGORY_ALIASES,
    expected_procedures_sha256: str | None = EXPECTED_PROCEDURES_SHA256,
    expected_procedures_records: int | None = EXPECTED_PROCEDURES_RECORDS,
    shard_pattern: str = "wikiHow*.json",
    min_steps: int = 3,
    limit: int | None = None,
) -> dict[str, Any]:
    try:
        manifest = build(
            input_dir=input_dir,
            output_dir=output_dir,
            shard_pattern=shard_pattern,
            min_steps=min_steps,
            limit=limit,
            category_aliases=category_aliases,
        )
    except SystemExit as exc:
        raise CorpusBuildError(str(exc)) from exc

    procedures = manifest["outputs"]["procedures"]
    actual_sha256 = procedures["sha256"]
    actual_records = procedures["records"]
    sha256_matched = (
        expected_procedures_sha256 is None
        or actual_sha256 == expected_procedures_sha256
    )
    records_matched = (
        expected_procedures_records is None
        or actual_records == expected_procedures_records
    )
    if not sha256_matched:
        raise CorpusBuildError(
            "wikihow_procedures.jsonl checksum mismatch: "
            f"expected {expected_procedures_sha256}, got {actual_sha256}"
        )
    if not records_matched:
        raise CorpusBuildError(
            "wikihow_procedures.jsonl record count mismatch: "
            f"expected {expected_procedures_records}, got {actual_records}"
        )

    manifest.pop("manifest", None)
    manifest["verification"] = {
        "expected_procedures_records": expected_procedures_records,
        "expected_procedures_sha256": expected_procedures_sha256,
        "matched": sha256_matched and records_matched,
        "procedures_records": actual_records,
        "procedures_sha256": actual_sha256,
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest"] = {
        "bytes": manifest_path.stat().st_size,
        "sha256": _sha256(manifest_path),
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic MemFlow WikiHow procedure JSONL."
    )
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--shard-pattern", default="wikiHow*.json")
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category-aliases", default=str(DEFAULT_CATEGORY_ALIASES))
    parser.add_argument(
        "--expected-procedures-sha256",
        default=EXPECTED_PROCEDURES_SHA256,
        help="Expected SHA-256 for the full wikihow_procedures.jsonl build.",
    )
    parser.add_argument(
        "--skip-expected-verification",
        action="store_true",
        help="Skip full-corpus checksum and record-count verification.",
    )
    args = parser.parse_args()

    expected_sha256 = (
        None if args.skip_expected_verification else args.expected_procedures_sha256
    )
    expected_records = (
        None if args.skip_expected_verification else EXPECTED_PROCEDURES_RECORDS
    )
    if args.limit is not None and not args.skip_expected_verification:
        raise SystemExit("Use --skip-expected-verification when building with --limit")

    manifest = build_wikihow_procedures(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        category_aliases=Path(args.category_aliases),
        expected_procedures_sha256=expected_sha256,
        expected_procedures_records=expected_records,
        shard_pattern=args.shard_pattern,
        min_steps=args.min_steps,
        limit=args.limit,
    )
    counts = manifest["counts"]
    print("Build OK")
    print(f"Articles read: {counts['articles_read']}")
    print(f"Procedures: {counts['procedures']}")
    print(f"Total steps: {counts['total_steps']}")


if __name__ == "__main__":
    main()
