#!/usr/bin/env python3
# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Download the SkillReason-Bench dataset from HuggingFace.

Mirrors the files needed for the SkillReason-Bench harness into
``data/skillreason-bench/``:

  - data/queries.jsonl   (3,729 implicit-request queries, gold in positive_ids)
  - data/skills.jsonl    (61,228 candidate skills, single unsharded JSONL)
  - data/stats.json      (corpus stats + sha256 checksums of both JSONL files)

After downloading, the sha256 of each JSONL file is verified against
``stats.json`` so a corrupted or partially transferred corpus fails loudly
before an expensive seeding run.

Usage:
    uv run benchmark/skill_reason_bench/download_data.py
    uv run benchmark/skill_reason_bench/download_data.py --force

Uses ``huggingface_hub`` when installed (``uv sync --extra benchmark``);
any hub download failure — package missing, network down, 401, hub API
error — falls back to a plain HTTPS ``resolve/main/<file>`` download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = "donghongjiang/skillreason-bench"
DEFAULT_OUT = SCRIPT_DIR / "data" / "skillreason-bench"

DATA_FILES = ["data/queries.jsonl", "data/skills.jsonl"]
META_FILES = ["data/stats.json"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the SkillReason-Bench dataset from HuggingFace."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the sha256 cross-check against data/stats.json.",
    )
    return parser.parse_args()


def _download_via_hub(repo: str, filename: str, out_path: Path, force: bool) -> bool:
    """Try huggingface_hub; return True on success, False to use HTTPS fallback.

    Returns False both when the package is unavailable and when the hub
    download itself fails (network, 401, hub API error), so a hub-enabled
    environment retries over plain HTTPS instead of aborting the run.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"  exists: {out_path}")
        return True
    print(f"  fetching {repo}:{filename} -> {out_path}")
    try:
        cached = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            local_dir=str(out_path.parent),
        )
    except Exception as exc:
        print(f"  huggingface_hub failed ({exc!r}); falling back to HTTPS")
        return False
    # hf_hub_download writes to <parent>/<filename>; ensure it is at out_path.
    cached_path = Path(cached)
    if cached_path.resolve() != out_path.resolve() and not out_path.exists():
        out_path.write_bytes(cached_path.read_bytes())
    return True


def _download_via_https(repo: str, filename: str, out_path: Path, force: bool) -> None:
    """Fallback: download via the HF resolve URL using urllib."""
    import urllib.request

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"  exists: {out_path}")
        return
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
    print(f"  fetching {url} -> {out_path}")
    urllib.request.urlretrieve(url, str(out_path))


def download_file(filename: str, out_dir: Path, force: bool) -> None:
    out_path = out_dir / filename
    if _download_via_hub(REPO, filename, out_path, force):
        return
    _download_via_https(REPO, filename, out_path, force)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(out_dir: Path) -> None:
    """Verify each JSONL file against the sha256 recorded in stats.json.

    Raises SystemExit on mismatch so a corrupted corpus never reaches seeding.
    """
    stats_path = out_dir / "data" / "stats.json"
    if not stats_path.exists():
        raise SystemExit(f"Cannot verify checksums: {stats_path} is missing")
    expected = json.loads(stats_path.read_text(encoding="utf-8")).get("sha256", {})
    if not expected:
        raise SystemExit(f"No sha256 map found in {stats_path}")
    for name, want in sorted(expected.items()):
        path = out_dir / "data" / name
        if not path.exists():
            raise SystemExit(f"Checksum target missing: {path}")
        got = _sha256(path)
        if got != want:
            raise SystemExit(
                f"sha256 mismatch for {name}: expected {want}, got {got}. "
                "Re-download with --force."
            )
        print(f"  sha256 OK: {name}")


def verify_data_file_checksum(path: Path, skip_verify: bool = False) -> None:
    """Gate a run on the sha256 recorded in the ``stats.json`` beside ``path``.

    ``verify_checksums`` only guards downloads made by this script (and only
    without ``--no-verify``); data placed by other means (manual copy,
    interrupted transfer) would otherwise seed or score unverified, so the
    runners re-check the file against stats.json before doing expensive work.
    A directory (smoke subset) or a missing stats.json has nothing to check
    against and says so instead of failing.
    """
    if skip_verify:
        print(f"Skipping checksum verification (--skip-verify): {path}")
        return
    if path.is_dir():
        print(f"  {path} is a directory; skipping checksum verification")
        return
    stats_path = path.parent / "stats.json"
    if not stats_path.exists():
        print(f"  no stats.json beside {path}; skipping checksum verification")
        return
    expected = json.loads(stats_path.read_text(encoding="utf-8")).get("sha256", {})
    want = expected.get(path.name)
    if want is None:
        raise SystemExit(
            f"{stats_path} has no sha256 entry for {path.name}; "
            "refusing to run against an unknown data file."
        )
    got = _sha256(path)
    if got != want:
        raise SystemExit(
            f"sha256 mismatch for {path.name}: expected {want}, got {got}. "
            "The file on disk does not match stats.json — re-download with --force."
        )
    print(f"  sha256 OK: {path.name}")


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO} -> {out_dir}")

    for name in META_FILES + DATA_FILES:
        download_file(name, out_dir, args.force)

    if args.no_verify:
        print("\nSkipping checksum verification (--no-verify).")
    else:
        print("\nVerifying checksums against data/stats.json:")
        verify_checksums(out_dir)

    print("\nDone. Layout:")
    for name in META_FILES + DATA_FILES:
        p = out_dir / name
        print(f"  {p} {'(present)' if p.exists() else '(MISSING)'}")


if __name__ == "__main__":
    main()
