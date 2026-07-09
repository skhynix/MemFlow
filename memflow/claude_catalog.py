# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code native skill catalog settings management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_NATIVE_CATALOG_MODE = "hidden_or_minimized"
SUPPORTED_NATIVE_CATALOG_MODES = {
    "visible",
    "hidden_or_minimized",
    "disabled",
}


@dataclass(frozen=True)
class CatalogModeResolution:
    raw_mode: Any
    effective_mode: str
    warnings: tuple[str, ...] = ()


def normalize_native_catalog_mode(config: Any) -> CatalogModeResolution:
    """Return the requested and effective Claude native catalog mode."""
    if not isinstance(config, dict):
        return CatalogModeResolution(
            raw_mode=None,
            effective_mode=DEFAULT_NATIVE_CATALOG_MODE,
            warnings=("invalid_config",),
        )

    if "claude" not in config:
        return CatalogModeResolution(
            raw_mode=None,
            effective_mode=DEFAULT_NATIVE_CATALOG_MODE,
        )

    claude_config = config.get("claude")
    if not isinstance(claude_config, dict):
        return CatalogModeResolution(
            raw_mode=None,
            effective_mode=DEFAULT_NATIVE_CATALOG_MODE,
            warnings=("invalid_claude_config",),
        )

    raw_mode = claude_config.get("native_catalog_mode")
    if raw_mode is None:
        return CatalogModeResolution(
            raw_mode=None,
            effective_mode=DEFAULT_NATIVE_CATALOG_MODE,
        )
    if isinstance(raw_mode, str) and raw_mode in SUPPORTED_NATIVE_CATALOG_MODES:
        return CatalogModeResolution(raw_mode=raw_mode, effective_mode=raw_mode)
    return CatalogModeResolution(
        raw_mode=raw_mode,
        effective_mode=DEFAULT_NATIVE_CATALOG_MODE,
        warnings=("invalid_native_catalog_mode",),
    )


def _mode_resolution_from_metadata(
    config: dict[str, Any] | None,
) -> CatalogModeResolution | None:
    if not isinstance(config, dict):
        return None
    metadata = config.get("_memflow_catalog_mode")
    if not isinstance(metadata, dict):
        return None
    effective_mode = metadata.get("effective")
    if (
        not isinstance(effective_mode, str)
        or effective_mode not in SUPPORTED_NATIVE_CATALOG_MODES
    ):
        return None

    raw_warnings = metadata.get("warnings", ())
    if not isinstance(raw_warnings, (list, tuple)):
        raw_warnings = (raw_warnings,)
    warnings = tuple(str(item) for item in raw_warnings if item)
    return CatalogModeResolution(
        raw_mode=metadata.get("raw"),
        effective_mode=effective_mode,
        warnings=warnings,
    )
