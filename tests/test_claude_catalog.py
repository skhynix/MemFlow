# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from memflow.claude_catalog import normalize_native_catalog_mode


def test_missing_config_uses_default_effective_mode():
    mode = normalize_native_catalog_mode({})

    assert mode.raw_mode is None
    assert mode.effective_mode == "hidden_or_minimized"
    assert mode.warnings == ()


def test_invalid_and_non_object_catalog_config_fall_back_with_warning():
    invalid_mode = normalize_native_catalog_mode(
        {"claude": {"native_catalog_mode": "nope"}}
    )
    non_object_claude = normalize_native_catalog_mode({"claude": "bad"})

    assert invalid_mode.effective_mode == "hidden_or_minimized"
    assert invalid_mode.warnings == ("invalid_native_catalog_mode",)
    assert non_object_claude.effective_mode == "hidden_or_minimized"
    assert non_object_claude.warnings == ("invalid_claude_config",)
