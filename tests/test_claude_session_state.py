# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import memflow.claude_session_state as session_state_module
from memflow.claude_session_state import (
    MAX_STATE_FILE_BYTES,
    SESSION_STATE_SCHEMA_VERSION,
    ClaudeSessionStateStore,
    hash_session_id,
    resolve_state_dir,
)

SESSION_ID = "raw-session-id-must-not-be-stored"


def _valid_payload(session_id: str = SESSION_ID) -> dict:
    return {
        "schema_version": SESSION_STATE_SCHEMA_VERSION,
        "session_id_hash": hash_session_id(session_id),
        "context_generation": 0,
        "last_reset_reason": None,
        "updated_at": "2026-07-22T12:00:00Z",
        "activations": {
            "path:/repo/alpha/SKILL.md": {
                "activation_fingerprint": "fingerprint-alpha",
                "name": "alpha",
                "emitted_chars": 123,
                "emitted_at": "2026-07-22T11:59:00Z",
            }
        },
    }


def _invalid_payload(variant: str) -> object:
    payload = _valid_payload()
    if variant == "root_type":
        return []
    if variant == "schema":
        payload["schema_version"] = "memflow.claude_session_state.v999"
    elif variant == "session_hash":
        payload["session_id_hash"] = hashlib.sha256(b"another-session").hexdigest()
    elif variant == "negative_generation":
        payload["context_generation"] = -1
    elif variant == "boolean_generation":
        payload["context_generation"] = True
    elif variant == "reset_reason_type":
        payload["last_reset_reason"] = 42
    elif variant == "updated_at_type":
        payload["updated_at"] = None
    elif variant == "activation_mapping":
        payload["activations"] = []
    elif variant == "activation_entry":
        payload["activations"] = {"path:/repo/alpha/SKILL.md": "bad"}
    elif variant == "activation_identity":
        payload["activations"] = {"": next(iter(payload["activations"].values()))}
    elif variant == "activation_fingerprint":
        next(iter(payload["activations"].values()))["activation_fingerprint"] = None
    elif variant == "activation_name":
        next(iter(payload["activations"].values()))["name"] = []
    elif variant == "activation_chars":
        next(iter(payload["activations"].values()))["emitted_chars"] = True
    elif variant == "activation_timestamp":
        next(iter(payload["activations"].values()))["emitted_at"] = None
    else:  # pragma: no cover - protects the test table itself
        raise AssertionError(f"unknown validation variant: {variant}")
    return payload


def test_session_hash_path_resolution_and_missing_state_are_side_effect_free(
    monkeypatch,
    tmp_path,
):
    expected_hash = hashlib.sha256(SESSION_ID.encode("utf-8")).hexdigest()
    config_path = tmp_path / ".memflow" / "claude-hook.json"

    def fail_if_resolved(*args, **kwargs):
        del args, kwargs
        raise AssertionError("state path normalization must be lexical")

    monkeypatch.setattr(Path, "resolve", fail_if_resolved)
    state_dir = resolve_state_dir(
        "nested/../claude-sessions",
        config_path=config_path,
    )
    store = ClaudeSessionStateStore(state_dir)

    assert hash_session_id(SESSION_ID) == expected_hash
    assert state_dir == config_path.parent / "claude-sessions"
    assert store.state_path(SESSION_ID) == state_dir / f"{expected_hash}.json"
    assert SESSION_ID not in store.state_path(SESSION_ID).name

    loaded = store.load(SESSION_ID)

    assert loaded.status == "missing"
    assert loaded.available is True
    assert loaded.state is not None
    assert loaded.state.context_generation == 0
    assert loaded.state.activations == {}
    assert not state_dir.exists()


def test_state_dir_home_expansion_is_lexical_and_side_effect_free(
    monkeypatch,
    tmp_path,
):
    fake_home = tmp_path / "uncreated-home"

    def fail_if_resolved(*args, **kwargs):
        del args, kwargs
        raise AssertionError("state path normalization must remain lexical")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "resolve", fail_if_resolved)

    state_dir = resolve_state_dir(
        "~/nested/../claude-sessions",
        config_path=tmp_path / "ignored" / "claude-hook.json",
    )

    assert state_dir == fake_home / "claude-sessions"
    assert not fake_home.exists()


@pytest.mark.parametrize(
    ("variant", "expected_status"),
    (
        ("root_type", "invalid_state"),
        ("schema", "schema_mismatch"),
        ("session_hash", "session_mismatch"),
        ("negative_generation", "invalid_state"),
        ("boolean_generation", "invalid_state"),
        ("reset_reason_type", "invalid_state"),
        ("updated_at_type", "invalid_state"),
        ("activation_mapping", "invalid_state"),
        ("activation_entry", "invalid_state"),
        ("activation_identity", "invalid_state"),
        ("activation_fingerprint", "invalid_state"),
        ("activation_name", "invalid_state"),
        ("activation_chars", "invalid_state"),
        ("activation_timestamp", "invalid_state"),
    ),
)
def test_invalid_state_is_unavailable_and_not_overwritten(
    tmp_path,
    variant,
    expected_status,
):
    store = ClaudeSessionStateStore(tmp_path / "state")
    store.state_dir.mkdir(mode=0o700)
    state_path = store.state_path(SESSION_ID)
    state_path.write_text(json.dumps(_invalid_payload(variant)), encoding="utf-8")
    original = state_path.read_bytes()

    loaded = store.load(SESSION_ID)

    assert loaded.available is False
    assert loaded.status == expected_status
    assert state_path.read_bytes() == original


def test_invalid_json_is_unavailable_and_not_overwritten(tmp_path):
    store = ClaudeSessionStateStore(tmp_path / "state")
    store.state_dir.mkdir(mode=0o700)
    state_path = store.state_path(SESSION_ID)
    state_path.write_text("{not-json", encoding="utf-8")
    original = state_path.read_bytes()

    loaded = store.load(SESSION_ID)

    assert loaded.status == "invalid_json"
    assert loaded.available is False
    assert state_path.read_bytes() == original


def test_json_integer_limit_failures_are_typed_and_preserve_state(
    monkeypatch,
    tmp_path,
):
    store = ClaudeSessionStateStore(tmp_path / "state")
    store.state_dir.mkdir(mode=0o700)
    state_path = store.state_path(SESSION_ID)
    serialized = json.dumps(_valid_payload(), separators=(",", ":"))
    original = serialized.replace(
        '"context_generation":0',
        f'"context_generation":{"9" * 5_000}',
    ).encode("utf-8")
    assert len(original) < MAX_STATE_FILE_BYTES
    state_path.write_bytes(original)

    original_json_loads = json.loads
    decoded_original = original.decode("utf-8")
    try:
        original_json_loads(decoded_original)
    except ValueError:
        pass
    else:

        def emulate_runtime_integer_limit(payload, *args, **kwargs):
            if payload == decoded_original:
                raise ValueError("integer string conversion limit exceeded")
            return original_json_loads(payload, *args, **kwargs)

        monkeypatch.setattr(
            session_state_module.json,
            "loads",
            emulate_runtime_integer_limit,
        )

    loaded = store.load(SESSION_ID)
    assert loaded.status == "invalid_json"
    assert loaded.available is False
    assert loaded.warnings == ("claude_session_state_invalid_json",)
    assert state_path.read_bytes() == original


def test_oversized_state_is_rejected_before_json_parsing_and_preserved(tmp_path):
    store = ClaudeSessionStateStore(tmp_path / "state")
    store.state_dir.mkdir(mode=0o700)
    state_path = store.state_path(SESSION_ID)
    state_path.write_bytes(b"{" + (b" " * MAX_STATE_FILE_BYTES))
    original_size = state_path.stat().st_size

    loaded = store.load(SESSION_ID)

    assert original_size == MAX_STATE_FILE_BYTES + 1
    assert loaded.status == "oversized"
    assert loaded.available is False
    assert state_path.stat().st_size == original_size


def test_unreadable_state_returns_typed_failure_without_overwrite(
    monkeypatch,
    tmp_path,
):
    store = ClaudeSessionStateStore(tmp_path / "state")
    store.state_dir.mkdir(mode=0o700)
    state_path = store.state_path(SESSION_ID)
    state_path.write_text(json.dumps(_valid_payload()), encoding="utf-8")
    original = state_path.read_bytes()
    original_open = Path.open

    def fail_state_open(path, *args, **kwargs):
        if path == state_path:
            raise PermissionError("unreadable state")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_state_open)

    loaded = store.load(SESSION_ID)

    assert loaded.status == "read_failed"
    assert loaded.available is False
    assert state_path.stat().st_size == len(original)


def test_unsupported_lock_platform_load_is_unavailable_without_state_io(
    monkeypatch,
    tmp_path,
):
    state_dir = tmp_path / "must-not-exist"
    store = ClaudeSessionStateStore(state_dir)
    monkeypatch.setattr(session_state_module, "fcntl", None)

    loaded = store.load(SESSION_ID)

    assert loaded.status == "unsupported_lock_platform"
    assert loaded.available is False
    assert not state_dir.exists()
