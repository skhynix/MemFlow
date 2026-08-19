# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import stat
import time
from pathlib import Path

import pytest

import memflow.claude_session_state as session_state_module
from memflow.claude_session_state import (
    MAX_STATE_FILE_BYTES,
    SESSION_STATE_SCHEMA_VERSION,
    ActivationUpdate,
    ClaudeSessionStateStore,
    StateOperationResult,
    hash_session_id,
    resolve_state_dir,
)

SESSION_ID = "raw-session-id-must-not-be-stored"
CONCURRENCY_TEST_LOCK_TIMEOUT_SECONDS = 1.0


def _activation(
    identity: str,
    fingerprint: str,
    *,
    name: str | None = None,
    emitted_chars: int = 100,
) -> ActivationUpdate:
    return ActivationUpdate(
        identity=identity,
        activation_fingerprint=fingerprint,
        name=name or identity.rsplit("/", 2)[-2],
        emitted_chars=emitted_chars,
    )


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


def _record_worker(
    state_dir: str,
    identity: str,
    fingerprint: str,
    barrier,
    results,
) -> None:
    barrier.wait(timeout=10)
    result = ClaudeSessionStateStore(
        state_dir,
        lock_timeout_seconds=CONCURRENCY_TEST_LOCK_TIMEOUT_SECONDS,
    ).record_activations(
        SESSION_ID,
        [_activation(identity, fingerprint)],
        expected_generation=0,
    )
    results.put(result.status)


def _advance_worker(state_dir: str, barrier, results) -> None:
    barrier.wait(timeout=10)
    result = ClaudeSessionStateStore(
        state_dir,
        lock_timeout_seconds=CONCURRENCY_TEST_LOCK_TIMEOUT_SECONDS,
    ).advance_context_generation(
        SESSION_ID,
        reason="future_compaction",
    )
    results.put(("advance", result.status))


def _racing_record_worker(state_dir: str, barrier, results) -> None:
    barrier.wait(timeout=10)
    result = ClaudeSessionStateStore(
        state_dir,
        lock_timeout_seconds=CONCURRENCY_TEST_LOCK_TIMEOUT_SECONDS,
    ).record_activations(
        SESSION_ID,
        [_activation("path:/repo/raced/SKILL.md", "raced")],
        expected_generation=0,
    )
    results.put(("record", result.status))


def _hold_lock_worker(lock_path: str, ready, release) -> None:
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        session_state_module.fcntl.flock(
            descriptor,
            session_state_module.fcntl.LOCK_EX,
        )
        ready.set()
        release.wait(timeout=10)
    finally:
        session_state_module.fcntl.flock(
            descriptor,
            session_state_module.fcntl.LOCK_UN,
        )
        os.close(descriptor)


def _join_processes(processes) -> None:
    for process in processes:
        process.join(timeout=15)
    for process in processes:
        assert not process.is_alive()
        assert process.exitcode == 0


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


def test_record_uses_private_paths_and_excludes_raw_session_data(tmp_path):
    state_parent = tmp_path / "new-parent"
    state_dir = state_parent / "nested" / "state"
    store = ClaudeSessionStateStore(state_dir)

    result = store.record_activations(
        SESSION_ID,
        [_activation("path:/repo/alpha/SKILL.md", "fingerprint-alpha")],
        expected_generation=0,
    )

    assert result.succeeded is True
    assert result.status == "recorded"
    state_path = store.state_path(SESSION_ID)
    state_text = state_path.read_text(encoding="utf-8")
    payload = json.loads(state_text)
    assert payload["schema_version"] == SESSION_STATE_SCHEMA_VERSION
    assert payload["session_id_hash"] == hash_session_id(SESSION_ID)
    assert SESSION_ID not in state_text
    assert "prompt" not in payload
    assert "body" not in payload
    assert stat.S_IMODE(state_parent.stat().st_mode) == 0o700
    assert stat.S_IMODE((state_parent / "nested").stat().st_mode) == 0o700
    assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.lock_path(SESSION_ID).stat().st_mode) == 0o600
    assert not list(state_dir.glob("*.tmp"))


def test_state_creation_preserves_preexisting_parent_permissions(tmp_path):
    existing_parent = tmp_path / "existing"
    existing_parent.mkdir(mode=0o755)
    existing_parent.chmod(0o755)
    state_dir = existing_parent / "new" / "state"
    store = ClaudeSessionStateStore(state_dir)

    result = store.record_activations(
        SESSION_ID,
        [_activation("path:/repo/alpha/SKILL.md", "alpha")],
        expected_generation=0,
    )

    assert result.status == "recorded"
    assert stat.S_IMODE(existing_parent.stat().st_mode) == 0o755
    assert stat.S_IMODE((existing_parent / "new").stat().st_mode) == 0o700
    assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700


def test_state_creation_accepts_concurrent_directory_creator(monkeypatch, tmp_path):
    state_dir = tmp_path / "state"
    store = ClaudeSessionStateStore(state_dir)
    original_is_dir = Path.is_dir
    original_stat = Path.stat
    raced = False

    def report_not_directory_after_concurrent_creation(path):
        nonlocal raced
        if path == state_dir and not raced:
            raced = True
            state_dir.mkdir(mode=0o700)
            return False
        return original_is_dir(path)

    def create_directory_after_missing_observation(path, *args, **kwargs):
        nonlocal raced
        if path == state_dir and not raced:
            raced = True
            state_dir.mkdir(mode=0o700)
            raise FileNotFoundError(state_dir)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "is_dir", report_not_directory_after_concurrent_creation)
    monkeypatch.setattr(Path, "stat", create_directory_after_missing_observation)

    result = store.record_activations(
        SESSION_ID,
        [_activation("path:/repo/alpha/SKILL.md", "alpha")],
        expected_generation=0,
    )

    assert raced is True
    assert result.status == "recorded"
    assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    "update",
    (
        ActivationUpdate(
            identity=1,  # type: ignore[arg-type]
            activation_fingerprint="fingerprint",
            name="alpha",
            emitted_chars=1,
        ),
        ActivationUpdate(
            identity="path:/repo/alpha/SKILL.md",
            activation_fingerprint=1,  # type: ignore[arg-type]
            name="alpha",
            emitted_chars=1,
        ),
        ActivationUpdate(
            identity="path:/repo/alpha/SKILL.md",
            activation_fingerprint="fingerprint",
            name="alpha",
            emitted_chars=True,
        ),
    ),
)
def test_invalid_activation_updates_fail_before_state_io(tmp_path, update):
    state_dir = tmp_path / "state"
    result = ClaudeSessionStateStore(state_dir).record_activations(
        SESSION_ID,
        [update],
        expected_generation=0,
    )

    assert result.status == "invalid_activations"
    assert result.succeeded is False
    assert not state_dir.exists()


def test_empty_activation_record_is_unchanged_without_state_io(tmp_path):
    state_dir = tmp_path / "cold-state"
    store = ClaudeSessionStateStore(state_dir)

    result = store.record_activations(
        SESSION_ID,
        [],
        expected_generation=0,
    )

    assert result.status == "unchanged"
    assert result.succeeded is True
    assert result.state is None
    assert not state_dir.exists()


@pytest.mark.parametrize(
    "invalidated_identities",
    (
        "path:/repo/alpha/SKILL.md",
        [""],
        [1],
    ),
)
def test_invalidated_identities_fail_before_state_io(
    tmp_path,
    invalidated_identities,
):
    state_dir = tmp_path / "cold-state"

    result = ClaudeSessionStateStore(state_dir).record_activations(
        SESSION_ID,
        [],
        expected_generation=0,
        invalidated_identities=invalidated_identities,
    )

    assert result.status == "invalid_invalidations"
    assert result.succeeded is False
    assert not state_dir.exists()


def test_raising_invalidation_iterable_fails_before_state_io(tmp_path):
    state_dir = tmp_path / "cold-state"

    class RaisingIterable:
        def __iter__(self):
            raise RuntimeError("invalidations unavailable")

    result = ClaudeSessionStateStore(state_dir).record_activations(
        SESSION_ID,
        [],
        expected_generation=0,
        invalidated_identities=RaisingIterable(),
    )

    assert result.status == "invalid_invalidations"
    assert result.succeeded is False
    assert not state_dir.exists()


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
    operation = store.record_activations(
        SESSION_ID,
        [_activation("path:/repo/new/SKILL.md", "new")],
        expected_generation=0,
    )

    assert loaded.available is False
    assert loaded.status == expected_status
    assert operation.succeeded is False
    assert operation.status == expected_status
    assert state_path.read_bytes() == original


def test_invalid_json_is_unavailable_and_not_overwritten(tmp_path):
    store = ClaudeSessionStateStore(tmp_path / "state")
    store.state_dir.mkdir(mode=0o700)
    state_path = store.state_path(SESSION_ID)
    state_path.write_text("{not-json", encoding="utf-8")
    original = state_path.read_bytes()

    loaded = store.load(SESSION_ID)
    operation = store.record_activations(
        SESSION_ID,
        [_activation("path:/repo/new/SKILL.md", "new")],
        expected_generation=0,
    )

    assert loaded.status == "invalid_json"
    assert loaded.available is False
    assert operation.status == "invalid_json"
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

    recorded = store.record_activations(
        SESSION_ID,
        [_activation("path:/repo/new/SKILL.md", "new")],
        expected_generation=0,
    )
    assert recorded.status == "invalid_json"
    assert recorded.succeeded is False
    assert recorded.warnings == ("claude_session_state_invalid_json",)
    assert state_path.read_bytes() == original

    advanced = store.advance_context_generation(
        SESSION_ID,
        reason="future_compaction",
    )
    assert advanced.status == "invalid_json"
    assert advanced.succeeded is False
    assert advanced.warnings == ("claude_session_state_invalid_json",)
    assert state_path.read_bytes() == original


def test_oversized_state_is_rejected_before_json_parsing_and_preserved(tmp_path):
    store = ClaudeSessionStateStore(tmp_path / "state")
    store.state_dir.mkdir(mode=0o700)
    state_path = store.state_path(SESSION_ID)
    state_path.write_bytes(b"{" + (b" " * MAX_STATE_FILE_BYTES))
    original_size = state_path.stat().st_size

    loaded = store.load(SESSION_ID)
    operation = store.record_activations(
        SESSION_ID,
        [_activation("path:/repo/new/SKILL.md", "new")],
        expected_generation=0,
    )

    assert original_size == MAX_STATE_FILE_BYTES + 1
    assert loaded.status == "oversized"
    assert loaded.available is False
    assert operation.status == "oversized"
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
    operation = store.record_activations(
        SESSION_ID,
        [_activation("path:/repo/new/SKILL.md", "new")],
        expected_generation=0,
    )

    assert loaded.status == "read_failed"
    assert loaded.available is False
    assert operation.status == "read_failed"
    assert os.stat(state_path).st_size == len(original)


def test_latest_fingerprint_replacement_preserves_other_activations(tmp_path):
    state_dir = tmp_path / "state"
    store = ClaudeSessionStateStore(state_dir)
    first = store.record_activations(
        SESSION_ID,
        [
            _activation("path:/repo/alpha/SKILL.md", "alpha-v1"),
            _activation("path:/repo/beta/SKILL.md", "beta-v1"),
        ],
        expected_generation=0,
    )
    second = store.record_activations(
        SESSION_ID,
        [_activation("path:/repo/alpha/SKILL.md", "alpha-v2")],
        expected_generation=0,
    )

    reloaded = ClaudeSessionStateStore(state_dir).load(SESSION_ID)

    assert first.status == "recorded"
    assert second.status == "recorded"
    assert reloaded.status == "ok"
    assert reloaded.state is not None
    assert (
        reloaded.state.activations["path:/repo/alpha/SKILL.md"].activation_fingerprint
        == "alpha-v2"
    )
    assert (
        reloaded.state.activations["path:/repo/beta/SKILL.md"].activation_fingerprint
        == "beta-v1"
    )


def test_record_atomically_invalidates_updates_and_preserves_unrelated_state(tmp_path):
    store = ClaudeSessionStateStore(tmp_path / "state")
    updated_identity = "path:/repo/updated/SKILL.md"
    invalidated_identity = "path:/repo/invalidated/SKILL.md"
    unrelated_identity = "path:/repo/unrelated/SKILL.md"
    seeded = store.record_activations(
        SESSION_ID,
        [
            _activation(updated_identity, "updated-v1"),
            _activation(invalidated_identity, "invalidated-v1"),
            _activation(unrelated_identity, "unrelated-v1"),
        ],
        expected_generation=0,
    )
    before = store.load(SESSION_ID)
    assert before.state is not None
    unrelated_before = before.state.activations[unrelated_identity]

    recorded = store.record_activations(
        SESSION_ID,
        [_activation(updated_identity, "updated-v2")],
        expected_generation=0,
        invalidated_identities=[invalidated_identity],
    )
    loaded = store.load(SESSION_ID)

    assert seeded.status == "recorded"
    assert recorded.status == "recorded"
    assert loaded.state is not None
    assert set(loaded.state.activations) == {updated_identity, unrelated_identity}
    assert (
        loaded.state.activations[updated_identity].activation_fingerprint
        == "updated-v2"
    )
    assert loaded.state.activations[unrelated_identity] == unrelated_before


def test_invalidation_only_record_removes_stored_activation(tmp_path):
    store = ClaudeSessionStateStore(tmp_path / "state")
    invalidated_identity = "path:/repo/invalidated/SKILL.md"
    unrelated_identity = "path:/repo/unrelated/SKILL.md"
    store.record_activations(
        SESSION_ID,
        [
            _activation(invalidated_identity, "invalidated"),
            _activation(unrelated_identity, "unrelated"),
        ],
        expected_generation=0,
    )

    invalidated = store.record_activations(
        SESSION_ID,
        [],
        expected_generation=0,
        invalidated_identities=[invalidated_identity],
    )
    loaded = store.load(SESSION_ID)

    assert invalidated.status == "recorded"
    assert loaded.state is not None
    assert set(loaded.state.activations) == {unrelated_identity}


def test_invalidation_wins_over_same_identity_update(tmp_path):
    store = ClaudeSessionStateStore(tmp_path / "state")
    identity = "path:/repo/overlap/SKILL.md"
    store.record_activations(
        SESSION_ID,
        [_activation(identity, "old")],
        expected_generation=0,
    )

    recorded = store.record_activations(
        SESSION_ID,
        [_activation(identity, "new")],
        expected_generation=0,
        invalidated_identities=[identity],
    )
    loaded = store.load(SESSION_ID)

    assert recorded.status == "recorded"
    assert loaded.state is not None
    assert identity not in loaded.state.activations


def test_combined_mutation_write_failure_preserves_all_previous_state(
    monkeypatch,
    tmp_path,
):
    store = ClaudeSessionStateStore(tmp_path / "state")
    updated_identity = "path:/repo/updated/SKILL.md"
    invalidated_identity = "path:/repo/invalidated/SKILL.md"
    store.record_activations(
        SESSION_ID,
        [
            _activation(updated_identity, "updated-v1"),
            _activation(invalidated_identity, "invalidated-v1"),
        ],
        expected_generation=0,
    )
    state_path = store.state_path(SESSION_ID)
    original = state_path.read_bytes()

    def fail_write(*_args, **_kwargs):
        raise OSError("write failed")

    monkeypatch.setattr(session_state_module, "_atomic_write_state", fail_write)

    failed = store.record_activations(
        SESSION_ID,
        [_activation(updated_identity, "updated-v2")],
        expected_generation=0,
        invalidated_identities=[invalidated_identity],
    )

    assert failed.status == "write_failed"
    assert failed.succeeded is False
    assert state_path.read_bytes() == original


def test_stale_generation_rejects_updates_and_invalidations_together(tmp_path):
    store = ClaudeSessionStateStore(tmp_path / "state")
    store.record_activations(
        SESSION_ID,
        [_activation("path:/repo/old/SKILL.md", "old")],
        expected_generation=0,
    )
    store.advance_context_generation(SESSION_ID, reason="future_compaction")
    current_identity = "path:/repo/current/SKILL.md"
    store.record_activations(
        SESSION_ID,
        [_activation(current_identity, "current")],
        expected_generation=1,
    )
    state_path = store.state_path(SESSION_ID)
    current_bytes = state_path.read_bytes()

    stale = store.record_activations(
        SESSION_ID,
        [_activation("path:/repo/stale/SKILL.md", "stale")],
        expected_generation=0,
        invalidated_identities=[current_identity],
    )
    loaded = store.load(SESSION_ID)

    assert stale.status == "stale_generation"
    assert stale.succeeded is False
    assert state_path.read_bytes() == current_bytes
    assert loaded.state is not None
    assert set(loaded.state.activations) == {current_identity}


def test_generation_advance_clears_state_and_rejects_stale_record(tmp_path):
    store = ClaudeSessionStateStore(tmp_path / "state")
    store.record_activations(
        SESSION_ID,
        [_activation("path:/repo/alpha/SKILL.md", "alpha")],
        expected_generation=0,
    )

    advanced = store.advance_context_generation(
        SESSION_ID,
        reason="future_compaction",
    )
    after_advance = store.state_path(SESSION_ID).read_bytes()
    stale = store.record_activations(
        SESSION_ID,
        [_activation("path:/repo/stale/SKILL.md", "stale")],
        expected_generation=0,
    )
    loaded = store.load(SESSION_ID)

    assert advanced.status == "advanced"
    assert advanced.state is not None
    assert advanced.state.context_generation == 1
    assert advanced.state.last_reset_reason == "future_compaction"
    assert advanced.state.activations == {}
    assert stale.status == "stale_generation"
    assert stale.succeeded is False
    assert store.state_path(SESSION_ID).read_bytes() == after_advance
    assert loaded.state == advanced.state


@pytest.mark.parametrize("failure_point", ("temporary", "flush", "replace"))
def test_atomic_write_failure_preserves_last_valid_state(
    monkeypatch,
    tmp_path,
    failure_point,
):
    state_dir = tmp_path / "state"
    store = ClaudeSessionStateStore(state_dir)
    seeded = store.record_activations(
        SESSION_ID,
        [_activation("path:/repo/alpha/SKILL.md", "alpha-v1")],
        expected_generation=0,
    )
    assert seeded.status == "recorded"
    state_path = store.state_path(SESSION_ID)
    original = state_path.read_bytes()

    def fail(*args, **kwargs):
        del args, kwargs
        raise OSError(f"injected {failure_point} failure")

    if failure_point == "temporary":
        monkeypatch.setattr(session_state_module.tempfile, "mkstemp", fail)
    elif failure_point == "flush":
        monkeypatch.setattr(session_state_module, "_flush_and_sync", fail)
    else:
        monkeypatch.setattr(session_state_module.os, "replace", fail)

    operation = store.record_activations(
        SESSION_ID,
        [_activation("path:/repo/alpha/SKILL.md", "alpha-v2")],
        expected_generation=0,
    )
    reloaded = ClaudeSessionStateStore(state_dir).load(SESSION_ID)

    assert operation.status == "write_failed"
    assert operation.succeeded is False
    assert state_path.read_bytes() == original
    assert reloaded.state is not None
    assert (
        reloaded.state.activations["path:/repo/alpha/SKILL.md"].activation_fingerprint
        == "alpha-v1"
    )
    assert not list(state_dir.glob("*.tmp"))


def test_oversized_serialized_update_preserves_last_valid_state(tmp_path):
    state_dir = tmp_path / "state"
    store = ClaudeSessionStateStore(state_dir)
    seeded = store.record_activations(
        SESSION_ID,
        [_activation("path:/repo/alpha/SKILL.md", "alpha-v1")],
        expected_generation=0,
    )
    assert seeded.status == "recorded"
    state_path = store.state_path(SESSION_ID)
    original = state_path.read_bytes()

    operation = store.record_activations(
        SESSION_ID,
        [
            _activation(
                "path:/repo/oversized/SKILL.md",
                "oversized",
                name="x" * MAX_STATE_FILE_BYTES,
            )
        ],
        expected_generation=0,
    )

    assert operation.status == "write_failed"
    assert state_path.read_bytes() == original
    assert not list(state_dir.glob("*.tmp"))


def test_unsupported_lock_platform_fails_before_state_io(monkeypatch, tmp_path):
    state_dir = tmp_path / "must-not-exist"
    store = ClaudeSessionStateStore(state_dir)
    monkeypatch.setattr(session_state_module, "fcntl", None)

    loaded = store.load(SESSION_ID)
    operation = store.record_activations(
        SESSION_ID,
        [_activation("path:/repo/alpha/SKILL.md", "alpha")],
        expected_generation=0,
    )

    assert loaded.status == "unsupported_lock_platform"
    assert loaded.available is False
    assert operation.status == "unsupported_lock_platform"
    assert operation.succeeded is False
    assert not state_dir.exists()


@pytest.mark.skipif(
    session_state_module.fcntl is None,
    reason="POSIX flock is unavailable",
)
def test_concurrent_records_preserve_both_identities(tmp_path):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    results = context.Queue()
    state_dir = str(tmp_path / "state")
    processes = [
        context.Process(
            target=_record_worker,
            args=(
                state_dir,
                "path:/repo/alpha/SKILL.md",
                "alpha",
                barrier,
                results,
            ),
        ),
        context.Process(
            target=_record_worker,
            args=(
                state_dir,
                "path:/repo/beta/SKILL.md",
                "beta",
                barrier,
                results,
            ),
        ),
    ]
    for process in processes:
        process.start()
    barrier.wait(timeout=10)
    _join_processes(processes)

    statuses = {results.get(timeout=2), results.get(timeout=2)}
    loaded = ClaudeSessionStateStore(state_dir).load(SESSION_ID)

    assert statuses == {"recorded"}
    assert loaded.state is not None
    assert set(loaded.state.activations) == {
        "path:/repo/alpha/SKILL.md",
        "path:/repo/beta/SKILL.md",
    }


@pytest.mark.skipif(
    session_state_module.fcntl is None,
    reason="POSIX flock is unavailable",
)
def test_concurrent_record_and_advance_cannot_roll_back_generation(tmp_path):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    results = context.Queue()
    state_dir = str(tmp_path / "state")
    processes = [
        context.Process(
            target=_racing_record_worker,
            args=(state_dir, barrier, results),
        ),
        context.Process(
            target=_advance_worker,
            args=(state_dir, barrier, results),
        ),
    ]
    for process in processes:
        process.start()
    barrier.wait(timeout=10)
    _join_processes(processes)

    statuses = dict((results.get(timeout=2), results.get(timeout=2)))
    loaded = ClaudeSessionStateStore(state_dir).load(SESSION_ID)

    assert statuses["advance"] == "advanced"
    assert statuses["record"] in {"recorded", "stale_generation"}
    assert loaded.state is not None
    assert loaded.state.context_generation == 1
    assert loaded.state.activations == {}


@pytest.mark.skipif(
    session_state_module.fcntl is None,
    reason="POSIX flock is unavailable",
)
def test_held_lock_times_out_with_typed_failure(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o700)
    store = ClaudeSessionStateStore(state_dir)
    holder = context.Process(
        target=_hold_lock_worker,
        args=(str(store.lock_path(SESSION_ID)), ready, release),
    )
    holder.start()
    try:
        assert ready.wait(timeout=10)
        started = time.monotonic()
        result = store.record_activations(
            SESSION_ID,
            [_activation("path:/repo/alpha/SKILL.md", "alpha")],
            expected_generation=0,
        )
        elapsed = time.monotonic() - started
    finally:
        release.set()
        _join_processes([holder])

    assert isinstance(result, StateOperationResult)
    assert result.status == "lock_timeout"
    assert result.succeeded is False
    assert 0.08 <= elapsed < 0.5
    assert not store.state_path(SESSION_ID).exists()
