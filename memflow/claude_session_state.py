# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort, generation-aware state for Claude skill activations."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - exercised through the unsupported-platform test
    import fcntl
except ImportError:  # pragma: no cover - depends on the host platform
    fcntl = None  # type: ignore[assignment]


SESSION_STATE_SCHEMA_VERSION = "memflow.claude_session_state.v1"
MAX_STATE_FILE_BYTES = 1_048_576


@dataclass(frozen=True)
class StoredActivation:
    """The latest emitted fingerprint for one stable activation identity."""

    activation_fingerprint: str
    name: str
    emitted_chars: int
    emitted_at: str


@dataclass(frozen=True)
class ClaudeSessionState:
    """Validated state for one hashed Claude session identifier."""

    session_id_hash: str
    context_generation: int
    last_reset_reason: str | None
    updated_at: str | None
    activations: dict[str, StoredActivation]


@dataclass(frozen=True)
class StateLoadResult:
    """Typed result of loading session state without raising into the hook."""

    status: str
    state: ClaudeSessionState | None = None
    warnings: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        """Whether the state can safely participate in a dedupe decision."""
        return self.state is not None


def hash_session_id(session_id: str) -> str:
    """Return the filename-safe SHA-256 digest for a non-empty session ID."""
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def resolve_state_dir(
    state_dir: str | Path,
    *,
    config_path: str | Path,
) -> Path:
    """Resolve state_dir lexically against the hook config's parent.

    This intentionally avoids ``Path.resolve`` so configuration normalization
    never stats or otherwise accesses the session-state path.
    """
    raw_state_dir = os.fspath(state_dir)
    if not raw_state_dir.strip():
        raise ValueError("state_dir must be a non-empty path")

    expanded_state_dir = Path(os.path.expanduser(raw_state_dir))
    if not expanded_state_dir.is_absolute():
        raw_config_path = os.fspath(config_path)
        expanded_config_path = Path(os.path.expanduser(raw_config_path))
        expanded_state_dir = expanded_config_path.parent / expanded_state_dir

    normalized = os.path.abspath(os.path.normpath(os.fspath(expanded_state_dir)))
    return Path(normalized)


class ClaudeSessionStateStore:
    """Store one privacy-preserving JSON state document per Claude session."""

    def __init__(self, state_dir: str | Path) -> None:
        raw_state_dir = os.fspath(state_dir)
        if not raw_state_dir.strip():
            raise ValueError("state_dir must be a non-empty path")
        normalized = os.path.abspath(
            os.path.normpath(os.path.expanduser(raw_state_dir))
        )
        self.state_dir = Path(normalized)

    def state_path(self, session_id: str) -> Path:
        """Return the hashed JSON state path without touching the filesystem."""
        return self.state_dir / f"{hash_session_id(session_id)}.json"

    def load(self, session_id: str) -> StateLoadResult:
        """Load and validate state for a prompt-time dedupe decision."""
        session_id_hash, failure = _validated_session_hash(session_id)
        if failure is not None:
            return StateLoadResult(failure, warnings=_warning(failure))
        if fcntl is None:
            return StateLoadResult(
                "unsupported_lock_platform",
                warnings=_warning("unsupported_lock_platform"),
            )

        return _read_state_file(
            self.state_dir / f"{session_id_hash}.json",
            expected_session_hash=session_id_hash,
        )


def _validated_session_hash(session_id: str) -> tuple[str, str | None]:
    try:
        return hash_session_id(session_id), None
    except (TypeError, ValueError):
        return "", "invalid_session_id"


def _read_state_file(
    path: Path,
    *,
    expected_session_hash: str,
) -> StateLoadResult:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_STATE_FILE_BYTES + 1)
    except FileNotFoundError:
        return StateLoadResult(
            "missing",
            state=_empty_state(expected_session_hash),
        )
    except OSError:
        return StateLoadResult("read_failed", warnings=_warning("read_failed"))

    if len(raw) > MAX_STATE_FILE_BYTES:
        return StateLoadResult("oversized", warnings=_warning("oversized"))
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return StateLoadResult("invalid_json", warnings=_warning("invalid_json"))
    try:
        payload = json.loads(decoded)
    except (ValueError, RecursionError):
        return StateLoadResult("invalid_json", warnings=_warning("invalid_json"))
    return _validate_state_payload(payload, expected_session_hash=expected_session_hash)


def _validate_state_payload(
    payload: object,
    *,
    expected_session_hash: str,
) -> StateLoadResult:
    if not isinstance(payload, dict):
        return StateLoadResult("invalid_state", warnings=_warning("invalid_state"))
    if payload.get("schema_version") != SESSION_STATE_SCHEMA_VERSION:
        return StateLoadResult(
            "schema_mismatch",
            warnings=_warning("schema_mismatch"),
        )
    if payload.get("session_id_hash") != expected_session_hash:
        return StateLoadResult(
            "session_mismatch",
            warnings=_warning("session_mismatch"),
        )

    generation = payload.get("context_generation")
    last_reset_reason = payload.get("last_reset_reason")
    updated_at = payload.get("updated_at")
    raw_activations = payload.get("activations")
    if not _is_non_negative_int(generation):
        return StateLoadResult("invalid_state", warnings=_warning("invalid_state"))
    if last_reset_reason is not None and not isinstance(last_reset_reason, str):
        return StateLoadResult("invalid_state", warnings=_warning("invalid_state"))
    if not isinstance(updated_at, str) or not updated_at:
        return StateLoadResult("invalid_state", warnings=_warning("invalid_state"))
    if not isinstance(raw_activations, dict):
        return StateLoadResult("invalid_state", warnings=_warning("invalid_state"))

    activations: dict[str, StoredActivation] = {}
    for identity, raw_activation in raw_activations.items():
        if not isinstance(identity, str) or not identity:
            return StateLoadResult("invalid_state", warnings=_warning("invalid_state"))
        if not isinstance(raw_activation, dict):
            return StateLoadResult("invalid_state", warnings=_warning("invalid_state"))

        fingerprint = raw_activation.get("activation_fingerprint")
        name = raw_activation.get("name")
        emitted_chars = raw_activation.get("emitted_chars")
        emitted_at = raw_activation.get("emitted_at")
        if not isinstance(fingerprint, str) or not fingerprint:
            return StateLoadResult("invalid_state", warnings=_warning("invalid_state"))
        if not isinstance(name, str):
            return StateLoadResult("invalid_state", warnings=_warning("invalid_state"))
        if not _is_non_negative_int(emitted_chars):
            return StateLoadResult("invalid_state", warnings=_warning("invalid_state"))
        if not isinstance(emitted_at, str) or not emitted_at:
            return StateLoadResult("invalid_state", warnings=_warning("invalid_state"))
        activations[identity] = StoredActivation(
            activation_fingerprint=fingerprint,
            name=name,
            emitted_chars=emitted_chars,
            emitted_at=emitted_at,
        )

    return StateLoadResult(
        "ok",
        state=ClaudeSessionState(
            session_id_hash=expected_session_hash,
            context_generation=generation,
            last_reset_reason=last_reset_reason,
            updated_at=updated_at,
            activations=activations,
        ),
    )


def _empty_state(session_id_hash: str) -> ClaudeSessionState:
    return ClaudeSessionState(
        session_id_hash=session_id_hash,
        context_generation=0,
        last_reset_reason=None,
        updated_at=None,
        activations={},
    )


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _warning(status: str) -> tuple[str, ...]:
    return (f"claude_session_state_{status}",)


__all__ = [
    "ClaudeSessionState",
    "ClaudeSessionStateStore",
    "MAX_STATE_FILE_BYTES",
    "SESSION_STATE_SCHEMA_VERSION",
    "StateLoadResult",
    "StoredActivation",
    "hash_session_id",
    "resolve_state_dir",
]
