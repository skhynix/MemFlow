# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort, generation-aware state for Claude skill activations."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import stat
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

try:  # pragma: no cover - exercised through the unsupported-platform test
    import fcntl
except ImportError:  # pragma: no cover - depends on the host platform
    fcntl = None  # type: ignore[assignment]


SESSION_STATE_SCHEMA_VERSION = "memflow.claude_session_state.v1"
MAX_STATE_FILE_BYTES = 1_048_576
DEFAULT_LOCK_TIMEOUT_SECONDS = 0.1


@dataclass(frozen=True)
class ActivationUpdate:
    """One rendered activation prepared for hook output."""

    identity: str
    activation_fingerprint: str
    name: str
    emitted_chars: int


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


@dataclass(frozen=True)
class StateOperationResult:
    """Typed result of a serialized state mutation."""

    status: str
    state: ClaudeSessionState | None = None
    warnings: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        """Whether the requested mutation completed or required no write."""
        return self.status in {"recorded", "advanced", "unchanged"}


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

    def __init__(
        self,
        state_dir: str | Path,
        *,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    ) -> None:
        raw_state_dir = os.fspath(state_dir)
        if not raw_state_dir.strip():
            raise ValueError("state_dir must be a non-empty path")
        try:
            normalized_timeout = float(lock_timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("lock_timeout_seconds must be non-negative") from exc
        if not math.isfinite(normalized_timeout) or normalized_timeout < 0:
            raise ValueError("lock_timeout_seconds must be non-negative")

        normalized = os.path.abspath(
            os.path.normpath(os.path.expanduser(raw_state_dir))
        )
        self.state_dir = Path(normalized)
        self.lock_timeout_seconds = normalized_timeout

    def state_path(self, session_id: str) -> Path:
        """Return the hashed JSON state path without touching the filesystem."""
        return self.state_dir / f"{hash_session_id(session_id)}.json"

    def lock_path(self, session_id: str) -> Path:
        """Return the stable per-session lock path."""
        return self.state_dir / f"{hash_session_id(session_id)}.lock"

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

    def record_activations(
        self,
        session_id: str,
        activations: Iterable[ActivationUpdate],
        *,
        expected_generation: int,
        invalidated_identities: Iterable[str] = (),
    ) -> StateOperationResult:
        """Merge updates and invalidations if the observed generation is current.

        Invalidations win when the same identity is present in both inputs.
        """
        if not _is_non_negative_int(expected_generation):
            return StateOperationResult(
                "invalid_expected_generation",
                warnings=_warning("invalid_expected_generation"),
            )

        updates, validation_status = _validated_updates(activations)
        if validation_status is not None:
            return StateOperationResult(
                validation_status,
                warnings=_warning(validation_status),
            )
        invalidations, validation_status = _validated_invalidations(
            invalidated_identities
        )
        if validation_status is not None:
            return StateOperationResult(
                validation_status,
                warnings=_warning(validation_status),
            )
        if not updates and not invalidations:
            return StateOperationResult("unchanged")

        def transform(
            current: ClaudeSessionState,
        ) -> StateOperationResult | tuple[str, ClaudeSessionState]:
            if current.context_generation != expected_generation:
                return StateOperationResult(
                    "stale_generation",
                    state=current,
                    warnings=_warning("stale_generation"),
                )
            timestamp = _utc_now()
            merged = dict(current.activations)
            for update in updates.values():
                merged[update.identity] = StoredActivation(
                    activation_fingerprint=update.activation_fingerprint,
                    name=update.name,
                    emitted_chars=update.emitted_chars,
                    emitted_at=timestamp,
                )
            for identity in invalidations:
                merged.pop(identity, None)
            return (
                "recorded",
                ClaudeSessionState(
                    session_id_hash=current.session_id_hash,
                    context_generation=current.context_generation,
                    last_reset_reason=current.last_reset_reason,
                    updated_at=timestamp,
                    activations=merged,
                ),
            )

        return self._mutate(session_id, transform)

    def advance_context_generation(
        self,
        session_id: str,
        *,
        reason: str,
    ) -> StateOperationResult:
        """Atomically advance context generation and clear all activations.

        This API is reserved for a future lifecycle-aware hook. The current
        UserPromptSubmit integration must not call it.
        """
        if not isinstance(reason, str) or not reason.strip():
            return StateOperationResult(
                "invalid_reset_reason",
                warnings=_warning("invalid_reset_reason"),
            )

        def transform(
            current: ClaudeSessionState,
        ) -> StateOperationResult | tuple[str, ClaudeSessionState]:
            timestamp = _utc_now()
            return (
                "advanced",
                ClaudeSessionState(
                    session_id_hash=current.session_id_hash,
                    context_generation=current.context_generation + 1,
                    last_reset_reason=reason,
                    updated_at=timestamp,
                    activations={},
                ),
            )

        return self._mutate(session_id, transform)

    def _mutate(
        self,
        session_id: str,
        transform: Callable[
            [ClaudeSessionState],
            StateOperationResult | tuple[str, ClaudeSessionState],
        ],
    ) -> StateOperationResult:
        session_id_hash, failure = _validated_session_hash(session_id)
        if failure is not None:
            return StateOperationResult(failure, warnings=_warning(failure))
        if fcntl is None:
            return StateOperationResult(
                "unsupported_lock_platform",
                warnings=_warning("unsupported_lock_platform"),
            )

        directory_failure = self._ensure_state_dir()
        if directory_failure is not None:
            return directory_failure

        lock_fd, lock_failure = self._acquire_lock(session_id_hash)
        if lock_failure is not None:
            return lock_failure
        assert lock_fd is not None

        try:
            state_path = self.state_dir / f"{session_id_hash}.json"
            loaded = _read_state_file(
                state_path,
                expected_session_hash=session_id_hash,
            )
            if not loaded.available:
                return StateOperationResult(
                    loaded.status,
                    warnings=loaded.warnings,
                )
            assert loaded.state is not None

            try:
                transformed = transform(loaded.state)
            except Exception:
                return StateOperationResult(
                    "operation_failed",
                    state=loaded.state,
                    warnings=_warning("operation_failed"),
                )
            if isinstance(transformed, StateOperationResult):
                return transformed

            success_status, new_state = transformed
            try:
                _atomic_write_state(state_path, new_state)
            except Exception:
                return StateOperationResult(
                    "write_failed",
                    state=loaded.state,
                    warnings=_warning("write_failed"),
                )
            return StateOperationResult(success_status, state=new_state)
        finally:
            self._release_lock(lock_fd)

    def _ensure_state_dir(self) -> StateOperationResult | None:
        try:
            _make_private_directories(self.state_dir)
        except OSError:
            return StateOperationResult(
                "state_dir_failed",
                warnings=_warning("state_dir_failed"),
            )
        return None

    def _acquire_lock(
        self,
        session_id_hash: str,
    ) -> tuple[int | None, StateOperationResult | None]:
        lock_path = self.state_dir / f"{session_id_hash}.lock"
        lock_fd = -1
        try:
            lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            os.fchmod(lock_fd, 0o600)
        except OSError:
            if lock_fd >= 0:
                os.close(lock_fd)
            return None, StateOperationResult(
                "lock_failed",
                warnings=_warning("lock_failed"),
            )

        deadline = time.monotonic() + self.lock_timeout_seconds
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock_fd, None
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    if time.monotonic() < deadline:
                        continue
                elif exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    os.close(lock_fd)
                    return None, StateOperationResult(
                        "lock_failed",
                        warnings=_warning("lock_failed"),
                    )

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    os.close(lock_fd)
                    return None, StateOperationResult(
                        "lock_timeout",
                        warnings=_warning("lock_timeout"),
                    )
                time.sleep(min(0.005, remaining))
                if time.monotonic() >= deadline:
                    os.close(lock_fd)
                    return None, StateOperationResult(
                        "lock_timeout",
                        warnings=_warning("lock_timeout"),
                    )

    @staticmethod
    def _release_lock(lock_fd: int) -> None:
        try:
            if fcntl is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            os.close(lock_fd)


def _validated_session_hash(session_id: str) -> tuple[str, str | None]:
    try:
        return hash_session_id(session_id), None
    except (TypeError, ValueError):
        return "", "invalid_session_id"


def _validated_updates(
    activations: Iterable[ActivationUpdate],
) -> tuple[dict[str, ActivationUpdate], str | None]:
    try:
        items = tuple(activations)
    except Exception:
        return {}, "invalid_activations"

    updates: dict[str, ActivationUpdate] = {}
    for update in items:
        if not isinstance(update, ActivationUpdate):
            return {}, "invalid_activations"
        if not isinstance(update.identity, str) or not update.identity:
            return {}, "invalid_activations"
        if (
            not isinstance(update.activation_fingerprint, str)
            or not update.activation_fingerprint
        ):
            return {}, "invalid_activations"
        if not isinstance(update.name, str):
            return {}, "invalid_activations"
        if not _is_non_negative_int(update.emitted_chars):
            return {}, "invalid_activations"
        updates[update.identity] = update
    return updates, None


def _validated_invalidations(
    identities: Iterable[str],
) -> tuple[set[str], str | None]:
    if isinstance(identities, (str, bytes)):
        return set(), "invalid_invalidations"
    try:
        items = tuple(identities)
    except Exception:
        return set(), "invalid_invalidations"

    invalidations: set[str] = set()
    for identity in items:
        if not isinstance(identity, str) or not identity:
            return set(), "invalid_invalidations"
        invalidations.add(identity)
    return invalidations, None


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


def _atomic_write_state(path: Path, state: ClaudeSessionState) -> None:
    payload = {
        "schema_version": SESSION_STATE_SCHEMA_VERSION,
        "session_id_hash": state.session_id_hash,
        "context_generation": state.context_generation,
        "last_reset_reason": state.last_reset_reason,
        "updated_at": state.updated_at,
        "activations": {
            identity: {
                "activation_fingerprint": activation.activation_fingerprint,
                "name": activation.name,
                "emitted_chars": activation.emitted_chars,
                "emitted_at": activation.emitted_at,
            }
            for identity, activation in state.activations.items()
        },
    }
    serialized = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if len(serialized.encode("utf-8")) > MAX_STATE_FILE_BYTES:
        raise ValueError("serialized Claude session state exceeds maximum size")

    file_descriptor = -1
    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            file_descriptor = -1
            handle.write(serialized)
            _flush_and_sync(handle)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _make_private_directories(path: Path) -> None:
    missing: list[Path] = []
    current = path
    while True:
        try:
            current_mode = current.stat().st_mode
        except FileNotFoundError:
            missing.append(current)
            parent = current.parent
            if parent == current:
                raise
            current = parent
            continue

        if not stat.S_ISDIR(current_mode):
            raise NotADirectoryError(current)
        break

    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            if not directory.is_dir():
                raise NotADirectoryError(directory)
        else:
            os.chmod(directory, 0o700)


def _flush_and_sync(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _warning(status: str) -> tuple[str, ...]:
    return (f"claude_session_state_{status}",)


__all__ = [
    "ActivationUpdate",
    "ClaudeSessionState",
    "ClaudeSessionStateStore",
    "DEFAULT_LOCK_TIMEOUT_SECONDS",
    "MAX_STATE_FILE_BYTES",
    "SESSION_STATE_SCHEMA_VERSION",
    "StateLoadResult",
    "StateOperationResult",
    "StoredActivation",
    "hash_session_id",
    "resolve_state_dir",
]
