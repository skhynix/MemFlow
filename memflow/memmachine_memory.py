# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optional, explicit access to existing MemMachine episodic and semantic memory."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Mapping
from typing import Any


def _nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _json_value(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("set_metadata numbers must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("set_metadata keys must be strings")
            _json_value(item)
        return
    raise TypeError("set_metadata must contain JSON values")


# Keep episodic and semantic memory queries available for potential future use.
# This optional client is intentionally separate from the skill-only core.
class MemMachineMemoryClient:
    """Query an existing project without creating or writing memory.

    The SDK and connection are initialized on the first valid search. Each
    search gets its own user-scoped Memory object; semantic set selection
    must be supplied separately by the caller.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080",
        org_id: str = "default",
        project_id: str = "memflow",
        api_key: str | None = None,
    ) -> None:
        self._base_url = base_url
        self._org_id = org_id
        self._project_id = project_id
        self._api_key = api_key
        self._client: Any = None
        self._project: Any = None
        self._closed = False
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> MemMachineMemoryClient:
        """Read explicit process settings without loading an environment file."""
        return cls(
            base_url=os.getenv("MEMMACHINE_BASE_URL", "http://localhost:8080"),
            org_id=os.getenv("MEMMACHINE_ORG_ID", "default"),
            project_id=os.getenv("MEMMACHINE_PROJECT", "memflow"),
            api_key=os.getenv("MEMMACHINE_API_KEY"),
        )

    def _get_project(self) -> Any:
        with self._lock:
            if self._closed:
                raise RuntimeError("MemMachine memory client is closed")
            if self._project is None:
                try:
                    from memmachine_client import MemMachineClient
                except ImportError as exc:
                    raise ImportError(
                        "MemMachine memory queries require memflow[memmachine]"
                    ) from exc

                kwargs: dict[str, Any] = {"base_url": self._base_url}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                client = MemMachineClient(**kwargs)
                try:
                    project = client.get_project(
                        org_id=self._org_id, project_id=self._project_id
                    )
                except Exception:
                    try:
                        client.close()
                    except Exception:
                        pass  # Preserve the project lookup failure.
                    raise
                self._client = client
                self._project = project
            return self._project

    def search(
        self,
        query: str,
        *,
        user_id: str,
        set_metadata: Mapping[str, Any],
        limit: int = 10,
        timeout: int | None = None,
    ) -> Any:
        """Return the SDK response unchanged, scoped to a user and semantic set."""
        _nonempty_string(query, "query")
        _nonempty_string(user_id, "user_id")
        if not isinstance(set_metadata, Mapping):
            raise TypeError("set_metadata must be a mapping")
        if not set_metadata:
            raise ValueError("set_metadata must not be empty")
        metadata = dict(set_metadata)
        try:
            _json_value(metadata)
        except RecursionError as exc:
            raise ValueError("set_metadata must be an acyclic JSON value") from exc
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be positive")

        project = self._get_project()
        memory = project.memory(metadata={"user_id": user_id})
        return memory.search(
            query=query, limit=limit, timeout=timeout, set_metadata=metadata
        )

    def close(self) -> None:
        """Release an initialized SDK client once; never connect during close."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._client is not None:
                self._client.close()
