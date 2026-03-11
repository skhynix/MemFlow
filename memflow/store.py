"""
Vector store abstraction for MemFlow.

Phase 1: EmulatedStore (in-memory, word-overlap search)
Phase 2: FileStore (.md files), MemMachineStore (production)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from memflow.models import Procedure, SearchResult


def _text_score(text: str, query: str) -> float:
    """Word-overlap relevance score in [0, 1]."""
    if not query.strip():
        return 1.0
    text_words = set(text.lower().split())
    query_words = set(query.lower().split())
    if not query_words:
        return 0.0
    return len(text_words & query_words) / len(query_words)


class BaseStore(ABC):
    """Abstract base for all storage backends."""

    @abstractmethod
    def add(self, procedure: Procedure) -> None: ...

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[SearchResult]: ...

    @abstractmethod
    def get(self, id: str) -> Procedure | None: ...

    @abstractmethod
    def delete(self, id: str) -> bool: ...

    @abstractmethod
    def list_all(self, user_id: str | None = None) -> list[Procedure]: ...


class EmulatedStore(BaseStore):
    """
    In-memory store with word-overlap search.

    All data is lost on process restart.
    Suitable for Phase 1 validation and testing.
    """

    def __init__(self) -> None:
        self._store: dict[str, Procedure] = {}

    def add(self, procedure: Procedure) -> None:
        self._store[procedure.id] = procedure

    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[SearchResult]:
        results = []
        for proc in self._store.values():
            if user_id and proc.user_id != user_id:
                continue
            score = _text_score(proc.title + " " + proc.content, query)
            if score > 0:
                results.append(SearchResult(procedure=proc, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def get(self, id: str) -> Procedure | None:
        return self._store.get(id)

    def delete(self, id: str) -> bool:
        if id in self._store:
            del self._store[id]
            return True
        return False

    def list_all(self, user_id: str | None = None) -> list[Procedure]:
        procs = list(self._store.values())
        if user_id:
            procs = [p for p in procs if p.user_id == user_id]
        return procs
