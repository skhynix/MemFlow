# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Small test stores with explicit search responses; no retrieval algorithm."""

from memflow.models import SearchResult, SkillRecord
from memflow.store import BaseStore


class FakeSkillStore(BaseStore):
    def __init__(self):
        self.records: dict[str, SkillRecord] = {}
        self.search_results: list[SearchResult] = []
        self.search_calls: list[dict] = []

    def add(self, skill):
        records = skill if isinstance(skill, list) else [skill]
        for skill in records:
            self.records[skill.id] = skill
        return len(records)

    def search(self, query, top_k=5, user_id=None, kind="skill"):
        self.search_calls.append(
            {"query": query, "top_k": top_k, "user_id": user_id, "kind": kind}
        )
        if kind != "skill":
            raise ValueError("only kind='skill' is supported")
        matches = [
            result
            for result in self.search_results
            if (not user_id or result.skill.user_id == user_id)
            and result.skill.kind == kind
        ][:top_k]
        return [list(matches) for _ in query] if isinstance(query, list) else matches

    def get(self, id):
        return self.records.get(id)

    def delete(self, id):
        if isinstance(id, list):
            return sum(self.delete(item) for item in id)
        return int(self.records.pop(id, None) is not None)

    def list(self, user_id=None):
        return [
            record
            for record in self.records.values()
            if not user_id or record.user_id == user_id
        ]
