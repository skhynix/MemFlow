# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""SkillReason-Bench benchmark harness for MemFlow.

SkillReason-Bench (https://huggingface.co/datasets/donghongjiang/skillreason-bench)
evaluates agent skill retrieval for implicit user requests: 3,729 concise task
goals over a corpus of 61,228 web-collected skills spanning ten domains, each
query carrying exactly one gold skill.

Public surface mirrors the skill_ret_bench harness layout:

- adapter: corpus normalization, seeding, and the MemFlow retrieval adapter
- evaluation: query loading, binary IR metrics, and the official-protocol
  metrics (Recall@{1,5,10,20}, NDCG@10, MRR@20)
"""

from __future__ import annotations

from .adapter import (
    SKILL_REASON_CONTENT_MAX_CHARS,
    SKILL_REASON_QUERY_INSTRUCTION,
    MemFlowSkillReasonAdapter,
    RetrievedSkill,
    SkillReasonRecord,
    iter_skill_reason_records,
    seed_skill_reason_corpus,
    skill_record_to_procedure,
)
from .evaluation import (
    SkillReasonEvaluationResult,
    SkillReasonQuery,
    aggregate_query_metrics,
    compute_binary_ir_metrics,
    compute_official_metrics,
    count_query_records,
    evaluate_skill_reason_queries,
    evaluate_skill_reason_queries_async,
    load_skill_reason_queries,
    write_official_predictions_jsonl,
)

__all__ = [
    "SKILL_REASON_CONTENT_MAX_CHARS",
    "SKILL_REASON_QUERY_INSTRUCTION",
    "MemFlowSkillReasonAdapter",
    "RetrievedSkill",
    "SkillReasonEvaluationResult",
    "SkillReasonQuery",
    "SkillReasonRecord",
    "aggregate_query_metrics",
    "compute_binary_ir_metrics",
    "compute_official_metrics",
    "count_query_records",
    "evaluate_skill_reason_queries",
    "evaluate_skill_reason_queries_async",
    "iter_skill_reason_records",
    "load_skill_reason_queries",
    "seed_skill_reason_corpus",
    "skill_record_to_procedure",
    "write_official_predictions_jsonl",
]
