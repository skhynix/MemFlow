# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the SkillReason-Bench benchmark harness."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sys
import types
from argparse import Namespace
from types import SimpleNamespace

import pytest

from benchmark.skill_reason_bench.adapter import (
    SKILL_REASON_CONTENT_MAX_CHARS,
    normalize_skill_reason_record,
    seed_skill_reason_corpus,
    skill_reason_record_to_skill,
    truncate_skill_reason_content,
)
from benchmark.skill_reason_bench.evaluation import (
    compute_official_metrics,
    load_skill_reason_queries,
)
from memflow.models import skill_search_text


def test_search_text_keeps_name_description_and_body() -> None:
    raw = {
        "skill_id": "openclaw-repo-skills-autoreview-skill-md",
        "name": "autoreview",
        "description": "Auto Review closeout.",
        "body": "# Auto Review\n\nRun the bundled structured review helper.",
        "author": "openclaw",
        "repo_slug": "openclaw",
        "path": "SKILL.md",
        "github_url": "https://github.com/openclaw/openclaw",
        "stars": 380125,
    }

    record = normalize_skill_reason_record(raw)
    skill = skill_reason_record_to_skill(record, user_id="benchmark")
    text = skill_search_text(skill)

    assert skill.kind == "skill"
    assert record.content.startswith(raw["description"])
    assert "Run the bundled structured review helper." in record.content
    assert raw["description"] in text
    assert "autoreview" in text
    # Source metadata is preserved verbatim for downstream utility analysis.
    assert skill.metadata["skill"]["stars"] == 380125
    assert skill.metadata["skill"]["github_url"] == raw["github_url"]


def test_content_cap_truncates_to_official_budget() -> None:
    raw = {
        "skill_id": "skill-id",
        "name": "long-skill",
        "description": "D" * 100,
        "body": "B" * (SKILL_REASON_CONTENT_MAX_CHARS * 3),
    }

    record = normalize_skill_reason_record(raw)
    capped, was_truncated = truncate_skill_reason_content(record)

    assert was_truncated
    assert len(capped.content) == SKILL_REASON_CONTENT_MAX_CHARS
    # The cap is only applied via truncate_skill_reason_content, so normalization
    # keeps the full text observable (and short docs pass through untouched).
    assert len(record.content) > SKILL_REASON_CONTENT_MAX_CHARS

    short = normalize_skill_reason_record({**raw, "body": "short body"})
    passed, truncated = truncate_skill_reason_content(short)
    assert not truncated
    assert passed.content == short.content


def test_loader_maps_positive_ids_to_relevant_set(tmp_path) -> None:
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        '{"query_id": 1, "query": "need X", "positive_ids": ["skill-a"],'
        ' "category": "cat", "domain": "software"}\n'
        '{"query_id": 2, "query": "   ", "positive_ids": ["skill-b"],'
        ' "category": "cat", "domain": "legal"}\n'
        '{"query_id": 3, "query": "need Y", "positive_ids": ["skill-c","skill-d"],'
        ' "category": "cat", "domain": "finance"}\n',
        encoding="utf-8",
    )

    loaded = load_skill_reason_queries(queries)

    # The blank query is dropped; the rest keep their gold IDs as strings.
    assert [q.query_id for q in loaded] == ["1", "3"]
    assert loaded[0].relevant_skill_ids == ["skill-a"]
    assert loaded[0].domain == "software"
    assert loaded[1].relevant_skill_ids == ["skill-c", "skill-d"]

    assert [q.query_id for q in load_skill_reason_queries(queries, max_queries=1)] == [
        "1"
    ]


# ---------------------------------------------------------------------------
# Official-metric parity
#
# The loop below is a transcription of the scoring section of the dataset's
# own evaluate.py (donghongjiang/skillreason-bench) so any drift between our
# compute_official_metrics and the upstream protocol fails this test.
# ---------------------------------------------------------------------------
def _upstream_reference_metrics(queries, predictions) -> dict:
    ks = (1, 5, 10, 20)
    recall = {k: 0.0 for k in ks}
    ndcg10 = 0.0
    mrr20 = 0.0
    for query_id, gold in queries.items():
        ranked = predictions.get(query_id, [])
        for k in ks:
            recall[k] += len(gold.intersection(ranked[:k])) / len(gold)
        first_rank = next(
            (i + 1 for i, sid in enumerate(ranked[:20]) if sid in gold), None
        )
        if first_rank is not None:
            mrr20 += 1.0 / first_rank
            if first_rank <= 10:
                ndcg10 += 1.0 / math.log2(first_rank + 1)
    total = len(queries)
    metrics = {f"Recall@{k}": recall[k] / total for k in ks}
    metrics["NDCG@10"] = ndcg10 / total
    metrics["MRR@20"] = mrr20 / total
    return metrics


def test_official_metrics_match_upstream_scoring_loop() -> None:
    gold = {
        "q1": {"hit-rank-1"},
        "q2": {"hit-rank-7"},
        "q3": {"hit-rank-15"},  # within MRR@20 but outside NDCG@10
        "q4": {"never-retrieved"},
        "q5": {"miss-beyond-20"},
        "q6": {"late-hit-rank-20"},
    }
    filler = [f"junk-{i}" for i in range(30)]
    predictions = {
        "q1": ["hit-rank-1", *filler],
        "q2": [*filler[:6], "hit-rank-7", *filler[6:]],
        "q3": [*filler[:14], "hit-rank-15", *filler[14:]],
        "q4": filler,
        "q5": [*filler[:20], "miss-beyond-20", *filler[20:]],
        "q6": [*filler[:19], "late-hit-rank-20"],
        # q7 is extra: it is not in the gold set and must not affect scoring.
    }

    ours = compute_official_metrics(predictions, gold)
    reference = _upstream_reference_metrics(gold, predictions)

    for key, want in reference.items():
        assert ours[key] == want, f"{key}: ours={ours[key]} upstream={want}"
    # Spot-check known positions: rank-15 hit keeps NDCG@10 at zero for q3.
    assert ours["NDCG@10"] == (1.0 + 1 / math.log2(8)) / len(gold)
    assert ours["MRR@20"] == (1.0 + 1 / 7 + 1 / 15 + 1 / 20) / len(gold)
    assert ours["prediction_coverage"] == 6 / 6
    assert ours["num_queries"] == 6


def test_official_metrics_reject_empty_gold_set() -> None:
    """A gold-less query set must fail loudly, not score all-zero."""
    with pytest.raises(ValueError, match="no gold queries"):
        compute_official_metrics({}, {})


def test_integrated_runner_awaits_corpus_seeding(monkeypatch, tmp_path) -> None:
    from benchmark.skill_reason_bench import run_skill_reason as runner

    awaited = False

    async def fake_seed_skill_reason_corpus(**kwargs):
        nonlocal awaited
        awaited = True
        return SimpleNamespace(
            active_corpus_size=3,
            to_dict=lambda: {"num_seeded": 3},
        )

    args = Namespace(
        user_id="benchmark",
        k_values=[1],
        corpus_path=tmp_path / "skills.jsonl",
        queries_path=tmp_path / "queries.jsonl",
        results_dir=tmp_path,
        results_filename="seed-result",
        seed_only=True,
        clear_existing=False,
        max_records=None,
        max_queries=None,
        embedding_batch_size=64,
        sync=False,
        max_workers=48,
        max_concurrency=64,
        skip_verify=True,
    )
    monkeypatch.setattr(runner, "_load_env_file", lambda *_: None)
    monkeypatch.setattr(runner, "_parse_args", lambda: args)
    monkeypatch.setattr(runner, "MemFlow", lambda: object())
    monkeypatch.setattr(
        runner, "seed_skill_reason_corpus", fake_seed_skill_reason_corpus
    )

    runner.main()

    assert awaited
    assert (tmp_path / "seed-result.json").exists()


def test_runners_default_to_official_dataset_paths() -> None:
    from benchmark.skill_reason_bench import run_retrieval, run_seeding

    assert run_seeding.DEFAULT_CORPUS_PATH.as_posix().endswith(
        "data/skillreason-bench/data/skills.jsonl"
    )
    assert run_retrieval.DEFAULT_QUERIES_PATH.as_posix().endswith(
        "data/skillreason-bench/data/queries.jsonl"
    )


def test_runners_reject_nonpositive_k_values_as_systemexit(
    monkeypatch, tmp_path
) -> None:
    """Both runners must fail the same way: CLI misuse is argparse-style."""
    from benchmark.skill_reason_bench import run_retrieval, run_skill_reason

    with pytest.raises(SystemExit, match="positive integer"):
        run_retrieval.run_evaluation(
            memflow=None,
            args=Namespace(
                k_values=[0, -3],
                queries_path=tmp_path / "queries.jsonl",
                user_id="bench",
                sync=True,
            ),
            backend="qdrant",
        )

    args = Namespace(
        user_id="benchmark",
        k_values=[0],
        corpus_path=tmp_path / "skills.jsonl",
        queries_path=tmp_path / "queries.jsonl",
        results_dir=tmp_path,
        results_filename=None,
        seed_only=False,
        clear_existing=False,
        max_records=None,
        max_queries=None,
        embedding_batch_size=64,
        sync=False,
        max_workers=48,
        max_concurrency=64,
        skip_verify=True,
    )
    monkeypatch.setattr(run_skill_reason, "_load_env_file", lambda *_: None)
    monkeypatch.setattr(run_skill_reason, "_parse_args", lambda: args)
    with pytest.raises(SystemExit, match="positive integer"):
        run_skill_reason.main()


def test_plot_k_grid_anchors_on_official_cutoffs() -> None:
    """Step grid must land on the official cutoffs, not 1, 6, 11, ..."""
    from benchmark.skill_reason_bench.plot_recall_curves import _k_grid

    grid = _k_grid(1, 100, 5)
    assert {1, 5, 10, 20, 100} <= set(grid)
    assert 6 not in grid
    assert grid == sorted(grid)
    # k_min always present even when it misses the step multiples.
    assert _k_grid(2, 10, 5) == [2, 5, 10]


class _RecordingStore:
    """Store double that records which bulk primitives the seeder uses.

    Unlike the SkillRet double, inserts land on the store directly (the
    SkillReason seeder embeds through ``store.add_async`` so the official
    corpus feeds the embedding server with ``batch_size`` honored, which the
    manager wrapper does not forward).

    Added skills land in ``existing_ids`` so the seeder's post-seeding
    completeness verify (which re-lists the store) sees them as present by
    default; ``drop_on_store`` simulates a backend that silently fails to
    persist some ids, so the verify reports them as missing.
    """

    def __init__(
        self,
        existing_ids: list[str],
        *,
        drop_on_store: set[str] | None = None,
    ) -> None:
        self.existing_ids = set(existing_ids)
        self.drop_on_store = drop_on_store or set()
        self.list_called = False
        self.list_ids_calls = 0
        self.delete_many_calls: list[list[str]] = []
        self.deleted: list[str] = []
        self.added: list = []
        self.embedding_batch_sizes: list[int] = []
        self.sync_add_calls = 0

    def list(self, user_id: str | None = None):
        self.list_called = True
        raise AssertionError("seed must not fetch full skill payloads via list()")

    def list_ids(self, user_id: str | None = None) -> list[str]:
        self.list_ids_calls += 1
        return sorted(self.existing_ids)

    def delete_many(self, ids: list[str]) -> int:
        self.delete_many_calls.append(list(ids))
        self.deleted.extend(ids)
        self.existing_ids -= set(ids)
        return len(ids)

    def delete(self, id):  # pragma: no cover - per-id path must not be used
        raise AssertionError("seed must not delete one id at a time")

    def _store(self, skills: list) -> int:
        self.added.extend(skills)
        for skill in skills:
            if skill.id in self.drop_on_store:
                continue
            self.existing_ids.add(skill.id)
        # Like QdrantStore, report submitted ids — an accepted-but-dropped
        # upsert still counts here, which is exactly what the verify catches.
        return len(skills)

    def add(self, skill, batch_size: int = 10) -> int:
        self.sync_add_calls += 1
        self.embedding_batch_sizes.append(batch_size)
        skills = skill if isinstance(skill, list) else [skill]
        return self._store(skills)

    async def add_async(self, skill, batch_size: int = 10, max_workers: int = 10):
        skills = skill if isinstance(skill, list) else [skill]
        self.embedding_batch_sizes.append(batch_size)
        return self._store(skills)


class _RecordingMemFlow:
    def __init__(self, store: _RecordingStore) -> None:
        self.store = store


def _write_corpus(tmp_path):
    corpus_path = tmp_path / "skills.jsonl"
    records = [
        {"skill_id": "s_001", "name": "Alpha", "description": "d a", "body": "body a"},
        {"skill_id": "s_002", "name": "Beta", "description": "d b", "body": "body b"},
    ]
    corpus_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return corpus_path


def test_seed_clear_existing_uses_bulk_primitives(tmp_path) -> None:
    """--clear-existing must not pull full payloads nor delete per id.

    Per-id deletion costs two serial round trips per document (get +
    delete); on the 61k-doc corpus that is hours of silent round trips.
    The bulk path is list_ids() + one batched delete_many().
    """
    corpus_path = _write_corpus(tmp_path)
    store = _RecordingStore(existing_ids=["s_001", "unrelated"])
    memflow = _RecordingMemFlow(store)

    stats = asyncio.run(
        seed_skill_reason_corpus(
            memflow=memflow,
            user_id="bench-user",
            corpus_path=corpus_path,
            clear_existing=True,
        )
    )

    assert store.list_called is False
    # Once to plan the clear, once to verify completeness after seeding.
    assert store.list_ids_calls == 2
    # Only corpus ids that exist are deleted; foreign ids stay untouched.
    assert store.deleted == ["s_001"]
    assert stats.num_deleted == 1
    assert stats.num_seeded == 2
    assert [p.id for p in store.added] == ["s_001", "s_002"]
    # The direct-store path exists to honor the embedding batch size.
    assert store.embedding_batch_sizes == [64]
    # Everything promised landed, so the completeness verify finds no gap.
    assert stats.num_missing == 0


def test_seed_reuse_path_uses_list_ids(tmp_path) -> None:
    """Reuse listing must also use list_ids, not full-payload list()."""
    corpus_path = _write_corpus(tmp_path)
    store = _RecordingStore(existing_ids=["s_001"])
    memflow = _RecordingMemFlow(store)

    stats = asyncio.run(
        seed_skill_reason_corpus(
            memflow=memflow,
            user_id="bench-user",
            corpus_path=corpus_path,
        )
    )

    assert store.list_called is False
    assert stats.num_reused == 1
    assert stats.num_seeded == 1
    assert [p.id for p in store.added] == ["s_002"]


# ---------------------------------------------------------------------------
# Download fallback and integrity gates
# ---------------------------------------------------------------------------
def test_hub_download_failure_falls_back_to_https(monkeypatch, tmp_path) -> None:
    """A hub *runtime* failure (network, 401, API error) must fall back too.

    Before, only ImportError opted out of huggingface_hub, so a hub-enabled
    environment aborted the whole download on any transient failure even
    though the docstring promised an HTTPS fallback.
    """

    def _broken_download(**kwargs):
        raise RuntimeError("simulated hub API failure")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = _broken_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    from benchmark.skill_reason_bench import download_data as dd

    https_calls: list[str] = []

    def _fake_urlretrieve(url: str, filename: str) -> None:
        https_calls.append(url)
        with open(filename, "wb") as fh:
            fh.write(b"data")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlretrieve", _fake_urlretrieve)

    out_path = tmp_path / "tasks.jsonl"
    dd.download_file("tasks.jsonl", tmp_path, force=False)

    assert https_calls == [
        f"https://huggingface.co/datasets/{dd.REPO}/resolve/main/tasks.jsonl"
    ]
    assert out_path.exists()


def test_loader_drops_queries_without_gold(tmp_path) -> None:
    """Gold-less records would crash the official block (div by len(gold)).

    The loader must drop them with a warning instead of silently diluting the
    binary averages or crashing compute_official_metrics at scoring time.
    """
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        '{"query_id": 1, "query": "need X", "positive_ids": ["skill-a"],'
        ' "category": "c", "domain": "software"}\n'
        '{"query_id": 2, "query": "need Y", "positive_ids": [],'
        ' "category": "c", "domain": "legal"}\n'
        '{"query_id": 3, "query": "need Z", "category": "c", "domain": "legal"}\n',
        encoding="utf-8",
    )

    loaded = load_skill_reason_queries(queries)

    # Only the record with a non-empty positive_ids survives.
    assert [q.query_id for q in loaded] == ["1"]

    # The surviving set scores through the official block without crashing.
    metrics = compute_official_metrics(
        {"1": ["skill-a"]}, {q.query_id: set(q.relevant_skill_ids) for q in loaded}
    )
    assert metrics["MRR@20"] == 1.0


def _stats_with_digest(path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (path.parent / "stats.json").write_text(
        json.dumps({"sha256": {path.name: digest}}), encoding="utf-8"
    )


def test_seeding_verify_gate_rejects_corrupted_corpus(tmp_path) -> None:
    """The seeding gate must re-check the corpus against stats.json.

    download_data verifies at download time only; a file truncated afterwards
    (or placed by other means) used to seed silently.
    """
    from benchmark.skill_reason_bench import run_seeding

    corpus_path = tmp_path / "skills.jsonl"
    corpus_path.write_text('{"skill_id": "s_001"}\n', encoding="utf-8")
    _stats_with_digest(corpus_path)
    corpus_path.write_text('{"skill_id": "truncated\n', encoding="utf-8")

    args = Namespace(corpus_path=corpus_path, skip_verify=False)
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        run_seeding.verify_data_file_checksum(args.corpus_path, args.skip_verify)

    # --skip-verify stays an explicit escape hatch for smoke corpora.
    run_seeding.verify_data_file_checksum(corpus_path, skip_verify=True)


def test_seeding_verify_gate_accepts_matching_digest(tmp_path) -> None:
    from benchmark.skill_reason_bench.download_data import verify_data_file_checksum

    corpus_path = tmp_path / "skills.jsonl"
    corpus_path.write_text('{"skill_id": "s_001"}\n', encoding="utf-8")
    _stats_with_digest(corpus_path)

    verify_data_file_checksum(corpus_path)  # must not raise


def test_seeding_verify_gate_without_stats_is_not_a_failure(tmp_path) -> None:
    """Smoke subsets in their own dir have no stats.json to check against."""
    from benchmark.skill_reason_bench.download_data import verify_data_file_checksum

    corpus_path = tmp_path / "skills.jsonl"
    corpus_path.write_text('{"skill_id": "s_001"}\n', encoding="utf-8")

    verify_data_file_checksum(corpus_path)  # must not raise
    verify_data_file_checksum(tmp_path)  # directory corpus: skipped, not fatal


# ---------------------------------------------------------------------------
# Seeder completeness verify + strict-mode note
# ---------------------------------------------------------------------------
def test_seed_reports_docs_missing_from_store(tmp_path, capsys) -> None:
    """Counters only record what the backend accepted, not what it stored.

    A dropped upsert (flaky endpoint) must surface as num_missing with a
    sample, not as a silently short corpus — the gap that motivated this
    check looked exactly like this on the full run (61,226 of 61,228).
    """
    corpus_path = _write_corpus(tmp_path)
    store = _RecordingStore([], drop_on_store={"s_002"})
    memflow = _RecordingMemFlow(store)

    stats = asyncio.run(
        seed_skill_reason_corpus(
            memflow=memflow,
            user_id="bench-user",
            corpus_path=corpus_path,
        )
    )

    assert stats.num_seeded == 2  # backend accepted both
    assert stats.num_missing == 1  # but only one is actually in the store
    assert stats.missing_ids_sample == ["s_002"]
    warning = capsys.readouterr().out
    assert "[WARNING]" in warning and "s_002" in warning


def test_seed_reuse_verify_excludes_shortfall_from_reuse(tmp_path) -> None:
    """Reuse-listed ids are part of the verify expectation, not just seeded ones.

    An id counted as reused but absent from the store at verify time is a
    genuine shortfall and must be reported.
    """
    corpus_path = _write_corpus(tmp_path)
    store = _RecordingStore(["s_001", "s_002"])
    memflow = _RecordingMemFlow(store)
    stats = asyncio.run(
        seed_skill_reason_corpus(
            memflow=memflow, user_id="bench-user", corpus_path=corpus_path
        )
    )
    assert stats.num_reused == 2 and stats.num_seeded == 0
    assert stats.num_missing == 0  # both reuse ids present at verify


def test_seed_sync_path_uses_store_add(tmp_path) -> None:
    """--sync must route through store.add (the sync primitive)."""
    corpus_path = _write_corpus(tmp_path)
    store = _RecordingStore([])
    memflow = _RecordingMemFlow(store)

    stats = asyncio.run(
        seed_skill_reason_corpus(
            memflow=memflow,
            user_id="bench-user",
            corpus_path=corpus_path,
            use_sync=True,
            embedding_batch_size=32,
        )
    )

    assert stats.num_seeded == 2
    assert store.sync_add_calls == 1  # one batch of 2 via store.add
    # store.add accepts batch_size too; dropping it would silently fall back
    # to the store default of 10 — the exact pathology the direct-store
    # bypass exists to avoid.
    assert store.embedding_batch_sizes == [32]


def test_seed_max_records_budget_trims_tail(tmp_path) -> None:
    """--max-records caps seeding and counts the dropped tail as skipped."""
    corpus_path = tmp_path / "skills.jsonl"
    records = [
        {"skill_id": f"s_{i:03d}", "name": f"N{i}", "description": "d", "body": "b"}
        for i in range(10)
    ]
    corpus_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    store = _RecordingStore([])
    memflow = _RecordingMemFlow(store)

    stats = asyncio.run(
        seed_skill_reason_corpus(
            memflow=memflow,
            user_id="bench-user",
            corpus_path=corpus_path,
            batch_size=4,
            max_records=6,
        )
    )

    assert stats.num_seeded == 6
    assert len(store.added) == 6
    assert {p.id for p in store.added} == {f"s_{i:03d}" for i in range(6)}
    assert stats.num_missing == 0


def test_seed_warns_when_embedding_strict_off(tmp_path, capsys) -> None:
    """A flaky endpoint silently stores fallback vectors unless strict is on."""
    corpus_path = _write_corpus(tmp_path)
    store = _RecordingStore([])
    memflow = _RecordingMemFlow(store)

    asyncio.run(
        seed_skill_reason_corpus(
            memflow=memflow, user_id="bench-user", corpus_path=corpus_path
        )
    )
    assert "MEMFLOW_EMBEDDING_STRICT" in capsys.readouterr().out


def test_seed_stays_quiet_when_embedding_strict_on(
    tmp_path, capsys, monkeypatch
) -> None:
    corpus_path = _write_corpus(tmp_path)
    store = _RecordingStore([])
    memflow = _RecordingMemFlow(store)
    monkeypatch.setenv("MEMFLOW_EMBEDDING_STRICT", "on")

    asyncio.run(
        seed_skill_reason_corpus(
            memflow=memflow, user_id="bench-user", corpus_path=corpus_path
        )
    )
    assert "MEMFLOW_EMBEDDING_STRICT is off" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Binary IR metrics (hand-computed expectations)
# ---------------------------------------------------------------------------
def test_binary_ir_metrics_hand_computed() -> None:
    from benchmark.skill_reason_bench.evaluation import compute_binary_ir_metrics

    # Two golds; one at rank 1, one at rank 3; no gold at rank 2.
    metrics = compute_binary_ir_metrics(
        retrieved_ids=["g1", "x", "g2", "y"],
        relevant_ids=["g1", "g2"],
        k_values=[1, 2, 3, 4],
    )

    assert metrics["hit_at_k"] == {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0}
    assert metrics["precision_at_k"] == {"1": 1.0, "2": 0.5, "3": 2 / 3, "4": 0.5}
    assert metrics["recall_at_k"] == {"1": 0.5, "2": 0.5, "3": 1.0, "4": 1.0}
    # binary NDCG@2: DCG = 1/log2(2) = 1.0 (only the rank-1 gold),
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 1/log2(3).
    assert metrics["ndcg_at_k"]["2"] == 1.0 / (1.0 + 1.0 / math.log2(3))
    assert metrics["reciprocal_rank"] == 1.0
    # MAP: precision 1/1 at first hit, 2/3 at second → (1 + 2/3)/2.
    assert metrics["average_precision"] == (1.0 + 2 / 3) / 2
    assert metrics["num_relevant"] == 2
    assert metrics["num_relevant_retrieved"] == 2

    # A duplicate of a gold must not count twice.
    dup = compute_binary_ir_metrics(
        retrieved_ids=["g1", "g1"], relevant_ids=["g1"], k_values=[2]
    )
    assert dup["precision_at_k"]["2"] == 0.5
    assert dup["average_precision"] == 1.0


def test_binary_ir_metrics_all_miss() -> None:
    from benchmark.skill_reason_bench.evaluation import compute_binary_ir_metrics

    metrics = compute_binary_ir_metrics(
        retrieved_ids=["a", "b"], relevant_ids=["z"], k_values=[1, 2]
    )
    assert metrics["hit_at_k"] == {"1": 0.0, "2": 0.0}
    assert metrics["ndcg_at_k"] == {"1": 0.0, "2": 0.0}
    assert metrics["reciprocal_rank"] == 0.0
    assert metrics["average_precision"] == 0.0


# ---------------------------------------------------------------------------
# End-to-end scoring assembly (sync/async paths) + adapter retrieval semantics
# ---------------------------------------------------------------------------
class _StubSearchStore:
    """Minimal MemFlow double: memflow.search(query, ...) -> SearchResult(s)."""

    def __init__(self, ranking_by_query: dict[str, list[str]]) -> None:
        self.ranking_by_query = ranking_by_query
        self.search_calls: list[tuple[str | list[str], int]] = []

    def search(self, query, user_id=None, top_k=5, kind="skill"):
        self.search_calls.append((query, top_k))
        from memflow.models import SearchResult, SkillRecord

        queries = query if isinstance(query, list) else [query]
        results = [
            [
                SearchResult(
                    skill=SkillRecord(
                        id=sid, title=sid, content="", category="general", tags=[]
                    ),
                    score=1.0 / (rank + 1),
                )
                for rank, sid in enumerate(self.ranking_by_query.get(text, [])[:top_k])
            ]
            for text in queries
        ]
        return results if isinstance(query, list) else results[0]


def _query(qid: str, text: str, gold: list[str], domain: str):
    from benchmark.skill_reason_bench.evaluation import SkillReasonQuery

    return SkillReasonQuery(
        query_id=qid,
        query=text,
        domain=domain,
        category="c",
        relevant_skill_ids=list(gold),
    )


@pytest.mark.parametrize(
    "method", ["retrieve", "retrieve_async", "retrieve_batch", "retrieve_batch_async"]
)
def test_adapter_retrieve_fetches_extra_and_excludes(method) -> None:
    from benchmark.skill_reason_bench.adapter import MemFlowSkillReasonAdapter

    store = _StubSearchStore({"q": ["excluded", "a", "b", "c", "d"]})

    class _MemFlowShim:
        search = store.search

    adapter = MemFlowSkillReasonAdapter(
        memflow=_MemFlowShim(),  # type: ignore[arg-type]
        user_id="bench",
        corpus_size=5,
        backend="test",
    )
    retrieve = getattr(adapter, method)
    if "batch" in method:
        results = retrieve([("q", {"excluded"})], k=2)
    else:
        results = retrieve("q", k=2, exclude_skill_ids={"excluded"})
    if method.endswith("_async"):
        results = asyncio.run(results)
    if "batch" in method:
        results = results[0]

    # The excluded id consumed a fetch slot: fetch_k = k + len(excluded) = 3.
    assert store.search_calls == [(["q"] if method == "retrieve_batch" else "q", 3)]
    assert [r.skill_id for r in results] == ["a", "b"]


@pytest.mark.parametrize("use_async", [False, True])
def test_evaluate_assembles_overall_official_and_domain_metrics(
    tmp_path, use_async
) -> None:
    from benchmark.skill_reason_bench.adapter import MemFlowSkillReasonAdapter
    from benchmark.skill_reason_bench.evaluation import (
        evaluate_skill_reason_queries,
        evaluate_skill_reason_queries_async,
        write_official_predictions_jsonl,
    )

    queries = [
        _query("q1", "need alpha", ["gold-a"], "software"),
        _query("q2", "need beta", ["gold-b"], "legal"),
        _query("q3", "need gamma", ["gold-c"], "legal"),
    ]
    store = _StubSearchStore(
        {
            "need alpha": ["gold-a", "n1", "n2"],
            "need beta": ["n1", "gold-b", "n2"],  # rank 2
            "need gamma": ["n1", "n2", "n3"],  # miss within fetch depth
        }
    )

    adapter = MemFlowSkillReasonAdapter(
        memflow=store,  # type: ignore[arg-type]
        user_id="bench",
        corpus_size=5,
        backend="test",
    )
    evaluate = (
        evaluate_skill_reason_queries_async
        if use_async
        else evaluate_skill_reason_queries
    )
    result = evaluate(
        retrieval_system=adapter,
        queries=queries,
        k_values=[1, 3],
        top_k=3,
    )
    if use_async:
        result = asyncio.run(result)

    assert set(result.query_results[0]["retrieved"][0]) == {
        "rank",
        "procedure_id",
        "title",
        "category",
        "tags",
        "score",
        "relevant",
    }

    overall = result.overall_metrics
    assert overall["num_queries"] == 3
    assert overall["hit_at_k"]["1"] == 1 / 3
    assert overall["hit_at_k"]["3"] == 2 / 3
    assert overall["mrr"] == (1.0 + 0.5 + 0.0) / 3

    official = result.official_metrics
    assert official["num_queries"] == 3
    assert official["Recall@1"] == 1 / 3
    assert official["Recall@5"] == 2 / 3
    assert official["NDCG@10"] == (1.0 + 1 / math.log2(3) + 0.0) / 3
    assert official["MRR@20"] == (1.0 + 0.5) / 3
    assert official["prediction_coverage"] == 1.0

    # Domain stratification: legal = {q2 hit@rank2, q3 miss}.
    legal = result.domain_stratified_metrics["legal"]
    assert legal["num_queries"] == 2
    assert legal["hit_at_k"]["1"] == 0.0
    assert legal["hit_at_k"]["3"] == 0.5
    assert set(result.domain_stratified_metrics) == {"software", "legal"}

    # Official-format prediction file round-trips into the scorer unchanged.
    ranked = {
        str(q.query_id): [r["procedure_id"] for r in qr["retrieved"]]
        for q, qr in zip(queries, result.query_results)
    }
    preds = tmp_path / "preds.jsonl"
    write_official_predictions_jsonl(preds, ranked)
    lines = [json.loads(line) for line in preds.read_text().splitlines()]
    assert {line["query_id"] for line in lines} == {"q1", "q2", "q3"}
    assert lines[1]["retrieved_skill_ids"] == ["n1", "gold-b", "n2"]
    rescored = compute_official_metrics(
        {line["query_id"]: line["retrieved_skill_ids"] for line in lines},
        {q.query_id: set(q.relevant_skill_ids) for q in queries},
    )
    for key in ("Recall@1", "Recall@5", "NDCG@10", "MRR@20"):
        assert rescored[key] == official[key]


def test_install_query_instruction_precedence(monkeypatch) -> None:
    """Pre-store-init env pin: explicit value wins, else the official constant."""
    from benchmark.skill_reason_bench.adapter import SKILL_REASON_QUERY_INSTRUCTION
    from benchmark.skill_reason_bench.run_retrieval import _install_query_instruction

    monkeypatch.delenv("VECTOR_EMBEDDING_QUERY_INSTRUCTION", raising=False)
    assert _install_query_instruction() == SKILL_REASON_QUERY_INSTRUCTION
    assert (
        os.environ["VECTOR_EMBEDDING_QUERY_INSTRUCTION"]
        == SKILL_REASON_QUERY_INSTRUCTION
    )

    monkeypatch.setenv("VECTOR_EMBEDDING_QUERY_INSTRUCTION", "custom instruct")
    assert _install_query_instruction() == "custom instruct"
