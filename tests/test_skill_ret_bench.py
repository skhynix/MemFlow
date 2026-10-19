# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the SkillRet benchmark harness."""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from benchmark.skill_ret_bench.adapter import (
    normalize_skill_ret_record,
    seed_skill_ret_corpus,
    skill_record_to_procedure,
)
from memflow.models import skill_search_text


def test_skillret_search_text_keeps_description_and_full_body() -> None:
    raw = {
        "id": "skill-id",
        "name": "commit-craft",
        "description": "Split code changes into coherent commits.",
        "skill_md": "---\nname: commit-craft\n---\n\n# Commit Craft\n\nUse git add.",
        "major": "Software Engineering",
        "sub": "Development",
    }

    record = normalize_skill_ret_record(raw)
    procedure = skill_record_to_procedure(record, user_id="benchmark")
    text = skill_search_text(procedure)

    assert record.content == raw["skill_md"]
    assert procedure.metadata["skill"]["name"] == raw["name"]
    assert procedure.metadata["skill"]["description"] == raw["description"]
    assert raw["description"] in text
    assert "Use git add." in text


def test_skillret_record_accepts_body_compatibility_alias() -> None:
    raw = {
        "id": "skill-id",
        "name": "commit-craft",
        "description": "Split commits.",
        "body": "# Commit Craft\n\nUse git add.",
    }

    record = normalize_skill_ret_record(raw)

    assert record.content == raw["body"]


def test_skillret_record_falls_back_from_empty_skill_md() -> None:
    raw = {
        "id": "skill-id",
        "name": "commit-craft",
        "description": "Split commits.",
        "skill_md": "",
    }

    record = normalize_skill_ret_record(raw)

    assert record.content == raw["description"]


def test_integrated_runner_awaits_corpus_seeding(monkeypatch, tmp_path) -> None:
    from benchmark.skill_ret_bench import run_skill_ret_bench as runner

    awaited = False

    async def fake_seed_skill_ret_corpus(**kwargs):
        nonlocal awaited
        awaited = True
        return SimpleNamespace(
            active_corpus_size=3,
            to_dict=lambda: {"num_seeded": 3},
        )

    args = Namespace(
        user_id="benchmark",
        k_values=[1],
        query_bank_path=None,
        corpus_path=tmp_path / "skills.jsonl",
        results_dir=tmp_path,
        results_filename="seed-result",
        seed_only=True,
        clear_existing=False,
        max_queries=None,
    )
    monkeypatch.setattr(runner, "_load_env_file", lambda *_: None)
    monkeypatch.setattr(runner, "_parse_args", lambda: args)
    monkeypatch.setattr(runner, "MemFlow", object)
    monkeypatch.setattr(runner, "seed_skill_ret_corpus", fake_seed_skill_ret_corpus)

    runner.main()

    assert awaited
    assert (tmp_path / "seed-result.json").exists()


def test_runners_default_to_official_evaluation_skill_split() -> None:
    from benchmark.skill_ret_bench import run_seeding, run_skill_ret_bench

    expected_suffix = "data/SKILLRET/data/skills/test.jsonl"

    assert run_seeding.DEFAULT_CORPUS_PATH.as_posix().endswith(expected_suffix)
    assert run_skill_ret_bench.DEFAULT_CORPUS_PATH.as_posix().endswith(expected_suffix)


class _RecordingStore:
    """Store double that records which bulk primitives the seeder uses."""

    def __init__(self, existing_ids: list[str]) -> None:
        self.existing_ids = set(existing_ids)
        self.list_called = False
        self.list_ids_calls = 0
        self.delete_many_calls: list[list[str]] = []
        self.deleted: list[str] = []

    def list(self, user_id: str | None = None):
        self.list_called = True
        raise AssertionError("seed must not fetch full procedure payloads via list()")

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


class _RecordingMemFlow:
    def __init__(self, store: _RecordingStore) -> None:
        self.store = store
        self.added: list = []

    async def add_async(self, procedure, max_workers: int = 4) -> dict:
        procs = procedure if isinstance(procedure, list) else [procedure]
        self.added.extend(procs)
        return {"num_seeded": len(procs), "num_skipped": 0}


def _write_corpus(tmp_path):
    corpus_path = tmp_path / "skills.jsonl"
    records = [
        {"id": "s_001", "name": "Alpha", "skill_md": "body a"},
        {"id": "s_002", "name": "Beta", "skill_md": "body b"},
    ]
    corpus_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return corpus_path


def test_seed_clear_existing_uses_bulk_primitives(tmp_path) -> None:
    """--clear-existing must not pull full payloads nor delete per id.

    Per-id deletion costs two serial round trips per document (get +
    delete). The bulk path is list_ids() + one batched delete_many().
    """
    corpus_path = _write_corpus(tmp_path)
    store = _RecordingStore(existing_ids=["s_001", "unrelated"])
    memflow = _RecordingMemFlow(store)

    stats = asyncio.run(
        seed_skill_ret_corpus(
            memflow=memflow,
            user_id="bench-user",
            corpus_path=corpus_path,
            clear_existing=True,
        )
    )

    assert store.list_called is False
    assert store.list_ids_calls == 1
    # Only corpus ids that exist are deleted; foreign ids stay untouched.
    assert store.deleted == ["s_001"]
    assert stats.num_deleted == 1
    assert stats.num_seeded == 2
    assert [p.id for p in memflow.added] == ["s_001", "s_002"]


def test_seed_reuse_path_uses_list_ids(tmp_path) -> None:
    """Reuse listing must also use list_ids, not full-payload list()."""
    corpus_path = _write_corpus(tmp_path)
    store = _RecordingStore(existing_ids=["s_001"])
    memflow = _RecordingMemFlow(store)

    stats = asyncio.run(
        seed_skill_ret_corpus(
            memflow=memflow,
            user_id="bench-user",
            corpus_path=corpus_path,
        )
    )

    assert store.list_called is False
    assert stats.num_reused == 1
    assert stats.num_seeded == 1
    assert [p.id for p in memflow.added] == ["s_002"]


# --- Explicit search-mode path (unified run_retrieval, no monkey-patch) ---


def _make_adapter(**kwargs):
    from benchmark.skill_ret_bench.adapter import MemFlowSkillRetAdapter

    calls = []

    class FakeStore:
        def search(self, *a, **kw):  # pragma: no cover - batch fallback only
            raise AssertionError("store.search should not be used")

    class FakeMemFlow:
        store = FakeStore()

        def search(self, query, **kwargs):
            calls.append((query, kwargs))
            return []

    adapter = MemFlowSkillRetAdapter(
        memflow=FakeMemFlow(),
        user_id="benchmark",
        corpus_size=1,
        backend="qdrant",
        **kwargs,
    )
    return adapter, calls


def test_adapter_without_search_fn_uses_memflow_search() -> None:
    adapter, calls = _make_adapter()

    adapter.retrieve("deploy app", k=3, exclude_procedure_ids={"x"})

    assert calls == [
        ("deploy app", {"user_id": "benchmark", "top_k": 4, "kind": "skill"})
    ]
    info = adapter.get_system_info()
    assert info["method"] == "memflow.search"
    assert "mode" not in info


def test_adapter_with_search_fn_bypasses_memflow_search() -> None:
    seen = []

    def search_fn(query, user_id=None, top_k=5, kind="skill"):
        seen.append((query, top_k))
        return []

    adapter, calls = _make_adapter(search_fn=search_fn, mode_label="hybrid")

    adapter.retrieve("deploy app", k=3)

    assert seen == [("deploy app", 3)]
    assert calls == []
    info = adapter.get_system_info()
    assert info["method"] == "search_fn:hybrid"
    assert info["mode"] == "hybrid"


def test_adapter_batch_falls_back_to_per_query_when_search_fn_injected() -> None:
    def search_fn(query, user_id=None, top_k=5, kind="skill"):
        return []

    adapter, calls = _make_adapter(search_fn=search_fn)

    results = adapter.retrieve_batch([("q1", set()), ("q2", {"x"})], k=2)

    assert results == [[], []]
    assert calls == []


def test_adapter_batch_fallback_applies_exclusions_and_overfetch() -> None:
    from memflow.models import Procedure, SearchResult

    def _hit(pid: str) -> SearchResult:
        return SearchResult(
            procedure=Procedure(id=pid, title=pid, content="x"), score=1.0
        )

    def search_fn(query, user_id=None, top_k=5, kind="skill"):
        return [_hit("x"), _hit("a"), _hit("b"), _hit("c")]

    adapter, _ = _make_adapter(search_fn=search_fn)

    results = adapter.retrieve_batch([("q1", {"x"})], k=2)

    # Excluded id dropped; fetch_k = k + len(excluded) = 3 kept enough
    # survivors after filtering to fill k.
    assert [[r.procedure_id for r in results[0]]] == [["a", "b"]]


def _mode_args(**overrides):
    base = dict(
        sparse_top_k=None,
        dense_top_k=None,
        prefetch_top_k=200,
        rrf_weights="0.6,0.4",
        rrf_k=60,
        rrf_top_k=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_parse_rrf_weights_accepts_default_and_rejects_bad_input() -> None:
    from benchmark.skill_ret_bench.run_retrieval import _parse_rrf_weights

    assert _parse_rrf_weights("0.6,0.4") == [0.6, 0.4]
    with pytest.raises(ValueError):
        _parse_rrf_weights("0.6")
    with pytest.raises(ValueError):
        _parse_rrf_weights("0.6,-0.4")


def test_resolve_mode_params_defaults_rrf_top_k_to_serving_value(
    monkeypatch,
) -> None:
    from benchmark.skill_ret_bench.run_retrieval import _resolve_mode_params

    monkeypatch.delenv("MEMFLOW_HYBRID_RRF_TOP_K", raising=False)
    params = _resolve_mode_params(_mode_args())

    assert params["sparse_top_k"] == 200
    assert params["dense_top_k"] == 200
    assert params["rrf_weights"] == [0.6, 0.4]
    assert params["rrf_k"] == 60
    # benchmark default == serving default (100) — numbers are
    # serving-comparable without extra flags
    assert params["rrf_top_k"] == 100
    assert params["rrf_top_k_source"] == "default"

    monkeypatch.setenv("MEMFLOW_HYBRID_RRF_TOP_K", "150")
    from_env = _resolve_mode_params(_mode_args())
    assert from_env["rrf_top_k"] == 150
    assert from_env["rrf_top_k_source"] == "env"

    explicit = _resolve_mode_params(
        _mode_args(rrf_top_k=300, dense_top_k=30),
    )
    assert explicit["rrf_top_k"] == 300
    assert explicit["rrf_top_k_source"] == "cli"
    assert explicit["dense_top_k"] == 30
    assert explicit["sparse_top_k"] == 200


def test_mode_search_fn_calls_search_hybrid_and_truncates() -> None:
    from benchmark.skill_ret_bench.run_retrieval import (
        _make_mode_search_fn,
        _resolve_mode_params,
    )

    seen = {}

    class FakeStore:
        def search_hybrid(self, **kwargs):
            seen.update(kwargs)
            return list(range(100))

    memflow = SimpleNamespace(store=FakeStore())
    params = _resolve_mode_params(_mode_args())
    search_fn = _make_mode_search_fn(memflow, "sparse", params)

    out = search_fn("q", user_id="u", top_k=10, kind="skill")

    assert out == list(range(10))
    assert seen["mode"] == "sparse"
    assert seen["rrf_top_k"] == 100  # serving default, max(100, top_k=10)
    assert seen["rrf_weights"] == [0.6, 0.4]
    assert seen["sparse_top_k"] == 200

    # top_k beyond rrf_top_k raises depth per query, like serving
    seen.clear()
    search_fn("q", user_id="u", top_k=150, kind="skill")
    assert seen["rrf_top_k"] == 150


def test_results_path_appends_mode_suffix(tmp_path) -> None:
    from benchmark.skill_ret_bench.run_retrieval import _results_path

    auto = _results_path(tmp_path, None, "hybrid")
    assert auto.name.startswith("skillret_retrieval_")
    assert auto.name.endswith("_hybrid.json")

    named = _results_path(tmp_path, "myrun.json", "dense")
    assert named.name == "myrun_dense.json"

    named_noext = _results_path(tmp_path, "myrun", None)
    assert named_noext.name == "myrun.json"


# --- Runner main() warnings (H1 sync bypass, M3 stage cap, mode dedupe) ---


def _runner_args(tmp_path, **overrides):
    base = dict(
        user_id="benchmark",
        k_values=[10, 30],
        max_queries=1,
        results_dir=tmp_path,
        results_filename="run",
        sync=False,
        max_concurrency=4,
        mode=None,
        query_bank_path=tmp_path / "queries.jsonl",
        # --mode knobs (unused unless mode is set)
        prefetch_top_k=200,
        sparse_top_k=None,
        dense_top_k=None,
        rrf_weights="0.6,0.4",
        rrf_k=60,
        rrf_top_k=None,
    )
    base.update(overrides)
    return Namespace(**base)


def _install_runner_mocks(monkeypatch, tmp_path, args) -> None:
    from benchmark.skill_ret_bench import run_retrieval as runner

    class FakeStore:
        supports_hybrid = True

        def count(self, user_id=None):
            return 2

    class FakeMemFlow:
        store = FakeStore()

        def __init__(self):
            pass

    eval_result = SimpleNamespace(
        overall_metrics={},
        category_stratified_metrics={},
        query_results=[],
    )

    (tmp_path / "queries.jsonl").write_text("{}", encoding="utf-8")

    async def fake_eval_async(**kwargs):
        return eval_result

    monkeypatch.setattr(runner, "_load_env_file", lambda *_: None)
    monkeypatch.setattr(runner, "_parse_args", lambda: args)
    monkeypatch.setattr(runner, "MemFlow", FakeMemFlow)
    monkeypatch.setattr(runner, "count_query_bank_records", lambda *_: 1)
    monkeypatch.setattr(runner, "load_skill_ret_query_bank", lambda **_: ["q1"])
    monkeypatch.setattr(runner, "evaluate_skill_ret_queries", lambda **_: eval_result)
    monkeypatch.setattr(runner, "evaluate_skill_ret_queries_async", fake_eval_async)
    for key in (
        "MEMFLOW_HYBRID_SEARCH_ENABLED",
        "MEMFLOW_RERANKER_PIPELINE",
        "MEMFLOW_RERANKER_LLM_TOP_K",
        "MEMFLOW_RERANKER_CE_TOP_K",
    ):
        monkeypatch.delenv(key, raising=False)


def test_sync_path_warns_when_env_config_is_bypassed(
    monkeypatch, tmp_path, capsys
) -> None:
    # H1: --sync runs plain dense batch retrieval; when env config says
    # hybrid/rerank, main() must warn loudly and record executed_path.
    from benchmark.skill_ret_bench import run_retrieval as runner

    _install_runner_mocks(monkeypatch, tmp_path, _runner_args(tmp_path, sync=True))
    monkeypatch.setenv("MEMFLOW_HYBRID_SEARCH_ENABLED", "on")
    monkeypatch.setenv("MEMFLOW_RERANKER_PIPELINE", "llm")

    runner.main()

    out = capsys.readouterr().out
    assert "BYPASSES" in out
    assert "hybrid search" in out and "reranker" in out
    result = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert result["settings"]["executed_path"] == "dense_batch:bypassed=hybrid+reranker"


def test_no_sync_no_bypass_warning(monkeypatch, tmp_path, capsys) -> None:
    from benchmark.skill_ret_bench import run_retrieval as runner

    _install_runner_mocks(monkeypatch, tmp_path, _runner_args(tmp_path, sync=False))
    monkeypatch.setenv("MEMFLOW_HYBRID_SEARCH_ENABLED", "on")
    monkeypatch.setenv("MEMFLOW_RERANKER_PIPELINE", "llm")

    runner.main()

    out = capsys.readouterr().out
    assert "BYPASSES" not in out
    result = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert "executed_path" not in result["settings"]


def test_stage_top_k_below_max_k_warns(monkeypatch, tmp_path, capsys) -> None:
    # M3: default LLM stage Top-K (10) below max(--k-values)=30 caps R@30 at
    # the stage output — main() must warn.
    from benchmark.skill_ret_bench import run_retrieval as runner

    _install_runner_mocks(
        monkeypatch,
        tmp_path,
        _runner_args(tmp_path, k_values=[10, 30]),
    )
    monkeypatch.setenv("MEMFLOW_RERANKER_PIPELINE", "llm")

    runner.main()

    out = capsys.readouterr().out
    assert "LLM-stage Top-K=10" in out
    assert "R@K for K>10 measures the stage cap" in out


def test_stage_top_k_not_below_max_k_no_warning(monkeypatch, tmp_path, capsys) -> None:
    from benchmark.skill_ret_bench import run_retrieval as runner

    _install_runner_mocks(
        monkeypatch,
        tmp_path,
        _runner_args(tmp_path, k_values=[10, 30]),
    )
    monkeypatch.setenv("MEMFLOW_RERANKER_PIPELINE", "llm")
    monkeypatch.setenv("MEMFLOW_RERANKER_LLM_TOP_K", "100")

    runner.main()

    out = capsys.readouterr().out
    assert "final-stage Top-K" not in out


def test_duplicate_modes_deduped_one_results_file(monkeypatch, tmp_path) -> None:
    # L5: `--mode hybrid hybrid` must not silently run/overwrite twice.
    from benchmark.skill_ret_bench import run_retrieval as runner

    _install_runner_mocks(
        monkeypatch,
        tmp_path,
        _runner_args(tmp_path, mode=["hybrid", "hybrid"]),
    )

    runner.main()

    assert list(tmp_path.glob("run_hybrid*.json")) == [tmp_path / "run_hybrid.json"]


def test_reranker_effective_resolves_fallbacks(monkeypatch) -> None:
    from benchmark.skill_ret_bench.run_retrieval import _reranker_effective

    monkeypatch.setenv("MEMFLOW_RERANKER_PIPELINE", "LLM")
    monkeypatch.delenv("MEMFLOW_RERANKER_LLM_TOP_K", raising=False)
    monkeypatch.delenv("MEMFLOW_RERANKER_CE_TOP_K", raising=False)
    monkeypatch.delenv("MEMFLOW_RERANKER_SCOPE", raising=False)
    monkeypatch.delenv("MEMFLOW_RERANKER_LLM_SCOPE", raising=False)
    monkeypatch.delenv("MEMFLOW_RERANKER_CE_SCOPE", raising=False)
    monkeypatch.delenv("RERANKER_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    monkeypatch.delenv("RERANKER_LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    effective = _reranker_effective()

    assert effective["pipeline"] == "llm"
    assert effective["llm_top_k"] == 10
    assert effective["llm_scope"] == "meta"
    assert effective["ce_scope"] == "full"
    assert effective["llm_provider"] == "openai-compatible"
    assert effective["llm_model"] is None
    # openai-compatible without any model -> snapshot warning recorded
    assert "llm_warning" in effective
