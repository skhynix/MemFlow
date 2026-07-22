# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import io
import json
import time
from dataclasses import replace

import pytest

import memflow.claude_hook as claude_hook_module
import memflow.claude_session_state as claude_session_state_module
import memflow.skill_context as skill_context_module
import memflow.skills as skills_module
from memflow.claude_hook import (
    ADAPTER_NAME,
    build_skill_context_request,
    default_manager_factory,
    load_hook_config,
    parse_hook_input,
    run_hook,
)
from memflow.claude_hook import (
    main as claude_hook_main,
)
from memflow.manager import MemFlow
from memflow.models import Procedure, SearchResult
from memflow.skill_context import (
    ContextRenderer,
    SkillCandidate,
    SkillContextRequest,
    SkillContextSelector,
    selected_skill_metadata,
)
from memflow.skills import load_skill
from memflow.store import EmulatedStore


def _write_skill(root, text: str) -> None:
    root.mkdir()
    (root / "SKILL.md").write_text(text, encoding="utf-8")


def _manager_with_skill(
    tmp_path,
    fake_llm,
    *,
    name: str = "commit-craft",
    body: str = "# Commit Craft\n\nSplit commits into reviewable units.\n",
    description: str = "Split code changes into coherent commits.",
    trust_state: str = "trusted",
):
    root = tmp_path / name
    _write_skill(
        root,
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "tags: [git, commits]\n"
        "aliases: [patch series]\n"
        "file_patterns: ['*.py']\n"
        "tools: [git]\n"
        "---\n"
        f"{body}",
    )
    manager = MemFlow(llm=fake_llm, store=EmulatedStore(), use_env=False)
    manager.add_skill(root, trust_state=trust_state)
    return manager


def _hook_input(
    prompt: str,
    *,
    event: str = "UserPromptSubmit",
    session_id: str = "session-123",
) -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "/tmp/transcript.jsonl",
            "cwd": "/work/project",
            "hook_event_name": event,
            "prompt": prompt,
        }
    )


def _config_path(tmp_path, **overrides):
    config = {
        "memflow": {"user_id": "default"},
        "retrieval": {
            "top_k": 3,
            "max_top_k": 5,
            "candidate_k": 10,
            "min_score": 0.1,
            "include_cwd_in_query": True,
        },
        "rendering": {
            "max_chars": 4000,
            "hard_max_chars": 5000,
            "max_chars_per_skill": 2500,
        },
        "logging": {
            "path": str(tmp_path / "hook-audit.jsonl"),
            "record_raw_prompt": False,
            "record_skill_body": False,
        },
    }
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(config.get(section), dict):
            config[section].update(values)
        else:
            config[section] = values
    path = tmp_path / "claude-hook.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _audit_rows(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


_DEFAULT_METADATA_SOURCE = object()


def _renderer_config(
    *,
    top_k: int = 3,
    max_chars: int = 20_000,
    max_chars_per_skill: int = 10_000,
    rendering_format: str = "selected_skills_xml_v1",
):
    return {
        "retrieval": {"top_k": top_k},
        "rendering": {
            "max_chars": max_chars,
            "hard_max_chars": max_chars,
            "max_chars_per_skill": max_chars_per_skill,
            "format": rendering_format,
        },
    }


def _renderer_candidate(
    *,
    name: str = "renderer-skill",
    body: str = "# Renderer Skill\n\nRender the selected skill.\n",
    procedure_id: str = "stable-renderer-id",
    source_path: str | None = "/stored/renderer-skill/SKILL.md",
    metadata_source_path=_DEFAULT_METADATA_SOURCE,
    description: str = "Use this renderer skill.",
    headings: tuple[str, ...] = ("Renderer Skill",),
    sha256: str = "raw-skill-sha",
    score: float = 0.9,
    reason: str = "matched_prompt_via_memflow_skill_search",
    provenance: str = "local",
    trust_mode: str = "instruction",
    trust_state: str = "trusted",
) -> SkillCandidate:
    if metadata_source_path is _DEFAULT_METADATA_SOURCE:
        metadata_source_path = source_path
    skill = {
        "name": name,
        "description": description,
        "sha256": sha256,
        "frontmatter": {
            "name": name,
            "description": description,
        },
        "aliases": [],
        "file_patterns": [],
        "tools": [],
    }
    if metadata_source_path is not None:
        skill["source_path"] = metadata_source_path
    procedure = Procedure(
        id=procedure_id,
        title=name,
        content=body,
        kind="skill",
        source_path=source_path,
        metadata={
            "skill": skill,
            "index": {
                "body_offset": 0,
                "headings": [{"text": heading} for heading in headings],
            },
        },
    )
    return SkillCandidate(
        procedure=procedure,
        score=score,
        reason=reason,
        provenance=provenance,
        trust_mode=trust_mode,
        trust_state=trust_state,
        warnings=(),
    )


def _render_target(
    candidate: SkillCandidate,
    *,
    config=None,
    trace_id: str = "trace-one",
    rank: int = 1,
):
    config = copy.deepcopy(config or _renderer_config())
    config["retrieval"]["top_k"] = max(config["retrieval"]["top_k"], rank)
    fillers = [
        _renderer_candidate(
            name=f"filler-{index}",
            procedure_id=f"filler-{index}",
            source_path=f"/stored/filler-{index}/SKILL.md",
            body=f"# Filler {index}\n\nFiller content.\n",
            headings=(f"Filler {index}",),
        )
        for index in range(1, rank)
    ]
    result = ContextRenderer(config).render(
        [*fillers, candidate],
        trace_id=trace_id,
    )
    assert len(result.skills) == rank
    return result, result.skills[-1]


def _candidate_with_skill_metadata(candidate: SkillCandidate, **updates):
    metadata = copy.deepcopy(candidate.procedure.metadata)
    metadata["skill"].update(updates)
    procedure = replace(candidate.procedure, metadata=metadata)
    return replace(candidate, procedure=procedure)


class _StaticSkillManager:
    def __init__(self, candidates: list[SkillCandidate]) -> None:
        self.candidates = candidates
        self.search_calls = 0

    def search_skills(self, query, user_id=None, top_k=5):
        del query, user_id
        self.search_calls += 1
        return [
            SearchResult(procedure=candidate.procedure, score=candidate.score)
            for candidate in self.candidates[:top_k]
        ]

    def get_skill(self, id_or_name, include_content=True):
        del id_or_name, include_content
        raise AssertionError("complete static results must not be hydrated")


@pytest.mark.parametrize(
    (
        "source_path",
        "metadata_source_path",
        "procedure_id",
        "expected_identity",
    ),
    [
        (
            "/stored/primary/SKILL.md",
            "/stored/metadata/SKILL.md",
            "stable-id",
            "path:/stored/primary/SKILL.md",
        ),
        (
            None,
            "/stored/metadata/SKILL.md",
            "stable-id",
            "path:/stored/metadata/SKILL.md",
        ),
        (None, None, "stable-id", "id:stable-id"),
        (123, None, "stable-id", "id:stable-id"),
        (None, 123, 456, None),
        (None, None, "", None),
    ],
)
def test_renderer_uses_stable_activation_identity(
    source_path,
    metadata_source_path,
    procedure_id,
    expected_identity,
):
    candidate = _renderer_candidate(
        source_path=source_path,
        metadata_source_path=metadata_source_path,
        procedure_id=procedure_id,
    )

    result, rendered = _render_target(candidate)

    assert rendered.identity == expected_identity
    metadata = selected_skill_metadata(rendered)
    assert metadata["identity"] == expected_identity
    assert metadata["activation_fingerprint"] == rendered.activation_fingerprint
    if expected_identity is None:
        assert rendered.activation_fingerprint is None
        assert "activation_identity_unavailable" in result.warnings
    else:
        assert len(rendered.activation_fingerprint or "") == 64
        assert "activation_identity_unavailable" not in result.warnings


def test_renderer_uses_metadata_path_when_primary_path_is_malformed():
    candidate = _renderer_candidate(
        source_path=123,
        metadata_source_path="/stored/metadata/SKILL.md",
        procedure_id="stable-id",
    )

    _result, rendered = _render_target(candidate)

    assert rendered.identity == "path:/stored/metadata/SKILL.md"
    assert 'source_path="/stored/metadata/SKILL.md"' in rendered.xml
    assert selected_skill_metadata(rendered)["source_path"] == (
        "/stored/metadata/SKILL.md"
    )


def test_pathless_activation_fingerprint_tracks_a_b_a_transitions():
    rendered = []
    for body, raw_sha in (
        ("# Stable\n\nPayload A\n", "sha-a-first"),
        ("# Stable\n\nPayload B\n", "sha-b"),
        ("# Stable\n\nPayload A\n", "sha-a-second"),
    ):
        candidate = _renderer_candidate(
            procedure_id="stable-pathless-id",
            source_path=None,
            metadata_source_path=None,
            body=body,
            headings=("Stable",),
            sha256=raw_sha,
        )
        _result, skill = _render_target(candidate)
        rendered.append(skill)

    assert [skill.identity for skill in rendered] == [
        "id:stable-pathless-id",
        "id:stable-pathless-id",
        "id:stable-pathless-id",
    ]
    fingerprints = [skill.activation_fingerprint for skill in rendered]
    assert fingerprints[0] != fingerprints[1]
    assert fingerprints[1] != fingerprints[2]
    assert fingerprints[0] == fingerprints[2]


@pytest.mark.parametrize(
    "volatile_field",
    [
        "rank",
        "score",
        "reason",
        "provenance",
        "trust_state",
        "trace_id",
        "timestamps",
        "prompt",
        "raw_sha",
        "rendering_format",
    ],
)
def test_activation_fingerprint_excludes_volatile_fields(volatile_field):
    base_candidate = _renderer_candidate()
    base_config = _renderer_config()
    base_trace = "trace-one"
    changed_candidate = base_candidate
    changed_config = copy.deepcopy(base_config)
    changed_trace = base_trace
    changed_rank = 1

    if volatile_field == "rank":
        changed_rank = 2
    elif volatile_field == "score":
        changed_candidate = replace(base_candidate, score=0.123)
    elif volatile_field == "reason":
        changed_candidate = replace(base_candidate, reason="another reason")
    elif volatile_field == "provenance":
        changed_candidate = replace(base_candidate, provenance="remote")
    elif volatile_field == "trust_state":
        changed_candidate = replace(base_candidate, trust_state="unknown")
    elif volatile_field == "trace_id":
        changed_trace = "trace-two"
    elif volatile_field == "timestamps":
        procedure = replace(
            base_candidate.procedure,
            created_at="1999-01-01T00:00:00",
            updated_at="2099-01-01T00:00:00",
        )
        changed_candidate = replace(base_candidate, procedure=procedure)
    elif volatile_field == "prompt":
        changed_config["prompt"] = "an unrelated user prompt"
    elif volatile_field == "raw_sha":
        changed_candidate = _candidate_with_skill_metadata(
            base_candidate,
            sha256="a-new-raw-skill-sha",
        )
    elif volatile_field == "rendering_format":
        changed_config["rendering"]["format"] = "arbitrary-config-format"

    _base_result, base = _render_target(
        base_candidate,
        config=base_config,
        trace_id=base_trace,
    )
    _changed_result, changed = _render_target(
        changed_candidate,
        config=changed_config,
        trace_id=changed_trace,
        rank=changed_rank,
    )

    assert changed.identity == base.identity
    assert changed.activation_fingerprint == base.activation_fingerprint
    if volatile_field == "rank":
        assert changed.rank != base.rank
        assert changed.xml != base.xml


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("identity", "id:changed"),
        ("name", "changed-name"),
        ("source_path", "/changed/SKILL.md"),
        ("trust_mode", "data"),
        ("when_to_use", "Use for a changed task."),
        ("headings", ("Changed heading",)),
        ("content", "Changed content"),
        ("truncated", True),
    ],
)
def test_activation_fingerprint_includes_every_emitted_semantic_field(
    field,
    changed_value,
):
    semantics = skill_context_module._RenderedSkillSemantics(
        identity="id:stable",
        name="semantic-skill",
        source_path="",
        trust_mode="instruction",
        when_to_use="Use for semantic tests.",
        headings=("Semantic heading",),
        content="Semantic content",
        truncated=False,
    )

    baseline = skill_context_module._activation_fingerprint(semantics)
    changed = skill_context_module._activation_fingerprint(
        replace(semantics, **{field: changed_value})
    )

    assert changed != baseline


def test_activation_fingerprint_uses_hard_coded_renderer_version(monkeypatch):
    candidate = _renderer_candidate()
    _result, before = _render_target(candidate)

    monkeypatch.setattr(
        skill_context_module,
        "_ACTIVATION_FORMAT",
        "selected_skills_xml_v1.activation_v2",
    )
    _result, after = _render_target(candidate)

    assert after.xml == before.xml
    assert after.activation_fingerprint != before.activation_fingerprint


def test_activation_fingerprint_matches_canonical_unicode_semantics():
    name = '技能 <& "quote"'
    source_path = "/stored/技能<&/SKILL.md"
    description = "使用 <xml> & data"
    heading = "标题 <&"
    body = "正文 <tag> & café 😀"
    candidate = _renderer_candidate(
        name=name,
        source_path=source_path,
        description=description,
        headings=(heading,),
        body=body,
    )

    _result, rendered = _render_target(candidate)

    payload = {
        "format": "selected_skills_xml_v1.activation_v1",
        "identity": f"path:{source_path}",
        "name": name,
        "source_path": source_path,
        "trust_mode": "instruction",
        "when_to_use": description,
        "headings": (heading,),
        "content": body,
        "truncated": False,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert rendered.activation_fingerprint == expected
    assert 'name="技能 &lt;&amp; &quot;quote&quot;"' in rendered.xml
    assert 'source_path="/stored/技能&lt;&amp;/SKILL.md"' in rendered.xml
    assert "<heading>标题 &lt;&amp;</heading>" in rendered.xml
    assert "正文 &lt;tag&gt; &amp; café 😀" in rendered.xml


def test_renderer_keeps_unfingerprintable_unicode_skill_in_baseline():
    candidate = _renderer_candidate(
        body="# Unfingerprintable\n\nPayload with an unpaired surrogate: \ud800\n",
        headings=("Unfingerprintable",),
    )

    result, rendered = _render_target(candidate)

    assert result.skills == (rendered,)
    assert rendered.xml in result.xml
    assert "\ud800" in result.xml
    assert rendered.identity == "path:/stored/renderer-skill/SKILL.md"
    assert rendered.activation_fingerprint is None
    assert selected_skill_metadata(rendered)["activation_fingerprint"] is None
    assert "activation_fingerprint_unavailable" in result.warnings


def test_empty_content_fallback_keeps_unfingerprintable_skill(monkeypatch):
    candidate = _renderer_candidate(
        name="fallback-\ud800",
        body="fallback body " * 500,
        headings=("Fallback",),
    )
    budget = 1_000
    original_render_skill_xml = skill_context_module._render_skill_xml
    render_calls = 0

    def exhaust_iterative_render_budget(semantics, rank):
        nonlocal render_calls
        render_calls += 1
        if render_calls <= 8:
            return "x" * (budget + 1)
        return original_render_skill_xml(semantics, rank)

    monkeypatch.setattr(
        skill_context_module,
        "_render_skill_xml",
        exhaust_iterative_render_budget,
    )

    render_result = skill_context_module._render_skill_with_budget(
        candidate,
        rank=1,
        budget=budget,
    )

    assert render_result is not None
    rendered, warnings = render_result
    assert render_calls == 9
    assert "fallback-\ud800" in rendered.xml
    assert rendered.identity == "path:/stored/renderer-skill/SKILL.md"
    assert rendered.activation_fingerprint is None
    assert "activation_fingerprint_unavailable" in warnings


def test_renderer_does_not_swallow_fingerprint_base_exceptions(monkeypatch):
    def interrupt_fingerprint(_semantics):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        skill_context_module,
        "_activation_fingerprint",
        interrupt_fingerprint,
    )

    with pytest.raises(KeyboardInterrupt):
        _render_target(_renderer_candidate())


def test_activation_fingerprint_uses_only_eight_emitted_headings():
    headings = tuple(f"Heading {index}" for index in range(1, 10))
    baseline_candidate = _renderer_candidate(
        body="Body without parsed headings.",
        headings=headings,
    )
    ninth_changed = _renderer_candidate(
        body="Body without parsed headings.",
        headings=(*headings[:8], "Changed ninth heading"),
    )
    eighth_changed = _renderer_candidate(
        body="Body without parsed headings.",
        headings=(*headings[:7], "Changed eighth heading", headings[8]),
    )

    _result, baseline = _render_target(baseline_candidate)
    _result, after_ninth = _render_target(ninth_changed)
    _result, after_eighth = _render_target(eighth_changed)

    assert baseline.activation_fingerprint == after_ninth.activation_fingerprint
    assert baseline.xml == after_ninth.xml
    assert "<heading>Heading 8</heading>" in baseline.xml
    assert "<heading>Heading 9</heading>" not in baseline.xml
    assert baseline.activation_fingerprint != after_eighth.activation_fingerprint


def test_activation_fingerprint_represents_fallback_heading_semantics():
    fallback_candidate = _renderer_candidate(
        body="Body without parsed headings.",
        headings=(),
    )
    explicit_equivalent = _renderer_candidate(
        body="Body without parsed headings.",
        headings=("No headings indexed.",),
    )
    changed_heading = _renderer_candidate(
        body="Body without parsed headings.",
        headings=("A real heading",),
    )

    _result, fallback = _render_target(fallback_candidate)
    _result, equivalent = _render_target(explicit_equivalent)
    _result, changed = _render_target(changed_heading)

    assert "<heading>No headings indexed.</heading>" in fallback.xml
    assert fallback.xml == equivalent.xml
    assert fallback.activation_fingerprint == equivalent.activation_fingerprint
    assert fallback.activation_fingerprint != changed.activation_fingerprint


def test_activation_fingerprint_ignores_undelivered_raw_tail():
    common_prefix = "P" * 2_000
    config = _renderer_config(max_chars_per_skill=600)
    first_candidate = _renderer_candidate(
        body=f"{common_prefix}TAIL-A",
        sha256="tail-a-sha",
    )
    second_candidate = _renderer_candidate(
        body=f"{common_prefix}TAIL-B",
        sha256="tail-b-sha",
    )

    _result, first = _render_target(first_candidate, config=config)
    _result, second = _render_target(second_candidate, config=config)

    assert 'truncated="true"' in first.xml
    assert first.xml == second.xml
    assert first.activation_fingerprint == second.activation_fingerprint


def test_activation_fingerprint_changes_at_truncation_boundary():
    candidate = _renderer_candidate(
        body="# Boundary\n\n" + "boundary content " * 40,
        headings=("Boundary",),
    )
    _result, full = _render_target(candidate)
    exact_config = _renderer_config(max_chars_per_skill=full.rendered_chars)
    below_config = _renderer_config(max_chars_per_skill=full.rendered_chars - 1)

    _result, exact = _render_target(candidate, config=exact_config)
    _result, below = _render_target(candidate, config=below_config)

    assert 'truncated="false"' in exact.xml
    assert exact.xml == full.xml
    assert exact.activation_fingerprint == full.activation_fingerprint
    assert 'truncated="true"' in below.xml
    assert below.activation_fingerprint != exact.activation_fingerprint


def test_context_renderer_compose_preserves_rendered_subset_and_sparse_ranks():
    candidates = [
        _renderer_candidate(
            name=f"compose-{rank}",
            procedure_id=f"compose-{rank}",
            source_path=f"/stored/compose-{rank}/SKILL.md",
            body=f"# Compose {rank}\n\nContent {rank}.\n",
            headings=(f"Compose {rank}",),
        )
        for rank in range(1, 4)
    ]
    renderer = ContextRenderer(_renderer_config(top_k=3))
    baseline = renderer.render(candidates, trace_id="compose-trace")

    assert renderer.compose(baseline.skills) == baseline.xml
    assert renderer.compose(()) == ""

    mixed_skills = (baseline.skills[0], baseline.skills[2])
    mixed = renderer.compose(mixed_skills)
    assert [skill.rank for skill in mixed_skills] == [1, 3]
    assert baseline.skills[0].xml in mixed
    assert baseline.skills[2].xml in mixed
    assert '<skill rank="1"' in mixed
    assert '<skill rank="2"' not in mixed
    assert '<skill rank="3"' in mixed
    assert mixed.startswith("<selected_skills>\n")
    assert mixed.endswith("</selected_skills>\n")


def test_hook_input_and_config_map_to_skill_context_request(tmp_path):
    prompt = "Please find relevant skills."
    hook_input = parse_hook_input(_hook_input(prompt))
    config_path = _config_path(
        tmp_path,
        memflow={"user_id": "alice", "project_scope": "repo:/work/project"},
    )
    config = load_hook_config(config_path)

    request = build_skill_context_request(hook_input, config)

    assert request == SkillContextRequest(
        prompt=prompt,
        cwd="/work/project",
        agent="claude-code",
        adapter=ADAPTER_NAME,
        session_id="session-123",
        transcript_path="/tmp/transcript.jsonl",
        user_id="alice",
        project_scope="repo:/work/project",
    )
    default_request = build_skill_context_request(hook_input, {"memflow": {}})
    assert default_request.user_id == "default"
    assert default_request.project_scope == "/work/project"


def test_run_hook_uses_skill_context_request_for_query_and_user_id(tmp_path):
    root = tmp_path / "recorded-skill"
    _write_skill(
        root,
        "---\n"
        "name: recorded-skill\n"
        "description: Records hook request boundaries.\n"
        "---\n"
        "# Recorded Skill\n\nUse request boundaries for retrieval.\n",
    )
    procedure = load_skill(root, trust_state="trusted")

    class RecordingManager:
        def __init__(self):
            self.calls = []

        def search_skills(self, query, user_id=None, top_k=5):
            self.calls.append(
                {
                    "query": query,
                    "user_id": user_id,
                    "top_k": top_k,
                }
            )
            return [SearchResult(procedure=procedure, score=0.9)]

        def get_skill(self, id_or_name, include_content=True):
            del id_or_name, include_content
            raise AssertionError("complete search results should not be hydrated")

    manager = RecordingManager()
    config_path = _config_path(tmp_path, memflow={"user_id": "alice"})
    prompt = "Use request boundaries for retrieval."

    output = run_hook(
        _hook_input(prompt),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    response = json.loads(output)
    context = response["hookSpecificOutput"]["additionalContext"]
    assert response["suppressOutput"] is True
    assert '<skill rank="1" name="recorded-skill"' in context
    assert manager.calls == [
        {
            "query": f"{prompt}\nCurrent working directory: /work/project",
            "user_id": "alice",
            "top_k": 10,
        }
    ]


def test_skill_context_selector_filters_dedupes_and_ranks_candidates(tmp_path):
    root = tmp_path / "selector-skill"
    _write_skill(
        root,
        "---\n"
        "name: selector-skill\n"
        "description: Exercise selector policy.\n"
        "---\n"
        "# Selector Skill\n\nselector policy split commits\n",
    )
    procedure = load_skill(root, trust_state="trusted")

    class SearchManager:
        def __init__(self):
            self.calls = []
            self.get_skill_calls = 0

        def search_skills(self, query, user_id=None, top_k=5):
            self.calls.append(
                {
                    "query": query,
                    "user_id": user_id,
                    "top_k": top_k,
                }
            )
            return [
                SearchResult(procedure=procedure, score=0.3),
                SearchResult(procedure=procedure, score=0.8),
                SearchResult(procedure=procedure, score=0.1),
            ]

        def get_skill(self, id_or_name, include_content=True):
            del id_or_name, include_content
            self.get_skill_calls += 1
            raise AssertionError("complete selector results should not be hydrated")

    config = load_hook_config(
        _config_path(
            tmp_path,
            retrieval={
                "candidate_k": 4,
                "min_score": 0.2,
                "include_cwd_in_query": True,
            },
        )
    )
    request = SkillContextRequest(
        prompt="selector policy split commits",
        cwd="/work/project",
        agent="claude-code",
        adapter=ADAPTER_NAME,
        session_id="session-123",
        transcript_path="/tmp/transcript.jsonl",
        user_id="alice",
        project_scope="/work/project",
    )
    manager = SearchManager()

    candidates, warnings = SkillContextSelector(config).select(manager, request)

    assert [candidate.score for candidate in candidates] == [0.8]
    assert candidates[0].procedure.id == procedure.id
    assert warnings == ["filtered_or_deduped_candidates"]
    assert manager.calls == [
        {
            "query": "selector policy split commits\n"
            "Current working directory: /work/project",
            "user_id": "alice",
            "top_k": 4,
        }
    ]
    assert manager.get_skill_calls == 0


def test_valid_hook_input_returns_parseable_claude_json(tmp_path, fake_llm):
    manager = _manager_with_skill(tmp_path, fake_llm)
    config_path = _config_path(tmp_path)
    prompt = "Please split these commits into reviewable patch series."

    output = run_hook(
        _hook_input(prompt),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    response = json.loads(output)
    assert response["suppressOutput"] is True
    hook_output = response["hookSpecificOutput"]
    assert hook_output["hookEventName"] == "UserPromptSubmit"
    context = hook_output["additionalContext"]
    assert context.startswith("<selected_skills>\n")
    assert context.endswith("</selected_skills>\n")
    assert "These local skills were selected for the current user prompt." in context
    assert "Use them only when relevant to this task." in context
    assert '<skill rank="1" name="commit-craft"' in context
    assert 'source_path="' in context
    assert 'trust_mode="instruction"' in context
    assert "<when_to_use>" in context
    assert "<outline>" in context
    assert '<content truncated="false">' in context
    assert "Split commits into reviewable units." in context
    assert "MemFlow" not in context
    assert "trace_id=" not in context
    assert "top_k=" not in context
    assert "catalog_mode=" not in context
    assert "score=" not in context
    assert "sha256=" not in context
    assert "<why>" not in context
    assert "matched_prompt_via_memflow_skill_search" not in context

    audit = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert audit["status"] == "injected"
    assert audit["adapter"] == "claude-code-user-prompt-submit"
    assert audit["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert audit["session_id_hash"] == hashlib.sha256(b"session-123").hexdigest()
    assert audit["trace_id"]
    assert audit["selected_skills"][0]["name"] == "commit-craft"
    assert audit["selected_skills"][0]["sha256"]
    assert audit["selected_skills"][0]["score"] > 0
    assert (
        audit["selected_skills"][0]["reason"]
        == "matched_prompt_via_memflow_skill_search"
    )
    assert audit["selected_skills"][0]["trust_mode"] == "instruction"


def test_below_threshold_results_return_empty_stdout(tmp_path, fake_llm):
    manager = _manager_with_skill(tmp_path, fake_llm)
    config_path = _config_path(tmp_path, retrieval={"min_score": 0.99})

    output = run_hook(
        _hook_input("split commits"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    assert output == ""
    audit = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert audit["status"] == "no_results"
    assert audit["selected_skills"] == []
    assert audit["session_dedupe"]["lifecycle_tracking"] == ("user_prompt_submit_only")


def test_invalid_json_and_non_user_prompt_events_fail_open(tmp_path, fake_llm):
    manager = _manager_with_skill(tmp_path, fake_llm)
    config_path = _config_path(
        tmp_path,
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "unsupported-event-state",
            }
        },
    )

    invalid_output = run_hook(
        "{not json",
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )
    non_prompt_output = run_hook(
        _hook_input("split commits", event="SessionStart"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    assert invalid_output == ""
    assert non_prompt_output == ""
    rows = _audit_rows(tmp_path / "hook-audit.jsonl")
    assert [row["status"] for row in rows] == ["fail_open", "fail_open"]
    assert rows[1]["warnings"] == ["unsupported_hook_event"]
    assert all("session_dedupe" not in row for row in rows)
    assert not (tmp_path / "unsupported-event-state").exists()


def test_memflow_errors_fail_open(tmp_path):
    config_path = _config_path(tmp_path)

    output = run_hook(
        _hook_input("split commits"),
        config_path=config_path,
        manager_factory=lambda _config: (_ for _ in ()).throw(RuntimeError("down")),
    )

    assert output == ""
    audit = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert audit["status"] == "fail_open"
    assert audit["warnings"] == ["RuntimeError"]
    assert audit["session_dedupe"]["lifecycle_tracking"] == ("user_prompt_submit_only")


def test_default_factory_avoids_optional_llm_dependencies(monkeypatch, tmp_path):
    import memflow.manager as manager_module

    def fail_if_llm_factory_is_used(*args, **kwargs):
        del args, kwargs
        raise AssertionError("hook retrieval path should not construct an LLM")

    monkeypatch.setattr(
        manager_module.LLMFactory, "create", fail_if_llm_factory_is_used
    )
    monkeypatch.setenv("MEMFLOW_BACKEND", "emulated")
    config = load_hook_config(tmp_path / "missing-config.json")
    config["memflow"]["env_file"] = str(tmp_path / "missing.env")

    manager = default_manager_factory(config)

    assert isinstance(manager.store, EmulatedStore)
    with pytest.raises(RuntimeError, match="does not support LLM calls"):
        manager.llm.generate([])


def test_retrieval_timeout_fails_open(tmp_path):
    class SlowManager:
        def search_skills(self, query, user_id=None, top_k=5):
            del query, user_id, top_k
            time.sleep(1)
            return []

    config_path = _config_path(tmp_path, retrieval={"timeout_ms": 50})

    output = run_hook(
        _hook_input("split commits"),
        config_path=config_path,
        manager_factory=lambda _config: SlowManager(),
    )

    assert output == ""
    audit = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert audit["status"] == "fail_open"
    assert audit["warnings"] == ["RetrievalTimeoutError"]


def test_manager_initialization_timeout_fails_open(tmp_path):
    class EmptyManager:
        def search_skills(self, query, user_id=None, top_k=5):
            del query, user_id, top_k
            return []

    def slow_factory(_config):
        time.sleep(1)
        return EmptyManager()

    config_path = _config_path(tmp_path, retrieval={"timeout_ms": 50})

    output = run_hook(
        _hook_input("split commits"),
        config_path=config_path,
        manager_factory=slow_factory,
    )

    assert output == ""
    audit = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert audit["status"] == "fail_open"
    assert audit["warnings"] == ["RetrievalTimeoutError"]


def test_full_search_results_do_not_hydrate_skills(tmp_path):
    root = tmp_path / "complete-skill"
    _write_skill(
        root,
        "---\n"
        "name: complete-skill\n"
        "description: Complete result.\n"
        "---\n"
        "# Complete Skill\n\ncomplete result split commits\n",
    )
    procedure = load_skill(root, trust_state="trusted")

    class FullResultManager:
        get_skill_calls = 0

        def search_skills(self, query, user_id=None, top_k=5):
            del query, user_id, top_k
            return [SearchResult(procedure=procedure, score=0.9)]

        def get_skill(self, id_or_name, include_content=True):
            del id_or_name, include_content
            self.get_skill_calls += 1
            raise AssertionError("complete search results should not be hydrated")

    manager = FullResultManager()
    config_path = _config_path(tmp_path)

    output = run_hook(
        _hook_input("complete result split commits"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    context = json.loads(output)["hookSpecificOutput"]["additionalContext"]
    assert '<skill rank="1" name="complete-skill"' in context
    assert manager.get_skill_calls == 0


def test_run_hook_does_not_parse_frontmatter_on_prompt_path(monkeypatch, tmp_path):
    root = tmp_path / "no-prompt-parse"
    _write_skill(
        root,
        "---\n"
        "name: no-prompt-parse\n"
        "description: Render from indexed metadata.\n"
        "---\n"
        "# No Prompt Parse\n\nPINEAPPLE_MARKER survives render.\n",
    )
    procedure = load_skill(root, trust_state="trusted")

    def fail_if_parsed(_text):
        raise AssertionError("prompt path must not parse skill frontmatter")

    monkeypatch.setattr(
        skills_module,
        "parse_skill_frontmatter",
        fail_if_parsed,
    )
    monkeypatch.setattr(
        skill_context_module,
        "parse_skill_frontmatter",
        fail_if_parsed,
        raising=False,
    )

    class IndexedOnlyManager:
        def search_skills(self, query, user_id=None, top_k=5):
            del query, user_id, top_k
            return [SearchResult(procedure=procedure, score=0.9)]

        def get_skill(self, id_or_name, include_content=True):
            del id_or_name, include_content
            raise AssertionError("complete search results should not hydrate")

    output = run_hook(
        _hook_input("PINEAPPLE_MARKER"),
        config_path=_config_path(tmp_path),
        manager_factory=lambda _config: IndexedOnlyManager(),
    )

    context = json.loads(output)["hookSpecificOutput"]["additionalContext"]
    assert '<skill rank="1" name="no-prompt-parse"' in context
    assert "PINEAPPLE_MARKER survives render." in context


def test_run_hook_retrieval_only_manager_avoids_indexing_and_sync(tmp_path):
    root = tmp_path / "hydrated-skill"
    _write_skill(
        root,
        "---\n"
        "name: hydrated-skill\n"
        "description: Hydrate from retrieval-only manager.\n"
        "---\n"
        "# Hydrated Skill\n\nretrieval only PINEAPPLE_MARKER\n",
    )
    procedure = load_skill(root, trust_state="trusted")
    partial = Procedure(
        title=procedure.title,
        content="",
        id=procedure.id,
        kind="skill",
        metadata={},
    )

    class RetrievalOnlyManager:
        def __init__(self):
            self.search_calls = 0
            self.get_skill_calls = 0

        def search_skills(self, query, user_id=None, top_k=5):
            del query, user_id, top_k
            self.search_calls += 1
            return [SearchResult(procedure=partial, score=0.9)]

        def get_skill(self, id_or_name, include_content=True):
            assert id_or_name == procedure.id
            assert include_content is True
            self.get_skill_calls += 1
            return procedure

        def add_skill(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("hook path must not index skills")

        def sync_skill(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("hook path must not sync skills")

    manager = RetrievalOnlyManager()

    output = run_hook(
        _hook_input("retrieval only"),
        config_path=_config_path(tmp_path),
        manager_factory=lambda _config: manager,
    )

    context = json.loads(output)["hookSpecificOutput"]["additionalContext"]
    assert "retrieval only PINEAPPLE_MARKER" in context
    assert manager.search_calls == 1
    assert manager.get_skill_calls == 1


def test_audit_logging_errors_fail_open(tmp_path, fake_llm):
    manager = _manager_with_skill(tmp_path, fake_llm)
    audit_dir = tmp_path / "audit-is-directory"
    audit_dir.mkdir()
    config_path = _config_path(tmp_path, logging={"path": str(audit_dir)})

    output = run_hook(
        _hook_input("split commits"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    assert output == ""


def test_renderer_escapes_xml_special_characters(tmp_path, fake_llm):
    manager = _manager_with_skill(
        tmp_path,
        fake_llm,
        name="'escape <& \"skill\"'",
        description="'Use <xml> & quotes'",
        body="# Escape & Stuff\n\nHandle <tag> & quotes.\n",
    )
    config_path = _config_path(tmp_path)

    output = run_hook(
        _hook_input("escape xml quotes"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    context = json.loads(output)["hookSpecificOutput"]["additionalContext"]
    assert 'name="escape &lt;&amp; &quot;skill&quot;"' in context
    assert "Use &lt;xml&gt; &amp; quotes" in context
    assert "Escape &amp; Stuff" in context
    assert "Handle &lt;tag&gt; &amp; quotes." in context


def test_renderer_respects_total_hard_and_per_skill_budgets(tmp_path, fake_llm):
    long_body = "# Long Skill\n\n" + "budget content " * 200
    manager = _manager_with_skill(tmp_path, fake_llm, body=long_body)
    config_path = _config_path(
        tmp_path,
        rendering={
            "max_chars": 1200,
            "hard_max_chars": 1150,
            "max_chars_per_skill": 900,
        },
    )

    output = run_hook(
        _hook_input("budget content commits"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    context = json.loads(output)["hookSpecificOutput"]["additionalContext"]
    audit = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert len(context) <= 1150
    assert 'truncated="true"' in context
    assert "...[truncated]" in context
    assert audit["selected_skills"][0]["rendered_chars"] <= 900


def test_audit_log_excludes_raw_prompt_and_skill_body_by_default(tmp_path, fake_llm):
    prompt_secret = "SECRET_PROMPT_SHOULD_NOT_BE_LOGGED"
    body_secret = "SECRET_SKILL_BODY_SHOULD_NOT_BE_LOGGED"
    manager = _manager_with_skill(
        tmp_path,
        fake_llm,
        body=f"# Privacy\n\nprivacy logging {body_secret}\n",
    )
    config_path = _config_path(tmp_path)

    output = run_hook(
        _hook_input(f"privacy logging {prompt_secret}"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    assert output
    audit_text = (tmp_path / "hook-audit.jsonl").read_text(encoding="utf-8")
    audit = json.loads(audit_text)
    assert "prompt" not in audit
    assert prompt_secret not in audit_text
    assert body_secret not in audit_text


def test_blocked_skill_is_not_injected(tmp_path, fake_llm):
    manager = _manager_with_skill(
        tmp_path,
        fake_llm,
        body="# Blocked\n\nblocked workflow split commits\n",
        trust_state="blocked",
    )
    config_path = _config_path(tmp_path)

    output = run_hook(
        _hook_input("blocked workflow split commits"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    assert output == ""
    audit = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert audit["status"] == "no_results"
    assert audit["selected_skills"] == []


def test_untrusted_skill_is_marked_as_data_context(tmp_path, fake_llm):
    manager = _manager_with_skill(
        tmp_path,
        fake_llm,
        body="# Unknown Trust\n\nunknown workflow split commits\n",
        trust_state="unknown",
    )
    config_path = _config_path(tmp_path)

    output = run_hook(
        _hook_input("unknown workflow split commits"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    context = json.loads(output)["hookSpecificOutput"]["additionalContext"]
    audit = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert 'trust_mode="data"' in context
    assert 'trust_mode="instruction"' not in context
    assert audit["selected_skills"][0]["trust_mode"] == "data"


def test_enforce_reuses_only_within_the_same_session(tmp_path):
    source_path = "/stored/session-skill/SKILL.md"
    identity = f"path:{source_path}"
    candidate = _renderer_candidate(
        name="session-skill",
        source_path=source_path,
        body="# Session Skill\n\nSame semantic payload.\n",
        headings=("Session Skill",),
    )
    manager = _StaticSkillManager([candidate])
    config_path = _config_path(
        tmp_path,
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "session-state",
            }
        },
    )

    first = run_hook(
        _hook_input("session skill", session_id="same-session"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )
    second = run_hook(
        _hook_input("session skill again", session_id="same-session"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )
    other_session = run_hook(
        _hook_input("session skill", session_id="other-session"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    first_context = json.loads(first)["hookSpecificOutput"]["additionalContext"]
    assert second == ""
    assert json.loads(other_session)["hookSpecificOutput"]["additionalContext"] == (
        first_context
    )
    rows = _audit_rows(tmp_path / "hook-audit.jsonl")
    assert [row["status"] for row in rows] == ["injected", "reused", "injected"]
    assert rows[0]["session_dedupe"]["state_load_status"] == "missing"
    assert rows[1]["selected_skills"][0]["identity"] == identity
    assert rows[1]["injected_skills"] == []
    assert rows[1]["reused_skills"] == [identity]
    assert rows[1]["skipped_skills"] == [
        {
            "identity": identity,
            "reason": "duplicate_activation_fingerprint",
        }
    ]
    assert rows[1]["session_dedupe"] == {
        "requested_rollout": "enforce",
        "rollout": "enforce",
        "configuration_status": "valid",
        "policy": "on_hash_change",
        "state_load_status": "ok",
        "context_generation": 0,
        "lifecycle_tracking": "user_prompt_submit_only",
        "dedupe_saved_chars": len(first_context),
    }


def test_enforce_emits_each_semantic_a_b_a_transition(tmp_path):
    source_path = "/stored/changing-skill/SKILL.md"
    config_path = _config_path(
        tmp_path,
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "session-state",
            }
        },
    )
    candidates = [
        _renderer_candidate(
            name="changing-skill",
            source_path=source_path,
            body=f"# Changing Skill\n\nPayload {payload}.\n",
            headings=("Changing Skill",),
            sha256=raw_sha,
        )
        for payload, raw_sha in (("A", "sha-a-1"), ("B", "sha-b"), ("A", "sha-a-2"))
    ]

    outputs = [
        run_hook(
            _hook_input(f"transition {index}"),
            config_path=config_path,
            manager_factory=lambda _config, candidate=candidate: _StaticSkillManager(
                [candidate]
            ),
        )
        for index, candidate in enumerate(candidates)
    ]

    assert all(outputs)
    rows = _audit_rows(tmp_path / "hook-audit.jsonl")
    assert [row["status"] for row in rows] == ["injected", "injected", "injected"]
    fingerprints = [row["selected_skills"][0]["activation_fingerprint"] for row in rows]
    assert fingerprints[0] != fingerprints[1]
    assert fingerprints[1] != fingerprints[2]
    assert fingerprints[0] == fingerprints[2]


@pytest.mark.parametrize("rollout", ("off", "shadow", "enforce"))
def test_fingerprint_failure_keeps_complete_hook_baseline(
    tmp_path,
    monkeypatch,
    rollout,
):
    state_dir = tmp_path / "fingerprint-failure-state"
    candidate = _renderer_candidate(
        name="unfingerprintable",
        source_path="/stored/unfingerprintable/SKILL.md",
        body="# Unfingerprintable\n\nPayload \ud800 remains renderable.\n",
        headings=("Unfingerprintable",),
    )
    config_path = _config_path(
        tmp_path,
        claude={
            "session_dedupe": {
                "rollout": rollout,
                "state_dir": str(state_dir),
            }
        },
    )

    if rollout == "off":

        class UnexpectedStateStore:
            def __init__(self, *_args, **_kwargs):
                raise AssertionError("off mode must not construct session state")

        monkeypatch.setattr(
            claude_hook_module,
            "ClaudeSessionStateStore",
            UnexpectedStateStore,
        )

    output = run_hook(
        _hook_input(f"{rollout} fingerprint failure"),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager([candidate]),
    )

    context = json.loads(output)["hookSpecificOutput"]["additionalContext"]
    identity = "path:/stored/unfingerprintable/SKILL.md"
    assert "\ud800" in context
    row = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert row["status"] == "injected"
    assert row["selected_skills"][0]["identity"] == identity
    assert row["selected_skills"][0]["activation_fingerprint"] is None
    assert "activation_fingerprint_unavailable" in row["warnings"]
    assert row["injected_skills"] == [identity]
    assert row["reused_skills"] == []
    assert row["would_reuse_skills"] == []
    assert row["skipped_skills"] == []
    if rollout == "off":
        assert not state_dir.exists()
    else:
        loaded = claude_session_state_module.ClaudeSessionStateStore(state_dir).load(
            "session-123"
        )
        assert loaded.state is not None
        assert loaded.state.activations == {}


def test_enforce_never_suppresses_unfingerprintable_skill_beside_valid_reuse(
    tmp_path,
):
    bad_source = "/stored/unfingerprintable-peer/SKILL.md"
    valid_source = "/stored/valid-peer/SKILL.md"
    bad_candidate = _renderer_candidate(
        name="unfingerprintable-peer",
        source_path=bad_source,
        body="# Unfingerprintable Peer\n\nPayload \ud800 remains renderable.\n",
        headings=("Unfingerprintable Peer",),
        score=0.9,
    )
    valid_candidate = _renderer_candidate(
        name="valid-peer",
        source_path=valid_source,
        score=0.8,
    )
    manager = _StaticSkillManager([bad_candidate, valid_candidate])
    config_path = _config_path(
        tmp_path,
        retrieval={"top_k": 2},
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "peer-state",
            }
        },
    )

    first = run_hook(
        _hook_input("seed valid peer"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )
    second = run_hook(
        _hook_input("reuse only valid peer"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    first_context = json.loads(first)["hookSpecificOutput"]["additionalContext"]
    second_context = json.loads(second)["hookSpecificOutput"]["additionalContext"]
    assert "unfingerprintable-peer" in first_context
    assert "valid-peer" in first_context
    assert "unfingerprintable-peer" in second_context
    assert "valid-peer" not in second_context
    row = _audit_rows(tmp_path / "hook-audit.jsonl")[-1]
    assert row["status"] == "mixed"
    assert row["injected_skills"] == [f"path:{bad_source}"]
    assert row["reused_skills"] == [f"path:{valid_source}"]
    assert "activation_fingerprint_unavailable" in row["warnings"]


@pytest.mark.parametrize("rollout", ("shadow", "enforce"))
def test_valid_unfingerprintable_valid_sequence_invalidates_stale_state(
    tmp_path,
    rollout,
):
    source_path = "/stored/stale-fingerprint/SKILL.md"
    identity = f"path:{source_path}"
    valid_a = _renderer_candidate(
        name="stale-fingerprint",
        source_path=source_path,
        body="# Stale Fingerprint\n\nPayload A.\n",
        headings=("Stale Fingerprint",),
    )
    unfingerprintable_b = _renderer_candidate(
        name="stale-fingerprint",
        source_path=source_path,
        body="# Stale Fingerprint\n\nPayload B contains \ud800.\n",
        headings=("Stale Fingerprint",),
    )
    config_path = _config_path(
        tmp_path,
        claude={
            "session_dedupe": {
                "rollout": rollout,
                "state_dir": "stale-fingerprint-state",
            }
        },
    )
    store = claude_session_state_module.ClaudeSessionStateStore(
        tmp_path / "stale-fingerprint-state"
    )

    outputs = []
    stored_fingerprints = []
    for index, candidate in enumerate((valid_a, unfingerprintable_b, valid_a)):
        outputs.append(
            run_hook(
                _hook_input(f"{rollout} transition {index}"),
                config_path=config_path,
                manager_factory=lambda _config, candidate=candidate: (
                    _StaticSkillManager([candidate])
                ),
            )
        )
        loaded = store.load("session-123")
        assert loaded.state is not None
        stored = loaded.state.activations.get(identity)
        stored_fingerprints.append(
            stored.activation_fingerprint if stored is not None else None
        )

    assert all(outputs)
    rows = _audit_rows(tmp_path / "hook-audit.jsonl")
    fingerprints = [row["selected_skills"][0]["activation_fingerprint"] for row in rows]
    assert fingerprints[0] is not None
    assert fingerprints == [fingerprints[0], None, fingerprints[0]]
    assert stored_fingerprints == [fingerprints[0], None, fingerprints[0]]
    assert all(row["reused_skills"] == [] for row in rows)
    assert all(row["would_reuse_skills"] == [] for row in rows)


def test_hook_records_valid_update_and_invalidation_in_one_state_mutation(
    tmp_path,
    monkeypatch,
):
    invalidated_source = "/stored/combined-invalidated/SKILL.md"
    updated_source = "/stored/combined-updated/SKILL.md"
    unrelated_source = "/stored/combined-unrelated/SKILL.md"
    invalidated_identity = f"path:{invalidated_source}"
    updated_identity = f"path:{updated_source}"
    unrelated_identity = f"path:{unrelated_source}"
    seed_candidates = [
        _renderer_candidate(
            name="combined-invalidated",
            source_path=invalidated_source,
            body="# Combined Invalidated\n\nOld valid payload.\n",
            headings=("Combined Invalidated",),
            score=0.9,
        ),
        _renderer_candidate(
            name="combined-updated",
            source_path=updated_source,
            body="# Combined Updated\n\nOld payload.\n",
            headings=("Combined Updated",),
            score=0.8,
        ),
        _renderer_candidate(
            name="combined-unrelated",
            source_path=unrelated_source,
            score=0.7,
        ),
    ]
    next_candidates = [
        _renderer_candidate(
            name="combined-invalidated",
            source_path=invalidated_source,
            body="# Combined Invalidated\n\nUnfingerprintable \ud800 payload.\n",
            headings=("Combined Invalidated",),
            score=0.9,
        ),
        _renderer_candidate(
            name="combined-updated",
            source_path=updated_source,
            body="# Combined Updated\n\nNew payload.\n",
            headings=("Combined Updated",),
            score=0.8,
        ),
    ]
    config_path = _config_path(
        tmp_path,
        retrieval={"top_k": 3},
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "combined-state",
            }
        },
    )
    run_hook(
        _hook_input("seed combined state"),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager(seed_candidates),
    )
    store = claude_session_state_module.ClaudeSessionStateStore(
        tmp_path / "combined-state"
    )
    before = store.load("session-123")
    assert before.state is not None
    unrelated_before = before.state.activations[unrelated_identity]
    original_record = (
        claude_session_state_module.ClaudeSessionStateStore.record_activations
    )
    recorded_calls = []

    def spy_record(
        self,
        session_id,
        activations,
        *,
        expected_generation,
        invalidated_identities=(),
    ):
        frozen_activations = tuple(activations)
        frozen_invalidations = tuple(invalidated_identities)
        recorded_calls.append(
            (frozen_activations, frozen_invalidations, expected_generation)
        )
        return original_record(
            self,
            session_id,
            frozen_activations,
            expected_generation=expected_generation,
            invalidated_identities=frozen_invalidations,
        )

    monkeypatch.setattr(
        claude_session_state_module.ClaudeSessionStateStore,
        "record_activations",
        spy_record,
    )

    output = run_hook(
        _hook_input("apply combined mutation"),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager(next_candidates),
    )
    loaded = store.load("session-123")

    assert output
    assert len(recorded_calls) == 1
    updates, invalidations, generation = recorded_calls[0]
    assert [update.identity for update in updates] == [updated_identity]
    assert set(invalidations) == {invalidated_identity}
    assert generation == 0
    assert loaded.state is not None
    assert set(loaded.state.activations) == {updated_identity, unrelated_identity}
    assert loaded.state.activations[unrelated_identity] == unrelated_before
    selected = _audit_rows(tmp_path / "hook-audit.jsonl")[-1]["selected_skills"]
    expected_updated_fingerprint = next(
        skill["activation_fingerprint"]
        for skill in selected
        if skill["identity"] == updated_identity
    )
    assert (
        loaded.state.activations[updated_identity].activation_fingerprint
        == expected_updated_fingerprint
    )


def test_pathless_enforce_records_each_a_b_a_transition_in_one_identity_slot(
    tmp_path,
):
    procedure_id = "pathless-hook-skill"
    identity = f"id:{procedure_id}"
    config_path = _config_path(
        tmp_path,
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "pathless-state",
            }
        },
    )
    candidates = [
        _renderer_candidate(
            name="pathless-hook-skill",
            procedure_id=procedure_id,
            source_path=None,
            metadata_source_path=None,
            body=f"# Pathless Hook Skill\n\nPayload {payload}.\n",
            headings=("Pathless Hook Skill",),
            sha256=raw_sha,
        )
        for payload, raw_sha in (("A", "sha-a-1"), ("B", "sha-b"), ("A", "sha-a-2"))
    ]
    store = claude_session_state_module.ClaudeSessionStateStore(
        tmp_path / "pathless-state"
    )
    outputs = []
    stored_fingerprints = []

    for index, candidate in enumerate(candidates):
        outputs.append(
            run_hook(
                _hook_input(f"pathless transition {index}"),
                config_path=config_path,
                manager_factory=lambda _config, candidate=candidate: (
                    _StaticSkillManager([candidate])
                ),
            )
        )
        loaded = store.load("session-123")
        assert loaded.state is not None
        assert set(loaded.state.activations) == {identity}
        stored_fingerprints.append(
            loaded.state.activations[identity].activation_fingerprint
        )

    assert all(outputs)
    rows = _audit_rows(tmp_path / "hook-audit.jsonl")
    fingerprints = [row["selected_skills"][0]["activation_fingerprint"] for row in rows]
    assert fingerprints[0] != fingerprints[1]
    assert fingerprints[1] != fingerprints[2]
    assert fingerprints[0] == fingerprints[2]
    assert stored_fingerprints == fingerprints
    assert [row["status"] for row in rows] == ["injected", "injected", "injected"]


def test_enforce_filters_completed_baseline_without_reallocating_budget(tmp_path):
    reused = _renderer_candidate(
        name="budget-reused",
        source_path="/a",
        body="# Budget Reused\n\n" + ("A" * 5_000),
        headings=("Budget Reused",),
        score=0.9,
        trust_mode="data",
    )
    retained = _renderer_candidate(
        name="budget-retained",
        source_path="/b",
        body="# Budget Retained\n\n" + ("B" * 1_500),
        headings=("Budget Retained",),
        score=0.8,
        trust_mode="data",
    )
    lower = _renderer_candidate(
        name="budget-lower",
        source_path="/c",
        body="# Budget Lower\n\nCompact lower-ranked payload.\n",
        headings=("Budget Lower",),
        score=0.7,
        trust_mode="data",
    )
    config_path = _config_path(
        tmp_path,
        retrieval={"top_k": 3},
        rendering={
            "max_chars": 1_600,
            "hard_max_chars": 1_600,
            "max_chars_per_skill": 900,
        },
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "budget-state",
            }
        },
    )
    renderer = ContextRenderer(load_hook_config(config_path))
    baseline = renderer.render([reused, retained, lower], trace_id="budget-baseline")
    counterfactual = renderer.render(
        [retained, lower],
        trace_id="budget-counterfactual",
    )

    assert [skill.candidate.procedure.title for skill in baseline.skills] == [
        "budget-reused",
        "budget-retained",
    ]
    assert baseline.skills[1].rank == 2
    assert 'truncated="true"' in baseline.skills[1].xml
    assert [skill.candidate.procedure.title for skill in counterfactual.skills] == [
        "budget-retained",
        "budget-lower",
    ]
    assert counterfactual.skills[0].rendered_chars > baseline.skills[1].rendered_chars

    run_hook(
        _hook_input("seed reused budget skill"),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager([reused]),
    )
    output = run_hook(
        _hook_input("filter completed budget baseline"),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager([reused, retained, lower]),
    )

    context = json.loads(output)["hookSpecificOutput"]["additionalContext"]
    assert context == renderer.compose((baseline.skills[1],))
    assert '<skill rank="2" name="budget-retained"' in context
    assert "budget-lower" not in context
    row = _audit_rows(tmp_path / "hook-audit.jsonl")[-1]
    assert [skill["name"] for skill in row["selected_skills"]] == [
        "budget-reused",
        "budget-retained",
    ]
    assert row["selected_skills"][1]["rank"] == 2
    assert row["selected_skills"][1]["rendered_chars"] == (
        baseline.skills[1].rendered_chars
    )
    assert row["reused_skills"] == ["path:/a"]
    assert row["injected_skills"] == ["path:/b"]


def test_enforce_mixed_output_preserves_sparse_rank_and_never_backfills(tmp_path):
    source_a = "/stored/baseline-a/SKILL.md"
    source_b = "/stored/baseline-b/SKILL.md"
    candidate_a = _renderer_candidate(
        name="baseline-a",
        source_path=source_a,
        score=0.9,
    )
    candidate_b = _renderer_candidate(
        name="baseline-b",
        source_path=source_b,
        score=0.8,
    )
    lower_candidate = _renderer_candidate(
        name="lower-candidate",
        source_path="/stored/lower/SKILL.md",
        score=0.7,
    )
    config_path = _config_path(
        tmp_path,
        retrieval={"top_k": 2},
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "session-state",
            }
        },
    )
    seed_manager = _StaticSkillManager([candidate_a])
    baseline_manager = _StaticSkillManager([candidate_a, candidate_b, lower_candidate])
    run_hook(
        _hook_input("seed A", session_id="mixed-session"),
        config_path=config_path,
        manager_factory=lambda _config: seed_manager,
    )
    full_output = run_hook(
        _hook_input("full baseline", session_id="baseline-session"),
        config_path=config_path,
        manager_factory=lambda _config: baseline_manager,
    )

    mixed_output = run_hook(
        _hook_input("mixed baseline", session_id="mixed-session"),
        config_path=config_path,
        manager_factory=lambda _config: baseline_manager,
    )

    baseline_context = json.loads(full_output)["hookSpecificOutput"][
        "additionalContext"
    ]
    mixed_context = json.loads(mixed_output)["hookSpecificOutput"]["additionalContext"]
    assert 'name="baseline-a"' not in mixed_context
    assert '<skill rank="2" name="baseline-b"' in mixed_context
    assert "lower-candidate" not in mixed_context
    row = _audit_rows(tmp_path / "hook-audit.jsonl")[-1]
    assert row["status"] == "mixed"
    assert [skill["name"] for skill in row["selected_skills"]] == [
        "baseline-a",
        "baseline-b",
    ]
    assert row["reused_skills"] == [f"path:{source_a}"]
    assert row["injected_skills"] == [f"path:{source_b}"]
    assert row["session_dedupe"]["dedupe_saved_chars"] == (
        len(baseline_context) - len(mixed_context)
    )


def test_enforce_reuses_rank_and_score_only_changes(tmp_path):
    stable_source = "/stored/rank-stable/SKILL.md"
    first = _renderer_candidate(
        name="rank-stable",
        source_path=stable_source,
        score=0.9,
    )
    higher = _renderer_candidate(
        name="new-higher-rank",
        source_path="/stored/higher/SKILL.md",
        score=0.95,
    )
    reranked = replace(first, score=0.4)
    config_path = _config_path(
        tmp_path,
        retrieval={"top_k": 2},
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "session-state",
            }
        },
    )
    run_hook(
        _hook_input("seed rank"),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager([first]),
    )

    output = run_hook(
        _hook_input("rerank"),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager([higher, reranked]),
    )

    context = json.loads(output)["hookSpecificOutput"]["additionalContext"]
    assert "new-higher-rank" in context
    assert "rank-stable" not in context
    row = _audit_rows(tmp_path / "hook-audit.jsonl")[-1]
    stable_metadata = next(
        skill for skill in row["selected_skills"] if skill["name"] == "rank-stable"
    )
    assert stable_metadata["rank"] == 2
    assert stable_metadata["score"] == 0.4
    assert row["reused_skills"] == [f"path:{stable_source}"]


def test_shadow_audits_reuse_but_never_reduces_context(tmp_path):
    candidate = _renderer_candidate(
        name="shadow-skill",
        source_path="/stored/shadow/SKILL.md",
    )
    manager = _StaticSkillManager([candidate])
    config_path = _config_path(
        tmp_path,
        claude={
            "session_dedupe": {
                "rollout": "shadow",
                "state_dir": "session-state",
            }
        },
    )

    first = run_hook(
        _hook_input("shadow first"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )
    second = run_hook(
        _hook_input("shadow second"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    assert (
        json.loads(first)["hookSpecificOutput"]["additionalContext"]
        == json.loads(second)["hookSpecificOutput"]["additionalContext"]
    )
    row = _audit_rows(tmp_path / "hook-audit.jsonl")[-1]
    identity = "path:/stored/shadow/SKILL.md"
    assert row["status"] == "injected"
    assert row["injected_skills"] == [identity]
    assert row["reused_skills"] == []
    assert row["would_reuse_skills"] == [identity]
    assert row["skipped_skills"] == []
    assert row["session_dedupe"]["dedupe_saved_chars"] == 0


@pytest.mark.parametrize("prompt", ["/clear", "/compact"])
def test_prompt_strings_do_not_advance_context_generation(
    tmp_path,
    monkeypatch,
    prompt,
):
    candidate = _renderer_candidate()
    manager = _StaticSkillManager([candidate])
    config_path = _config_path(
        tmp_path,
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "prompt-string-state",
            }
        },
    )

    def unexpected_advance(*_args, **_kwargs):
        raise AssertionError("UserPromptSubmit must not advance lifecycle generation")

    monkeypatch.setattr(
        claude_session_state_module.ClaudeSessionStateStore,
        "advance_context_generation",
        unexpected_advance,
    )
    first = run_hook(
        _hook_input("seed prompt-string session"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )
    after_prompt_string = run_hook(
        _hook_input(prompt),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    assert first
    assert after_prompt_string == ""
    row = _audit_rows(tmp_path / "hook-audit.jsonl")[-1]
    assert row["status"] == "reused"
    assert row["session_dedupe"]["context_generation"] == 0


def test_default_off_emits_baseline_and_never_constructs_state_store(
    tmp_path,
    monkeypatch,
):
    candidate = _renderer_candidate()

    class UnexpectedStateStore:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("off mode must not construct session state")

    monkeypatch.setattr(
        claude_hook_module,
        "ClaudeSessionStateStore",
        UnexpectedStateStore,
    )
    config_path = _config_path(tmp_path)

    output = run_hook(
        _hook_input("off mode"),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager([candidate]),
    )

    assert output
    assert not (tmp_path / "claude-sessions").exists()
    row = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert row["status"] == "injected"
    assert row["session_dedupe"]["rollout"] == "off"
    assert row["session_dedupe"]["state_load_status"] == "off"
    assert row["session_dedupe"]["lifecycle_tracking"] == ("user_prompt_submit_only")


def test_enforce_missing_session_and_no_results_do_no_state_work(tmp_path):
    candidate = _renderer_candidate(score=0.99)
    config_path = _config_path(
        tmp_path,
        retrieval={"min_score": 0.95},
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "session-state",
            }
        },
    )

    missing_session = run_hook(
        _hook_input("missing session", session_id=""),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager([candidate]),
    )
    no_results = run_hook(
        _hook_input("no results"),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager([]),
    )

    assert missing_session
    assert no_results == ""
    assert not (tmp_path / "session-state").exists()
    rows = _audit_rows(tmp_path / "hook-audit.jsonl")
    assert rows[0]["status"] == "injected"
    assert "session_dedupe_missing_session_id" in rows[0]["warnings"]
    assert rows[0]["session_dedupe"]["state_load_status"] == "missing_session_id"
    assert rows[1]["status"] == "no_results"
    assert rows[1]["session_dedupe"]["state_load_status"] == "not_needed"


def test_enforce_corrupt_state_fails_open_without_overwriting_it(tmp_path):
    candidate = _renderer_candidate()
    config_path = _config_path(
        tmp_path,
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "session-state",
            }
        },
    )
    store = claude_session_state_module.ClaudeSessionStateStore(
        tmp_path / "session-state"
    )
    state_path = store.state_path("session-123")
    state_path.parent.mkdir()
    invalid_state = b"{not valid session state"
    state_path.write_bytes(invalid_state)

    output = run_hook(
        _hook_input("corrupt state"),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager([candidate]),
    )

    assert output
    assert state_path.read_bytes() == invalid_state
    row = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert row["status"] == "injected"
    assert row["session_dedupe"]["state_load_status"] == "invalid_json"
    assert "claude_session_state_invalid_json" in row["warnings"]


@pytest.mark.parametrize("rollout", ("shadow", "enforce"))
def test_stateful_rollout_unsupported_lock_platform_fails_open_without_state_io(
    tmp_path,
    monkeypatch,
    rollout,
):
    candidate = _renderer_candidate()
    config_path = _config_path(
        tmp_path,
        claude={
            "session_dedupe": {
                "rollout": rollout,
                "state_dir": "unsupported-state",
            }
        },
    )
    monkeypatch.setattr(claude_session_state_module, "fcntl", None)

    output = run_hook(
        _hook_input("unsupported lock platform"),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager([candidate]),
    )

    assert output
    assert not (tmp_path / "unsupported-state").exists()
    row = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert row["session_dedupe"]["rollout"] == rollout
    assert row["session_dedupe"]["state_load_status"] == ("unsupported_lock_platform")
    assert row["reused_skills"] == []
    assert row["would_reuse_skills"] == []
    assert "claude_session_state_unsupported_lock_platform" in row["warnings"]


def test_unexpected_state_load_error_fails_open_without_state_record(
    tmp_path,
    monkeypatch,
):
    candidate = _renderer_candidate()
    config_path = _config_path(
        tmp_path,
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "load-error-state",
            }
        },
    )

    class LoadErrorStore:
        def __init__(self, _state_dir):
            pass

        def load(self, _session_id):
            raise RuntimeError("state load failed")

    monkeypatch.setattr(
        claude_hook_module,
        "ClaudeSessionStateStore",
        LoadErrorStore,
    )

    output = run_hook(
        _hook_input("load failure"),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager([candidate]),
    )

    assert output
    assert not (tmp_path / "load-error-state").exists()
    row = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert row["status"] == "injected"
    assert row["session_dedupe"]["state_load_status"] == "load_failed"
    assert "session_dedupe_state_load_failed" in row["warnings"]


def test_unexpected_activation_compare_error_fails_open_without_state_record(
    tmp_path,
    monkeypatch,
):
    candidate = _renderer_candidate()
    config_path = _config_path(
        tmp_path,
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "compare-error-state",
            }
        },
    )

    def fail_compare(_baseline, _activations):
        raise RuntimeError("comparison failed")

    monkeypatch.setattr(
        claude_hook_module,
        "_matching_activations",
        fail_compare,
    )

    output = run_hook(
        _hook_input("compare failure"),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager([candidate]),
    )

    assert output
    assert not (tmp_path / "compare-error-state").exists()
    row = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert row["status"] == "injected"
    assert row["session_dedupe"]["state_load_status"] == "compare_failed"
    assert "session_dedupe_compare_failed" in row["warnings"]


@pytest.mark.parametrize("failure_status", ["write_failed", "lock_timeout"])
def test_post_audit_state_failure_preserves_output_and_next_call_injects(
    tmp_path,
    monkeypatch,
    failure_status,
):
    candidate = _renderer_candidate()
    manager = _StaticSkillManager([candidate])
    config_path = _config_path(
        tmp_path,
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "record-failure-state",
            }
        },
    )

    def fail_record(
        self,
        session_id,
        activations,
        *,
        expected_generation,
        invalidated_identities=(),
    ):
        del (
            self,
            session_id,
            activations,
            expected_generation,
            invalidated_identities,
        )
        return claude_session_state_module.StateOperationResult(failure_status)

    with monkeypatch.context() as patch:
        patch.setattr(
            claude_session_state_module.ClaudeSessionStateStore,
            "record_activations",
            fail_record,
        )
        failed_record_output = run_hook(
            _hook_input("record failure"),
            config_path=config_path,
            manager_factory=lambda _config: manager,
        )

    retry_output = run_hook(
        _hook_input("retry after record failure"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )
    reused_output = run_hook(
        _hook_input("reuse after successful retry"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    assert failed_record_output
    assert retry_output
    assert reused_output == ""
    rows = _audit_rows(tmp_path / "hook-audit.jsonl")
    assert [row["status"] for row in rows] == [
        "injected",
        "state_record_failed",
        "injected",
        "reused",
    ]
    failure_row = rows[1]
    assert failure_row["trace_id"] == rows[0]["trace_id"]
    assert failure_row["session_id_hash"] == rows[0]["session_id_hash"]
    assert failure_row["audit_event"] == "session_dedupe_state_record"
    assert failure_row["warnings"] == [f"claude_session_state_{failure_status}"]
    assert failure_row["session_dedupe"] == {
        "requested_rollout": "enforce",
        "rollout": "enforce",
        "configuration_status": "valid",
        "policy": "on_hash_change",
        "state_load_status": "missing",
        "state_record_status": failure_status,
        "context_generation": 0,
        "lifecycle_tracking": "user_prompt_submit_only",
    }


def test_audit_failure_leaves_existing_activation_state_unchanged(tmp_path):
    source_path = "/stored/audit-order/SKILL.md"
    candidate_a = _renderer_candidate(
        name="audit-order",
        source_path=source_path,
        body="# Audit Order\n\nPayload A.\n",
        headings=("Audit Order",),
    )
    candidate_b = _renderer_candidate(
        name="audit-order",
        source_path=source_path,
        body="# Audit Order\n\nPayload B contains \ud800.\n",
        headings=("Audit Order",),
    )
    config_path = _config_path(
        tmp_path,
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "audit-order-state",
            }
        },
    )
    first = run_hook(
        _hook_input("seed audit state"),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager([candidate_a]),
    )
    assert first
    store = claude_session_state_module.ClaudeSessionStateStore(
        tmp_path / "audit-order-state"
    )
    state_path = store.state_path("session-123")
    state_before = state_path.read_bytes()
    audit_directory = tmp_path / "audit-is-directory"
    audit_directory.mkdir()
    config_path = _config_path(
        tmp_path,
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "audit-order-state",
            }
        },
        logging={"path": str(audit_directory)},
    )

    failed = run_hook(
        _hook_input("changed payload with failed audit"),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager([candidate_b]),
    )

    assert failed == ""
    assert state_path.read_bytes() == state_before
    recovered_config = _config_path(
        tmp_path,
        claude={
            "session_dedupe": {
                "rollout": "enforce",
                "state_dir": "audit-order-state",
            }
        },
        logging={"path": str(tmp_path / "recovered-audit.jsonl")},
    )
    recovered = run_hook(
        _hook_input("changed payload after audit recovers"),
        config_path=recovered_config,
        manager_factory=lambda _config: _StaticSkillManager([candidate_b]),
    )
    assert recovered
    loaded = store.load("session-123")
    assert loaded.state is not None
    assert f"path:{source_path}" not in loaded.state.activations


def test_config_defaults_unknown_fields_and_top_k_clamping(tmp_path):
    missing = load_hook_config(tmp_path / "missing.json")
    assert missing["schema_version"] == "memflow.claude_hook.v1"
    assert missing["retrieval"]["top_k"] == 3
    assert missing["retrieval"]["candidate_k"] == 20
    assert missing["claude"]["native_catalog_mode"] == "hidden_or_minimized"
    assert missing["claude"]["session_dedupe"] == {
        "rollout": "off",
        "policy": "on_hash_change",
        "state_dir": "claude-sessions",
    }
    assert missing["_memflow_session_dedupe"] == {
        "requested_rollout": "off",
        "rollout": "off",
        "configuration_status": "valid",
        "policy": "on_hash_change",
        "state_dir": str(tmp_path / "claude-sessions"),
        "warnings": [],
    }

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "retrieval": {
                    "top_k": 10,
                    "max_top_k": 2,
                    "candidate_k": 1,
                },
                "claude": {
                    "session_dedupe": {
                        "rollout": "shadow",
                        "state_dir": "nested/../session-state",
                        "future_field": {"kept": True},
                    }
                },
                "future_gateway_field": {"kept": True},
            }
        ),
        encoding="utf-8",
    )

    clamped = load_hook_config(path)
    assert clamped["retrieval"]["top_k"] == 2
    assert clamped["retrieval"]["candidate_k"] == 2
    assert clamped["future_gateway_field"] == {"kept": True}
    assert clamped["claude"]["session_dedupe"]["future_field"] == {"kept": True}
    assert clamped["_memflow_session_dedupe"] == {
        "requested_rollout": "shadow",
        "rollout": "shadow",
        "configuration_status": "valid",
        "policy": "on_hash_change",
        "state_dir": str(tmp_path / "session-state"),
        "warnings": [],
    }


@pytest.mark.parametrize(
    ("raw_session_dedupe", "expected_warning"),
    [
        ("bad", "invalid_session_dedupe_config"),
        ({"rollout": "sometimes"}, "invalid_session_dedupe_rollout"),
        ({"policy": "always"}, "invalid_session_dedupe_policy"),
        ({"state_dir": ""}, "invalid_session_dedupe_state_dir"),
        ({"state_dir": 42}, "invalid_session_dedupe_state_dir"),
    ],
)
def test_invalid_session_dedupe_config_falls_back_to_off_without_state_work(
    tmp_path,
    raw_session_dedupe,
    expected_warning,
):
    config_path = _config_path(
        tmp_path,
        claude={"session_dedupe": raw_session_dedupe},
    )
    loaded = load_hook_config(config_path)
    candidate = _renderer_candidate()

    output = run_hook(
        _hook_input("invalid dedupe config"),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager([candidate]),
    )

    assert output
    assert loaded["_memflow_session_dedupe"]["rollout"] == "off"
    assert expected_warning in loaded["_memflow_session_dedupe"]["warnings"]
    assert not (tmp_path / "claude-sessions").exists()
    row = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert row["session_dedupe"]["rollout"] == "off"
    assert row["session_dedupe"]["configuration_status"] == "invalid_fallback_off"
    assert expected_warning in row["warnings"]


def test_invalid_shadow_state_dir_audits_effective_off_without_state_work(tmp_path):
    config_path = _config_path(
        tmp_path,
        claude={
            "session_dedupe": {
                "rollout": "shadow",
                "policy": "on_hash_change",
                "state_dir": "",
            }
        },
    )
    candidate = _renderer_candidate()

    output = run_hook(
        _hook_input("invalid shadow state directory"),
        config_path=config_path,
        manager_factory=lambda _config: _StaticSkillManager([candidate]),
    )

    assert output
    assert not (tmp_path / "claude-sessions").exists()
    row = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert row["status"] == "injected"
    assert row["would_reuse_skills"] == []
    assert row["session_dedupe"]["requested_rollout"] == "shadow"
    assert row["session_dedupe"]["rollout"] == "off"
    assert row["session_dedupe"]["configuration_status"] == "invalid_fallback_off"
    assert row["session_dedupe"]["state_load_status"] == "off"
    assert "invalid_session_dedupe_state_dir" in row["warnings"]


def test_invalid_catalog_mode_falls_back_and_audits_warning(tmp_path, fake_llm):
    manager = _manager_with_skill(tmp_path, fake_llm)
    config_path = _config_path(
        tmp_path,
        claude={"native_catalog_mode": "not-a-mode"},
    )
    loaded_config = load_hook_config(config_path)

    assert loaded_config["claude"]["native_catalog_mode"] == "not-a-mode"
    assert loaded_config["_memflow_catalog_mode"] == {
        "raw": "not-a-mode",
        "effective": "hidden_or_minimized",
        "warnings": ["invalid_native_catalog_mode"],
    }

    output = run_hook(
        _hook_input("split commits"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    assert output
    context = json.loads(output)["hookSpecificOutput"]["additionalContext"]
    assert "catalog_mode=" not in context
    audit = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert audit["native_catalog_mode"] == "hidden_or_minimized"
    assert audit["native_catalog_mode_raw"] == "not-a-mode"
    assert audit["native_catalog_mode_effective"] == "hidden_or_minimized"
    assert "invalid_native_catalog_mode" in audit["warnings"]


def test_non_object_claude_config_falls_back_without_fail_open(tmp_path, fake_llm):
    manager = _manager_with_skill(tmp_path, fake_llm)
    config_path = _config_path(tmp_path, claude="bad")

    output = run_hook(
        _hook_input("split commits"),
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    assert output
    audit = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert audit["status"] == "injected"
    assert audit["native_catalog_mode"] == "hidden_or_minimized"
    assert "invalid_claude_config" in audit["warnings"]


def test_user_prompt_hook_does_not_mutate_claude_settings(tmp_path, fake_llm):
    manager = _manager_with_skill(tmp_path, fake_llm)
    config_path = _config_path(tmp_path)
    settings_path = tmp_path / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    original_settings = {
        "disableBundledSkills": False,
        "skillOverrides": {"manual-skill": "off"},
    }
    settings_path.write_text(json.dumps(original_settings), encoding="utf-8")
    hook_input = json.dumps(
        {
            "session_id": "session-123",
            "transcript_path": "/tmp/transcript.jsonl",
            "cwd": str(tmp_path),
            "hook_event_name": "UserPromptSubmit",
            "prompt": "split commits",
        }
    )

    output = run_hook(
        hook_input,
        config_path=config_path,
        manager_factory=lambda _config: manager,
    )

    assert output
    assert json.loads(settings_path.read_text(encoding="utf-8")) == original_settings
    assert not (tmp_path / ".memflow" / "claude-catalog-state.json").exists()


@pytest.mark.parametrize(
    "option",
    ("--dry-run", "--apply", "--mode", "--settings-path", "--project-root"),
)
def test_runtime_cli_rejects_setup_only_options(option):
    with pytest.raises(SystemExit):
        claude_hook_main(
            [option],
            stdin=io.StringIO(""),
            stdout=io.StringIO(),
        )
