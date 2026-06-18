# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import time

import pytest

from memflow.claude_hook import default_manager_factory, load_hook_config, run_hook
from memflow.manager import MemFlow
from memflow.models import SearchResult
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


def _hook_input(prompt: str, *, event: str = "UserPromptSubmit") -> str:
    return json.dumps(
        {
            "session_id": "session-123",
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
    assert context.startswith("<memflow_selected_skills")
    assert '<skill rank="1" name="commit-craft"' in context
    assert '<content truncated="false">' in context
    assert "Split commits into reviewable units." in context

    audit = _audit_rows(tmp_path / "hook-audit.jsonl")[0]
    assert audit["status"] == "injected"
    assert audit["adapter"] == "claude-code-user-prompt-submit"
    assert audit["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert audit["session_id_hash"] == hashlib.sha256(b"session-123").hexdigest()
    assert audit["selected_skills"][0]["name"] == "commit-craft"
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


def test_invalid_json_and_non_user_prompt_events_fail_open(tmp_path, fake_llm):
    manager = _manager_with_skill(tmp_path, fake_llm)
    config_path = _config_path(tmp_path)

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


def test_config_defaults_unknown_fields_and_top_k_clamping(tmp_path):
    missing = load_hook_config(tmp_path / "missing.json")
    assert missing["schema_version"] == "memflow.claude_hook.v1"
    assert missing["retrieval"]["top_k"] == 3
    assert missing["retrieval"]["candidate_k"] == 20

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "retrieval": {
                    "top_k": 10,
                    "max_top_k": 2,
                    "candidate_k": 1,
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
