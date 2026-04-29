# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the interactive CLI."""

from __future__ import annotations

import io

from memflow.cli import (
    _create_manager,
    format_verbose_trace,
    run_repl,
)
from memflow.models import Procedure, SearchResult


class FakeManager:
    def __init__(self, result: dict):
        self.result = result
        self.calls = []

    def chat(self, message, user_id=None, history=None, allow_execute=False):
        self.calls.append(
            {
                "message": message,
                "user_id": user_id,
                "history": history,
                "allow_execute": allow_execute,
            }
        )
        return self.result


def test_format_verbose_trace_for_search():
    procedure = Procedure(title="Deploy app", content="1. Run deploy.sh")
    result = {
        "response": "Found it",
        "intents": ["SEARCH"],
        "primary_intent": "SEARCH",
        "handler_results": {
            "SEARCH": {
                "response": "Found it",
                "intent": "SEARCH",
                "results": [SearchResult(procedure=procedure, score=0.75)],
            }
        },
    }

    trace = format_verbose_trace(
        result,
        user_id="alice",
        allow_execute=False,
        history_count=2,
    )

    assert "[trace] chat()" in trace
    assert "intents: SEARCH" in trace
    assert "search.results: 1" in trace
    assert "Deploy app (score=0.750)" in trace


def test_format_verbose_trace_for_add_memory_type():
    result = {
        "response": "Saved",
        "intents": ["ADD"],
        "primary_intent": "ADD",
        "handler_results": {
            "ADD": {
                "response": "Saved",
                "intent": "ADD",
                "data": {
                    "type": "procedural",
                    "results": [{"title": "Restart service", "event": "ADD"}],
                },
            }
        },
    }

    trace = format_verbose_trace(
        result,
        user_id="demo",
        allow_execute=False,
        history_count=0,
    )

    assert "memory_type: procedural" in trace
    assert "stored: 1" in trace
    assert "Restart service (ADD)" in trace


def test_run_repl_passes_input_to_chat_and_tracks_history():
    manager = FakeManager(
        {
            "response": "hello",
            "intents": ["CONVERSATION"],
            "primary_intent": "CONVERSATION",
        }
    )
    inputs = iter(["hi", "again", "/exit"])
    output = io.StringIO()

    run_repl(
        manager,
        user_id="user1",
        input_fn=lambda _prompt: next(inputs),
        output=output,
    )

    assert output.getvalue().splitlines() == ["hello", "hello"]
    assert manager.calls[0]["message"] == "hi"
    assert manager.calls[0]["history"] == []
    assert manager.calls[1]["message"] == "again"
    assert manager.calls[1]["history"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_create_manager_uses_current_public_api_name(monkeypatch):
    import memflow.manager as manager_module

    class DummyMemFlow:
        pass

    monkeypatch.setattr(manager_module, "MemFlow", DummyMemFlow)

    assert isinstance(_create_manager(), DummyMemFlow)


def test_run_repl_toggles_verbose_and_execute():
    result = {
        "response": "ok",
        "intents": ["EXECUTE"],
        "primary_intent": "EXECUTE",
        "requires_confirmation": True,
    }
    manager = FakeManager(result)
    inputs = iter(["/verbose on", "/execute on", "run date", "/exit"])
    output = io.StringIO()

    run_repl(
        manager,
        input_fn=lambda _prompt: next(inputs),
        output=output,
    )

    text = output.getvalue()
    assert "verbose: on" in text
    assert "execute: on" in text
    assert "[trace] chat()" in text
    assert manager.calls[0]["allow_execute"] is True
