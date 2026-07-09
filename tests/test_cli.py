# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the interactive CLI."""

from __future__ import annotations

import io
import time

from memflow.cli import (
    StatusLine,
    _create_manager,
    _handle_prompt_cancel,
    _handle_prompt_clear_screen,
    _handle_prompt_newline,
    _handle_prompt_submit,
    format_verbose_trace,
    main,
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


class TtyOutput(io.StringIO):
    def isatty(self):
        return True


class FakePromptBuffer:
    def __init__(self, text=""):
        self.text = text
        self.inserted = []
        self.reset_count = 0

    def insert_text(self, text):
        self.inserted.append(text)
        self.text += text

    def reset(self):
        self.text = ""
        self.reset_count += 1


class FakePromptRenderer:
    def __init__(self):
        self.clear_count = 0

    def clear(self):
        self.clear_count += 1


class FakePromptApp:
    def __init__(self, text=""):
        self.current_buffer = FakePromptBuffer(text)
        self.renderer = FakePromptRenderer()
        self.exit_result = None
        self.exit_exception = None

    def exit(self, *, result=None, exception=None):
        self.exit_result = result
        self.exit_exception = exception


class FakePromptEvent:
    def __init__(self, text=""):
        self.app = FakePromptApp(text)


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


def test_main_without_command_uses_legacy_repl_path():
    output = io.StringIO()

    rc = main([], stdout=output, input_fn=lambda _prompt: "/exit")

    assert rc == 0
    assert output.getvalue() == ""


def test_top_level_single_prompt_uses_legacy_chat_path():
    manager = FakeManager(
        {
            "response": "hello",
            "intents": ["CONVERSATION"],
            "primary_intent": "CONVERSATION",
        }
    )
    output = io.StringIO()

    rc = main(
        ["-p", "hi", "--user-id", "alice", "--no-history"],
        stdout=output,
        manager_factory=lambda: manager,
    )

    assert rc == 0
    assert output.getvalue() == "hello\n\n"
    assert manager.calls == [
        {
            "message": "hi",
            "user_id": "alice",
            "history": None,
            "allow_execute": False,
        }
    ]


def test_chat_subcommand_single_prompt_uses_existing_chat_path():
    manager = FakeManager(
        {
            "response": "hello",
            "intents": ["CONVERSATION"],
            "primary_intent": "CONVERSATION",
        }
    )
    output = io.StringIO()

    rc = main(
        ["chat", "-p", "hi", "--user-id", "alice", "--no-history"],
        stdout=output,
        manager_factory=lambda: manager,
    )

    assert rc == 0
    assert output.getvalue() == "hello\n\n"
    assert manager.calls == [
        {
            "message": "hi",
            "user_id": "alice",
            "history": None,
            "allow_execute": False,
        }
    ]


def test_run_repl_strips_surrounding_blank_response_lines():
    manager = FakeManager(
        {
            "response": "\n\n    hello\n\n",
            "intents": ["CONVERSATION"],
            "primary_intent": "CONVERSATION",
        }
    )
    inputs = iter(["hi", "/exit"])
    output = io.StringIO()

    run_repl(
        manager,
        input_fn=lambda _prompt: next(inputs),
        output=output,
    )

    assert output.getvalue() == "    hello\n"
    assert manager.calls[0]["message"] == "hi"


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


def test_run_repl_clear_command_resets_chat_history():
    manager = FakeManager(
        {
            "response": "hello",
            "intents": ["CONVERSATION"],
            "primary_intent": "CONVERSATION",
        }
    )
    inputs = iter(["hi", "/clear", "again", "/exit"])
    output = io.StringIO()

    run_repl(
        manager,
        input_fn=lambda _prompt: next(inputs),
        output=output,
    )

    assert "history: cleared" in output.getvalue()
    assert manager.calls[0]["history"] == []
    assert manager.calls[1]["message"] == "again"
    assert manager.calls[1]["history"] == []


def test_prompt_submit_accepts_current_buffer():
    event = FakePromptEvent("hello")

    _handle_prompt_submit(event)

    assert event.app.exit_result == "hello"
    assert event.app.exit_exception is None


def test_prompt_ctrl_j_inserts_newline():
    event = FakePromptEvent("hello")

    _handle_prompt_newline(event)

    assert event.app.current_buffer.text == "hello\n"
    assert event.app.current_buffer.inserted == ["\n"]


def test_prompt_ctrl_l_clears_screen_without_changing_buffer():
    event = FakePromptEvent("hello")

    _handle_prompt_clear_screen(event)

    assert event.app.current_buffer.text == "hello"
    assert event.app.renderer.clear_count == 1


def test_prompt_ctrl_c_clears_draft_without_exiting():
    event = FakePromptEvent("draft")

    _handle_prompt_cancel(event)

    assert event.app.current_buffer.text == ""
    assert event.app.current_buffer.reset_count == 1
    assert event.app.renderer.clear_count == 1
    assert event.app.exit_exception is None


def test_prompt_ctrl_c_exits_on_empty_buffer():
    event = FakePromptEvent("")

    _handle_prompt_cancel(event)

    assert event.app.current_buffer.reset_count == 0
    assert event.app.exit_exception is KeyboardInterrupt


def test_run_repl_shows_status_on_tty_output():
    manager = FakeManager(
        {
            "response": "\n\nhello\n",
            "intents": ["CONVERSATION"],
            "primary_intent": "CONVERSATION",
        }
    )
    inputs = iter(["hi", "/exit"])
    output = TtyOutput()

    run_repl(
        manager,
        input_fn=lambda _prompt: next(inputs),
        output=output,
    )

    text = output.getvalue()
    assert "\rProcessing" in text
    assert "\r             \r\nhello\n" in text
    assert "\r             \r\n\nhello\n" not in text


def test_status_line_animates_processing_frames():
    output = TtyOutput()

    with StatusLine(output, interval=0.01):
        time.sleep(0.05)

    text = output.getvalue()
    assert "\rProcessing" in text
    assert "\rProcessing." in text
    assert "\rProcessing.." in text
    assert "\rProcessing..." in text
