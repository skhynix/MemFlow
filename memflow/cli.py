# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Interactive console interface for MemFlow.
"""

from __future__ import annotations

import argparse
import sys
import threading
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from memflow.manager import MemFlow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memflow.py",
        description="Start an interactive MemFlow chat() console.",
    )
    parser.add_argument(
        "--user-id",
        default="default",
        help="User scope for memory operations. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print chat() intent and handler trace metadata.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow EXECUTE intents to run immediately.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not pass prior turns back into chat().",
    )
    return parser


def _format_count(value: object) -> str:
    try:
        return str(len(value))  # type: ignore[arg-type]
    except TypeError:
        return "unknown"


def _strip_surrounding_blank_lines(text: str) -> str:
    """Remove blank padding lines without changing indentation in content."""
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


class StatusLine:
    """Show transient progress while a synchronous chat() call is running."""

    def __init__(
        self,
        output: TextIO,
        frames: tuple[str, ...] = (
            "Processing",
            "Processing.",
            "Processing..",
            "Processing...",
        ),
        interval: float = 0.25,
    ) -> None:
        self._output = output
        self._frames = frames
        self._interval = interval
        self._enabled = bool(getattr(output, "isatty", lambda: False)())
        self._max_width = max((len(frame) for frame in frames), default=0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> None:
        if not self._enabled or not self._frames:
            return
        self._write_frame(self._frames[0])
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def _animate(self) -> None:
        index = 1
        while not self._stop.wait(self._interval):
            self._write_frame(self._frames[index % len(self._frames)])
            index += 1

    def _write_frame(self, frame: str) -> None:
        padding = " " * (self._max_width - len(frame))
        self._output.write(f"\r{frame}{padding}")
        self._output.flush()

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._enabled or not self._frames:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._output.write("\r" + (" " * self._max_width) + "\r")
        self._output.flush()


def format_verbose_trace(
    result: dict,
    *,
    user_id: str,
    allow_execute: bool,
    history_count: int,
) -> str:
    """Format chat() metadata for CLI verbose mode."""
    lines = [
        "[trace] chat()",
        f"  user_id: {user_id}",
        f"  allow_execute: {str(allow_execute).lower()}",
        f"  history_messages: {history_count}",
    ]

    intents = result.get("intents")
    if intents:
        lines.append(f"  intents: {', '.join(str(i) for i in intents)}")
    else:
        intent = result.get("intent", "unknown")
        lines.append(f"  intents: {intent}")

    lines.append(f"  primary_intent: {result.get('primary_intent', 'unknown')}")

    if result.get("requires_confirmation"):
        lines.append("  execute: waiting_for_confirmation")
        return "\n".join(lines)

    handler_results = result.get("handler_results") or {}
    if not handler_results:
        lines.append("  handlers: none")
        return "\n".join(lines)

    lines.append("  handlers:")
    for intent, handler_result in handler_results.items():
        lines.append(f"    {intent}:")
        if intent == "SEARCH":
            search_results = handler_result.get("results", [])
            lines.append(f"      search.results: {_format_count(search_results)}")
            for item in search_results[:3]:
                procedure = getattr(item, "procedure", None)
                title = getattr(procedure, "title", None)
                score = getattr(item, "score", None)
                if title is not None and score is not None:
                    lines.append(f"      - {title} (score={score:.3f})")
        elif intent == "ADD":
            data = handler_result.get("data") or {}
            memory_type = data.get("type", "unknown")
            lines.append(f"      memory_type: {memory_type}")
            if data.get("routed_to"):
                lines.append(f"      routed_to: {data['routed_to']}")
            if data.get("skipped"):
                lines.append(f"      skipped: {data['skipped']}")
            stored = data.get("results", [])
            lines.append(f"      stored: {_format_count(stored)}")
            for item in stored:
                title = item.get("title") if isinstance(item, dict) else None
                event = item.get("event") if isinstance(item, dict) else None
                if title:
                    suffix = f" ({event})" if event else ""
                    lines.append(f"      - {title}{suffix}")
        elif intent == "EXECUTE":
            data = handler_result.get("data") or {}
            run_result = data.get("result")
            plan = getattr(run_result, "plan", None)
            steps = getattr(plan, "steps", []) if plan else []
            step_results = getattr(run_result, "step_results", []) if run_result else []
            lines.append("      search: used_by_run_context")
            lines.append(f"      plan.steps: {_format_count(steps)}")
            lines.append(f"      execute.results: {_format_count(step_results)}")
            learned = getattr(run_result, "learned", None)
            if learned is not None:
                lines.append(f"      learned: {getattr(learned, 'title', 'Procedure')}")
        else:
            lines.append("      search: context_lookup")

    return "\n".join(lines)


def _handle_command(
    line: str,
    *,
    state: dict,
    output: TextIO,
) -> bool:
    parts = line.strip().split()
    command = parts[0].lower()
    value = parts[1].lower() if len(parts) > 1 else None

    if command in {"/exit", "/quit"}:
        state["running"] = False
        return True

    if command == "/help":
        print("/verbose [on|off]  Toggle trace output.", file=output)
        print("/execute [on|off]  Toggle immediate EXECUTE handling.", file=output)
        print("/user <id>         Change the user scope.", file=output)
        print("/exit             Quit.", file=output)
        return True

    if command == "/verbose":
        if value in {"on", "true", "1"}:
            state["verbose"] = True
        elif value in {"off", "false", "0"}:
            state["verbose"] = False
        else:
            state["verbose"] = not state["verbose"]
        print(f"verbose: {'on' if state['verbose'] else 'off'}", file=output)
        return True

    if command == "/execute":
        if value in {"on", "true", "1"}:
            state["allow_execute"] = True
        elif value in {"off", "false", "0"}:
            state["allow_execute"] = False
        else:
            state["allow_execute"] = not state["allow_execute"]
        print(f"execute: {'on' if state['allow_execute'] else 'off'}", file=output)
        return True

    if command == "/user":
        if len(parts) < 2:
            print(f"user: {state['user_id']}", file=output)
        else:
            state["user_id"] = parts[1]
            print(f"user: {state['user_id']}", file=output)
        return True

    return False


def run_repl(
    manager: "MemFlow | None" = None,
    *,
    manager_factory: Callable[[], "MemFlow"] | None = None,
    user_id: str = "default",
    verbose: bool = False,
    allow_execute: bool = False,
    use_history: bool = True,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> int:
    history: list[dict] = []
    state = {
        "running": True,
        "user_id": user_id,
        "verbose": verbose,
        "allow_execute": allow_execute,
    }
    active_manager = manager

    while state["running"]:
        try:
            message = input_fn("> ")
        except EOFError:
            print(file=output)
            break
        except KeyboardInterrupt:
            print(file=output)
            break

        message = message.strip()
        if not message:
            continue

        if message.startswith("/") and _handle_command(
            message, state=state, output=output
        ):
            continue

        context = list(history) if use_history else None
        init_error = None
        result = None
        with StatusLine(output):
            if active_manager is None:
                if manager_factory is None:
                    from memflow.manager import MemFlow

                    manager_factory = MemFlow
                try:
                    active_manager = manager_factory()
                except ModuleNotFoundError as exc:
                    init_error = f"Unable to initialize MemFlow: missing optional dependency '{exc.name}'."
                except Exception as exc:
                    init_error = f"Unable to initialize MemFlow: {exc}"

            if init_error is None:
                result = active_manager.chat(
                    message,
                    user_id=state["user_id"],
                    history=context,
                    allow_execute=state["allow_execute"],
                )

        if init_error is not None:
            print(init_error, file=output)
            continue

        response = _strip_surrounding_blank_lines(result.get("response", ""))
        if getattr(output, "isatty", lambda: False)():
            print(file=output)
        print(response, file=output)

        if state["verbose"]:
            print(
                format_verbose_trace(
                    result,
                    user_id=state["user_id"],
                    allow_execute=state["allow_execute"],
                    history_count=len(history),
                ),
                file=output,
            )

        if use_history:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})

    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_repl(
        manager_factory=_create_manager,
        user_id=args.user_id,
        verbose=args.verbose,
        allow_execute=args.execute,
        use_history=not args.no_history,
    )


def _create_manager() -> "MemFlow":
    from memflow.manager import MemFlow

    return MemFlow()


if __name__ == "__main__":
    raise SystemExit(main())
