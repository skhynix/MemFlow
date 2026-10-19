# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Root CLI contracts, including rejection before service initialization."""

import io

import pytest

from memflow.cli import main


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CLI parsing must not initialize external services")

    monkeypatch.setattr("memflow.manager.MemFlow.__init__", forbidden)
    monkeypatch.setattr("memflow.store.QdrantStore.__init__", forbidden)
    monkeypatch.setattr("memflow.store.VectorStore._embed_chunk", forbidden)
    monkeypatch.setattr("memflow.store.VectorStore._embed_chunk_async", forbidden)


def test_no_arguments_prints_root_help():
    output = io.StringIO()
    assert main([], stdout=output) == 0
    text = output.getvalue()
    assert "usage: memflow" in text
    assert "skill" in text
    assert "claude" in text
    assert "chat" not in text


@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["skill", "--help"],
        ["skill", "add", "--help"],
        ["skill", "sync", "--help"],
        ["skill", "list", "--help"],
        ["claude", "--help"],
        ["claude", "configure", "--help"],
        ["claude", "status", "--help"],
    ],
)
def test_help_exits_zero(args, capsys):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "usage: memflow" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize(
    "args",
    [
        ["chat"],
        ["-p", "hello"],
        ["--prompt", "hello"],
        ["--execute"],
        ["--no-history"],
        ["--verbose"],
        ["--user-id", "alice"],
        ["unknown"],
        ["skill", "search", "hello"],
        ["skill", "get", "id"],
        ["skill", "delete", "id"],
    ],
)
def test_invalid_commands_exit_two(args, capsys):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
