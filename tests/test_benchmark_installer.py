# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""The installer manages SkillRet; SkillReason has its own downloader."""

import pytest

from benchmark import install_benchmark


@pytest.mark.parametrize("command", ["skill_ret_bench", "all"])
def test_installer_forwards_skillret_options(monkeypatch, command):
    calls = []
    monkeypatch.setattr(
        "sys.argv",
        ["install_benchmark", command, "--force", "--commit-hash-skillret", "abc123"],
    )
    monkeypatch.setattr(
        install_benchmark,
        "install_skill_ret_bench",
        lambda **kwargs: calls.append(kwargs),
    )
    assert install_benchmark.main() == 0
    assert calls == [{"force": True, "commit_hash": "abc123"}]


@pytest.mark.parametrize(
    "args",
    [
        ["proced_mem_bench"],
        ["wikihow_procedure_silver"],
        ["skill_reason_bench"],
        ["all", "--raw-dir", "/tmp/raw"],
        ["all", "--commit-hash-procmem", "abc123"],
    ],
)
def test_unsupported_installer_input_has_no_side_effects(monkeypatch, args):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid arguments must not install data")

    monkeypatch.setattr("sys.argv", ["install_benchmark", *args])
    monkeypatch.setattr(install_benchmark, "install_skill_ret_bench", forbidden)
    with pytest.raises(SystemExit) as exc:
        install_benchmark.main()
    assert exc.value.code == 2
