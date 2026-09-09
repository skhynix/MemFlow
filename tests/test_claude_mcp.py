# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import shlex
import sys
from types import SimpleNamespace

import pytest

import memflow.claude_mcp as mcp_setup
from memflow.claude_setup import apply_claude_setup_plan, build_claude_setup_plan
from memflow.cli import main


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "my project"
    root.mkdir()
    (root / ".env").write_text("QDRANT_BASE_URL=http://localhost:6333\n")
    monkeypatch.chdir(root)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude config"))
    return root


def _write_servers(project, servers):
    path = mcp_setup.claude_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"projects": {str(project): {"mcpServers": servers}}}))


@pytest.fixture
def claude_calls(project, monkeypatch):
    calls = []
    monkeypatch.setattr(mcp_setup.shutil, "which", lambda name: "/test/bin/claude")

    def run(_claude, cwd, args):
        calls.append(args)
        path = mcp_setup.claude_config_path()
        data = json.loads(path.read_text()) if path.exists() else {}
        servers = (
            data.setdefault("projects", {})
            .setdefault(str(cwd), {})
            .setdefault("mcpServers", {})
        )
        if args[0] == "add-json":
            servers["memflow"] = json.loads(args[-1])
        else:
            servers.pop("memflow")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    monkeypatch.setattr(mcp_setup, "_run_claude", run)
    return calls


def _invoke(*args):
    stdout, stderr = io.StringIO(), io.StringIO()
    code = main(["claude", *args], stdout=stdout, stderr=stderr)
    output = json.loads(stdout.getvalue()) if stdout.getvalue() else None
    return code, output, stderr.getvalue()


@pytest.mark.parametrize("custom_env", [False, True])
def test_setup_connects_hook_and_mcp_with_absolute_paths(
    project, claude_calls, custom_env
):
    _write_servers(project, {"other": {"type": "http", "url": "https://example.test"}})
    env_path = project / (".env.memflow" if custom_env else ".env")
    action = []
    if custom_env:
        env_path.write_text("")
        (project / ".env").unlink()
        action = ["--hook", "on", "--mcp", "on", "--env-file", env_path.name]

    code, status, error = _invoke("configure", *action, "--apply")

    assert (code, error) == (0, "")
    assert status["env_file"] == str(env_path)
    assert status["hook"]["installed_after"] is True
    assert status["mcp"]["installed_after"] is True
    config_path = project / ".memflow" / "claude-hook.json"
    config = json.loads(config_path.read_text())
    assert config["memflow"]["env_file"] == str(env_path)
    assert "native_catalog_mode" not in config["claude"]
    server = mcp_setup.local_mcp_server(project)
    assert server["command"] == sys.executable
    assert server["args"] == ["-m", "memflow.mcp_server", "--config", str(config_path)]
    assert shlex.split(status["hook"]["command"])[4] == str(config_path)
    saved = json.loads(mcp_setup.claude_config_path().read_text())
    assert "other" in saved["projects"][str(project)]["mcpServers"]
    assert len(claude_calls) == 1
    code, status, error = _invoke("status")
    assert (code, error) == (0, "")
    assert status["mcp"]["matches_config"] is True
    assert status["env_file_exists"] is True
    assert status["mismatches"] == []


def test_default_dry_run_does_not_write_or_run_claude(project, claude_calls):
    code, status, error = _invoke("configure")

    assert (code, error) == (0, "")
    assert status["applied"] is False
    assert status["mcp"]["changed"] is True
    assert not (project / ".memflow").exists()
    assert not (project / ".claude").exists()
    assert not mcp_setup.claude_config_path().exists()
    assert claude_calls == []


def test_repeated_setup_reuses_mcp_registration(project, claude_calls):
    assert _invoke("configure", "--apply")[0] == 0
    code, status, error = _invoke("configure", "--apply")
    assert (code, error) == (0, "")
    assert status["changed"] is False

    assert len(claude_calls) == 1
    settings = json.loads((project / ".claude" / "settings.local.json").read_text())
    assert len(settings["hooks"]["UserPromptSubmit"]) == 1


@pytest.mark.parametrize(
    "hook,mcp",
    [(None, None), ("off", "off"), ("on", "off"), ("off", "on"), ("on", "on")],
    ids=["fresh", "disconnected", "hook-only", "mcp-only", "connected"],
)
@pytest.mark.parametrize("apply", [False, True], ids=["dry-run", "apply"])
def test_env_file_only_preserves_integration_settings(
    project, claude_calls, monkeypatch, hook, mcp, apply
):
    monkeypatch.setattr("memflow.claude_catalog.Path.home", lambda: project / "home")
    if hook is not None:
        assert (
            _invoke(
                "configure",
                "--hook",
                "on",
                "--mcp",
                "on",
                "--catalog",
                "hidden_or_minimized",
                "--hook-command",
                "custom-memflow-hook",
                "--apply",
            )[0]
            == 0
        )
        disable = []
        if hook == "off":
            disable.extend(["--hook", "off"])
        if mcp == "off":
            disable.extend(["--mcp", "off"])
        if disable:
            assert _invoke("configure", *disable, "--apply")[0] == 0
    config_path = project / ".memflow" / "claude-hook.json"
    paths = [
        project / ".claude" / "settings.local.json",
        project / ".memflow" / "claude-catalog-state.json",
        mcp_setup.claude_config_path(),
    ]
    before = {path: path.read_bytes() if path.exists() else None for path in paths}
    config_before = config_path.read_bytes() if config_path.exists() else None
    claude_calls.clear()
    (project / ".env.memflow").write_text("QDRANT_COLLECTION_NAME=another\n")

    code, status, error = _invoke(
        "configure",
        "--env-file",
        ".env.memflow",
        "--apply" if apply else "--dry-run",
    )

    assert (code, error) == (0, "")
    assert status["env_file"] == str(project / ".env.memflow")
    assert status["settings_changed"] is False
    assert status["state_changed"] is False
    assert status["mcp"] is None
    assert status["hook"]["installed_after"] is (hook == "on")
    assert {
        path: path.read_bytes() if path.exists() else None for path in paths
    } == before
    assert claude_calls == []
    if apply:
        config_after = json.loads(config_path.read_text())
        assert config_after["memflow"]["env_file"] == str(project / ".env.memflow")
        if config_before is not None:
            expected_config = json.loads(config_before)
            expected_config["memflow"]["env_file"] = str(project / ".env.memflow")
            assert config_after == expected_config
    else:
        assert (
            config_path.read_bytes() if config_path.exists() else None
        ) == config_before


def test_individual_actions_preserve_other_integration_settings(project, claude_calls):
    assert _invoke("configure", "--apply")[0] == 0
    assert _invoke("configure", "--hook", "off", "--apply")[0] == 0
    assert mcp_setup.local_mcp_server(project) is not None
    assert len(claude_calls) == 1
    assert _invoke("configure", "--hook", "on", "--apply")[0] == 0
    assert _invoke("configure", "--mcp", "off", "--apply")[0] == 0
    assert mcp_setup.local_mcp_server(project) is None
    code, status, error = _invoke("status")
    assert (code, error) == (0, "")
    assert status["hook"]["installed"] is True


@pytest.mark.parametrize("action", [[], ["--mcp", "off"]])
def test_foreign_local_server_is_preserved(project, claude_calls, action):
    original = {"type": "http", "url": "https://private.test", "env": {"KEY": "secret"}}
    _write_servers(project, {"memflow": original})

    code, status, error = _invoke("configure", *action, "--apply")

    assert code == 1
    assert status is None
    assert "already exists" in error
    assert "secret" not in error
    assert mcp_setup.local_mcp_server(project) == original
    assert not (project / ".memflow").exists()
    assert claude_calls == []


def test_project_server_is_not_silently_shadowed(project, claude_calls):
    (project / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"memflow": {"command": "user-server"}}})
    )
    code, _, error = _invoke("configure", "--apply")
    assert code == 1
    assert "project or user" in error
    assert claude_calls == []


@pytest.mark.parametrize("action", [[], ["--hook", "on"], ["--mcp", "on"]])
@pytest.mark.parametrize("explicit_env_file", [False, True])
def test_missing_env_file_fails_without_changes(
    project, claude_calls, action, explicit_env_file
):
    env_args = ["--env-file", "missing.env"] if explicit_env_file else []
    if not explicit_env_file:
        (project / ".env").unlink()

    code, status, error = _invoke("configure", *action, *env_args, "--apply")

    assert code == 1
    assert status is None
    assert "environment file is not a file" in error
    assert not (project / ".memflow").exists()
    assert not (project / ".claude").exists()
    assert not mcp_setup.claude_config_path().exists()
    assert claude_calls == []


@pytest.mark.parametrize(
    "action", [[], ["--hook", "on"], ["--mcp", "on"], ["--env-file", "skills.env"]]
)
def test_missing_saved_env_file_preserves_existing_settings(
    project, claude_calls, action
):
    saved_env = project / "skills.env"
    saved_env.write_text("")
    assert (
        _invoke(
            "configure",
            "--hook",
            "on",
            "--mcp",
            "on",
            "--env-file",
            str(saved_env),
            "--apply",
        )[0]
        == 0
    )
    saved_env.unlink()
    paths = [
        project / ".memflow" / "claude-hook.json",
        project / ".claude" / "settings.local.json",
        mcp_setup.claude_config_path(),
    ]
    before = {path: path.read_bytes() for path in paths}
    claude_calls.clear()

    code, status, error = _invoke("configure", *action, "--apply")

    assert code == 1
    assert status is None
    assert f"environment file is not a file: {saved_env}" in error
    assert {path: path.read_bytes() for path in paths} == before
    assert claude_calls == []


@pytest.mark.parametrize(
    "action",
    [["--hook", "off"], ["--mcp", "off"], ["--hook", "off", "--mcp", "off"]],
)
def test_disable_integrations_without_env_file(project, claude_calls, action):
    assert _invoke("configure", "--apply")[0] == 0
    (project / ".env").unlink()

    code, _, error = _invoke("configure", *action, "--apply")

    assert (code, error) == (0, "")
    code, status, error = _invoke("status")
    assert (code, error) == (0, "")
    assert status["hook"]["installed"] is ("--hook" not in action)
    assert status["mcp"]["installed"] is ("--mcp" not in action)
    assert status["env_file_exists"] is False


def test_missing_claude_fails_before_writing_hook(project, monkeypatch):
    monkeypatch.setattr(mcp_setup.shutil, "which", lambda name: None)
    code, _, error = _invoke("configure", "--apply")
    assert code == 1
    assert "claude was not found on PATH" in error
    assert not (project / ".memflow").exists()
    assert not (project / ".claude").exists()


def test_mcp_update_failure_restores_previous_registration(
    project, claude_calls, monkeypatch
):
    assert _invoke("configure", "--apply")[0] == 0
    original = mcp_setup.local_mcp_server(project)
    old_config = (project / ".memflow" / "claude-hook.json").read_text()
    real_run = mcp_setup._run_claude
    monkeypatch.setattr(sys, "executable", "/new installation/bin/python")

    def fail_new_registration(claude, cwd, args):
        if args[0] == "add-json" and json.loads(args[-1])["command"] == sys.executable:
            raise RuntimeError("test registration failure")
        real_run(claude, cwd, args)

    monkeypatch.setattr(mcp_setup, "_run_claude", fail_new_registration)
    code, _, error = _invoke("configure", "--apply")
    assert code == 1
    assert "registration failure" in error
    assert mcp_setup.local_mcp_server(project) == original
    assert (project / ".memflow" / "claude-hook.json").read_text() == old_config


def test_registration_failure_leaves_no_hook(project, claude_calls, monkeypatch):
    def fail(*_args):
        raise RuntimeError("test registration failure")

    monkeypatch.setattr(mcp_setup, "_run_claude", fail)
    code, _, error = _invoke("configure", "--apply")
    assert code == 1
    assert "registration failure" in error
    assert not (project / ".claude").exists()
    assert not (project / ".memflow").exists()


def test_concurrent_mcp_edit_is_preserved(project, claude_calls):
    plan = build_claude_setup_plan(project_root=project)
    _write_servers(project, {"memflow": {"command": "concurrent-user-server"}})
    with pytest.raises(RuntimeError, match="changed during setup"):
        apply_claude_setup_plan(plan)
    assert claude_calls == []


def test_status_reports_mcp_drift(project, claude_calls):
    assert _invoke("configure", "--apply")[0] == 0
    _write_servers(project, {})
    code, status, error = _invoke("status")
    assert (code, error) == (0, "")
    assert status["mcp"]["installed"] is False
    assert status["mcp"]["matches_config"] is False
    assert status["mismatches"] == ["mcp_settings"]


def test_claude_config_directory_is_stable_when_changing_projects(project, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "relative config")
    observed = {}

    def run(_args, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mcp_setup.subprocess, "run", run)
    other_project = project / "other"
    mcp_setup._run_claude("claude", other_project, ["remove", "memflow"])

    assert observed["cwd"] == other_project
    assert observed["env"]["CLAUDE_CONFIG_DIR"] == str(project / "relative config")
