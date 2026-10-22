"""Public dispersive MCP assembly forwards entry launch inputs to the process."""

import subprocess
from pathlib import Path

import pytest
from zcu_tools.mcp.core.bridge import MCPBridgeConfig
from zcu_tools.mcp.dispersive import server


def test_launch_forwards_entry_context_to_gui(tmp_path, monkeypatch):
    commands: list[list[str]] = []

    class ExitedProcess:
        pid = 99999999

        def __init__(self, command: list[str], **_kwargs: object):
            commands.append(command)

        def poll(self) -> int:
            return 2

    monkeypatch.setattr(subprocess, "Popen", ExitedProcess)
    script = tmp_path / "scripts" / "run_dispersive_gui.py"
    script.parent.mkdir()
    script.write_text("# synthetic launch target\n")
    config = MCPBridgeConfig(
        tool_prefix="dispersive_",
        server_display_name="test",
        server_instructions="",
        app_name="dispersive",
        app_slug="dispersive",
        default_port=0,
        mcp_version=1,
        wire_version=5,
        pid_file=tmp_path / "pid",
        log_file=tmp_path / "gui.log",
        run_script_name=script.name,
    )
    app = server.build_server(config, tmp_path)
    with pytest.raises(RuntimeError, match="process exited during startup"):
        app.tools["dispersive_launch"]["handler"](
            {
                "project_root": str(tmp_path / "project"),
                "entry": "synthetic",
                "component": "q1",
                "port": 0,
                "auto_connect": False,
            }
        )
    assert len(commands) == 1
    assert commands[0][1] == str(script)
    assert commands[0][-6:] == [
        "--project-root",
        str(tmp_path / "project"),
        "--entry",
        "synthetic",
        "--component",
        "q1",
    ]
