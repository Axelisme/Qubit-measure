"""Tool tables keep their session and transport when other tables are built."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from zcu_tools.gui.app.main.services.remote.method_specs import METHOD_SPECS
from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig, ToolTable
from zcu_tools.mcp.measure.assembly import build_measure_tools
from zcu_tools.mcp.measure.session import GuiRpcError, MeasureMcpSession
from zcu_tools.mcp.measure.tool_context import MeasureToolContext


@dataclass
class RecordingTransport:
    label: str
    version: int
    error: dict[str, Any] | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)
    deliver_reply: Callable[[dict[str, Any]], None] | None = None
    is_open: bool = True

    def attach(
        self,
        deliver_reply: Callable[[dict[str, Any]], None],
        deliver_event: Callable[[dict[str, Any]], None],
        on_closed: Callable[[], None],
    ) -> None:
        self.deliver_reply = deliver_reply

    def send_line(self, payload: dict[str, Any]) -> None:
        self.requests.append(payload)
        method = payload["method"]
        if method == "resources.versions":
            result = {
                "versions": {
                    "tab:t:cfg": self.version,
                    "tab:t": self.version,
                    "tab:t:result": self.version,
                    "tab:t:analyze": self.version,
                    "context": self.version,
                }
            }
        elif method == "tab.get_cfg":
            result = {"tree": {"label": self.label}}
        elif method in {"tab.set_cfg", "tab.load_data"}:
            result = {"owner": self.label}
        else:
            raise AssertionError(f"unexpected RPC {method}")
        if self.deliver_reply is None:
            raise AssertionError("transport was not attached")
        if self.error is not None and method in {"tab.set_cfg", "tab.load_data"}:
            self.deliver_reply({"id": payload["id"], "ok": False, "error": self.error})
        else:
            self.deliver_reply({"id": payload["id"], "ok": True, "result": result})

    def close(self) -> None:
        self.is_open = False


def make_tools(tmp_path: Path, transport: RecordingTransport) -> ToolTable:
    config = MCPBridgeConfig(
        tool_prefix="gui_",
        server_display_name="measure-test",
        server_instructions="",
        app_name="gui",
        default_port=8765,
        mcp_version=74,
        wire_version=55,
        pid_file=tmp_path / "unused.pid",
        log_file=tmp_path / "unused.log",
        run_script_name="unused.py",
    )
    session = MeasureMcpSession(
        config,
        resolve_connect_port=lambda config, requested: config.default_port,
        port_is_open=lambda port: False,
    )
    bridge = McpBridge(config, on_event=session.deliver_event)
    session.attach_bridge(bridge)
    bridge.set_transport(transport)
    context = MeasureToolContext(
        config,
        session,
        METHOD_SPECS,
        resolve_connect_port=lambda config, requested: config.default_port,
    )
    return build_measure_tools(context)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("gui_tab_set_cfg", {"tab_id": "t", "edits": [{"path": "gain", "value": 0.5}]}),
        ("gui_tab_load_data", {"tab_id": "t", "data_path": "A.h5"}),
    ],
)
def test_tool_errors_preserve_wire_code_and_reason(
    tmp_path: Path,
    tool: str,
    arguments: dict[str, Any],
) -> None:
    transport = RecordingTransport(
        "A",
        11,
        error={
            "code": "precondition_failed",
            "reason": "invalid_data_file",
            "message": "invalid input",
        },
    )
    tools = make_tools(tmp_path, transport)
    with pytest.raises(GuiRpcError, match="invalid input") as error:
        tools[tool]["handler"](arguments)
    assert (error.value.code, error.value.reason) == (
        "precondition_failed",
        "invalid_data_file",
    )


def test_interleaved_tables_keep_override_and_generated_calls_in_own_session(
    tmp_path: Path,
) -> None:
    first = RecordingTransport("A", 11)
    second = RecordingTransport("B", 29)
    tools_a = make_tools(tmp_path, first)
    assert tools_a["gui_tab_get_cfg"]["handler"]({"tab_id": "t"}) == {
        "tree": {"label": "A"},
    }
    tools_b = make_tools(tmp_path, second)
    assert tools_b["gui_tab_get_cfg"]["handler"]({"tab_id": "t"}) == {
        "tree": {"label": "B"},
    }

    assert tools_a["gui_tab_set_cfg"]["handler"](
        {
            "tab_id": "t",
            "edits": [{"path": "gain", "value": 0.5}],
        }
    ) == {"owner": "A"}
    assert tools_b["gui_tab_set_cfg"]["handler"](
        {
            "tab_id": "t",
            "edits": [{"path": "gain", "value": 0.8}],
        }
    ) == {"owner": "B"}
    for tools, owner in ((tools_a, "A"), (tools_b, "B")):
        assert tools["gui_tab_load_data"]["handler"](
            {
                "tab_id": "t",
                "data_path": f"{owner}.h5",
            }
        ) == {"owner": owner}
    for transport, version, gain in ((first, 11, 0.5), (second, 29, 0.8)):
        calls = [
            call
            for call in transport.requests
            if call["method"] != "resources.versions"
        ]
        assert [call["method"] for call in calls] == [
            "tab.get_cfg",
            "tab.set_cfg",
            "tab.load_data",
        ]
        assert calls[1]["params"] == {
            "tab_id": "t",
            "edits": [{"path": "gain", "value": gain}],
        }
        assert calls[2]["params"]["expected_versions"] == {
            "tab:t": version,
            "tab:t:result": version,
            "tab:t:analyze": version,
            "context": version,
        }
