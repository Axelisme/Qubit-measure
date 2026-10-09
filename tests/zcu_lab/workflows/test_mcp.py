"""Public workflow MCP tools and stdio behavior over a recording GUI transport."""

import base64
import json
import sys
from collections.abc import Iterator
from dataclasses import replace
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import pytest
from PIL import Image
from zcu_tools.mcp.core.bridge import (
    GuiTransportTimeoutError,
    McpBridge,
    MCPBridgeConfig,
)
from zcu_tools.mcp.core.reply import ToolReply

from tests.mcp.fluxdep._support import GuiError, GuiResponse, RecordingTransport
from zcu_lab.workflows.mcp import build_workflow_tools, run_workflow_mcp


class TextContent(TypedDict):
    """A text content block containing a JSON receipt or error diagnostic."""

    type: Literal["text"]
    text: str


class ImageContent(TypedDict):
    """A PNG content block with ASCII base64 data and its MIME type."""

    type: Literal["image"]
    data: str
    mimeType: Literal["image/png"]


class CallResult(TypedDict):
    """Ordered tool content; isError marks invocation or image delivery failure."""

    content: list[TextContent | ImageContent]
    isError: NotRequired[bool]


class CallReply(TypedDict):
    """One JSON-RPC stdio reply with the submitted request identity and result."""

    jsonrpc: Literal["2.0"]
    id: int
    result: CallResult


@pytest.fixture
def config(tmp_path: Path) -> MCPBridgeConfig:
    """Use task-local paths; this fixture never launches a GUI."""
    return MCPBridgeConfig(
        tool_prefix="workflows_",
        server_display_name="workflows",
        server_instructions="Workflow RPC contract",
        app_name="workflows",
        default_port=8768,
        mcp_version=1,
        wire_version=1,
        pid_file=tmp_path / "gui.pid",
        log_file=tmp_path / "gui.log",
        run_script_name="run_workflows_gui.py",
        app_slug="workflows",
    )


@pytest.fixture
def transport() -> RecordingTransport:
    """Record only this test's synchronous native request/reply."""
    return RecordingTransport()


@pytest.fixture
def bridge(
    config: MCPBridgeConfig, transport: RecordingTransport
) -> Iterator[McpBridge]:
    """Bind the real bridge to one recording transport and always disconnect."""
    client = McpBridge(config, transport=transport)
    yield client
    client.disconnect()


def invoke_stdio(
    config: MCPBridgeConfig,
    bridge: McpBridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    arguments: dict[str, object],
) -> CallReply:
    """Invoke the real public stdio caller in-process; EOF closes the transport."""
    request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "workflows_" + method.replace(".", "_"),
            "arguments": arguments,
        },
    }
    with (
        TextIOWrapper(
            BytesIO((json.dumps(request) + "\n").encode()), encoding="utf-8"
        ) as input_stream,
        TextIOWrapper(BytesIO(), encoding="utf-8") as output_stream,
        monkeypatch.context() as patch,
    ):
        patch.setattr(sys, "stdin", input_stream)
        patch.setattr(sys, "stdout", output_stream)
        run_workflow_mcp(config, tmp_path, bridge=bridge)
        output_stream.seek(0)
        return json.loads(output_stream.read())


@pytest.mark.parametrize(
    ("method", "arguments", "params"),
    [
        ("workflow.status", {}, {}),
        (
            "workflow.start",
            {"run_id": "run:a", "plan": [], "tunables": []},
            {"run_id": "run:a", "plan": [], "tunables": []},
        ),
        (
            "workflow.tunables",
            {"run_id": "run:a", "batch": [{"path": ["x"], "value": "q.x"}]},
            {"run_id": "run:a", "batch": [{"path": ["x"], "value": "q.x"}]},
        ),
        (
            "workflow.records",
            {"run_id": "run:a", "since": 4},
            {"run_id": "run:a", "since": 4},
        ),
        ("workflow.records", {"run_id": "run:a", "since": None}, {"run_id": "run:a"}),
        ("workflow.figures", {"run_id": "run:a"}, {"run_id": "run:a"}),
        ("workflow.pause", {"run_id": "run:a"}, {"run_id": "run:a"}),
        ("workflow.resume", {"run_id": "run:a"}, {"run_id": "run:a"}),
        ("workflow.stop", {"run_id": "run:a"}, {"run_id": "run:a"}),
    ],
)
def test_native_tool_forwards_once(
    config: MCPBridgeConfig,
    bridge: McpBridge,
    transport: RecordingTransport,
    tmp_path: Path,
    method: str,
    arguments: dict[str, object],
    params: dict[str, object],
) -> None:
    native = {"run_id": "run:a", "status": "failed", "records": []}
    transport.replies.append(GuiResponse(result=native))
    tools = build_workflow_tools(config, bridge, tmp_path)
    assert (
        tools["workflows_" + method.replace(".", "_")]["handler"](arguments) == native
    )
    assert [(r["method"], r["params"]) for r in transport.requests] == [
        (method, params)
    ]
    bridge.disconnect()


def test_form_inspection_tool_forwards_candidate_once(
    config: MCPBridgeConfig,
    bridge: McpBridge,
    transport: RecordingTransport,
    tmp_path: Path,
) -> None:
    native = {
        "schema": {},
        "input": {"guess_us": {"expr": "2 * 20"}},
        "values": [{"path": ["guess_us"], "value": 40.0}],
        "errors": [],
        "revision": 2,
    }
    transport.replies.append(GuiResponse(result=native))
    tools = build_workflow_tools(config, bridge, tmp_path)
    arguments = {
        "form": "tunables",
        "run_id": "run:a",
        "batch": [{"path": ["guess_us"], "value": {"expr": "2 * 20"}}],
    }
    assert tools["workflows_workflow_form"]["handler"](arguments) == native
    assert [(r["method"], r["params"]) for r in transport.requests] == [
        ("workflow.form", arguments),
    ]


def test_mismatched_bridge_fails_before_sending(
    config: MCPBridgeConfig,
    bridge: McpBridge,
    transport: RecordingTransport,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="config"):
        build_workflow_tools(replace(config, app_name="other"), bridge, tmp_path)
    assert transport.requests == []


@pytest.mark.parametrize("method", ["workflow.pause", "workflow.figure"])
def test_missing_required_target_fails_without_rpc(
    config: MCPBridgeConfig,
    bridge: McpBridge,
    transport: RecordingTransport,
    tmp_path: Path,
    method: str,
) -> None:
    tools = build_workflow_tools(config, bridge, tmp_path)
    with pytest.raises(ValueError, match="run_id"):
        tools["workflows_" + method.replace(".", "_")]["handler"]({})
    assert transport.requests == []


def test_figure_stdio_delivers_one_image_and_disconnects(
    config: MCPBridgeConfig,
    bridge: McpBridge,
    transport: RecordingTransport,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixels = BytesIO()
    Image.new("RGB", (2, 3), "red").save(pixels, format="PNG")
    png = pixels.getvalue()
    transport.replies.append(
        GuiResponse(
            result={
                "run_id": "run:a",
                "name": "T1/live",
                "png_base64": base64.b64encode(png).decode(),
            }
        )
    )
    reply = invoke_stdio(
        config,
        bridge,
        tmp_path,
        monkeypatch,
        "workflow.figure",
        {"run_id": "run:a", "name": "T1/live"},
    )
    assert reply["id"] == 7
    result = reply["result"]
    assert not result.get("isError", False)
    text, picture = result["content"]
    assert text["type"] == "text"
    assert picture["type"] == "image"
    assert json.loads(text["text"]) == {"run_id": "run:a", "name": "T1/live"}
    assert picture["type"] == "image" and picture["mimeType"] == "image/png"
    assert base64.b64decode(picture["data"]) == png
    assert [(r["method"], r["params"]) for r in transport.requests] == [
        ("workflow.figure", {"run_id": "run:a", "name": "T1/live"})
    ]
    assert not bridge.is_connected and transport.close_count == 1


@pytest.mark.parametrize(
    "encoded", ["not base64!", base64.b64encode(b"not PNG").decode()]
)
def test_corrupt_figure_is_tool_error_without_retry(
    config: MCPBridgeConfig,
    bridge: McpBridge,
    transport: RecordingTransport,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoded: str,
) -> None:
    transport.replies.append(
        GuiResponse(
            result={
                "run_id": "run:a",
                "name": "T1/live",
                "png_base64": encoded,
            }
        )
    )
    reply = invoke_stdio(
        config,
        bridge,
        tmp_path,
        monkeypatch,
        "workflow.figure",
        {"run_id": "run:a", "name": "T1/live"},
    )
    assert reply["result"].get("isError", False)
    assert len(transport.requests) == 1
    assert transport.close_count == 1


def test_native_invocation_error_retains_reason_and_disconnects(
    config: MCPBridgeConfig,
    bridge: McpBridge,
    transport: RecordingTransport,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport.replies.append(
        GuiResponse(error=GuiError("invalid_params", "wrong run", "run_mismatch"))
    )
    reply = invoke_stdio(
        config, bridge, tmp_path, monkeypatch, "workflow.stop", {"run_id": "old"}
    )
    assert reply["result"].get("isError", False)
    error = reply["result"]["content"][0]
    assert error["type"] == "text"
    assert "run_mismatch" in error["text"]
    assert len(transport.requests) == 1 and transport.close_count == 1


@pytest.mark.parametrize(
    "failure",
    [ConnectionError("wire down"), GuiTransportTimeoutError("workflow.status", 6.0)],
)
def test_transport_failure_propagates_without_retry(
    config: MCPBridgeConfig,
    bridge: McpBridge,
    transport: RecordingTransport,
    tmp_path: Path,
    failure: Exception,
) -> None:
    transport.failure = failure
    tools = build_workflow_tools(config, bridge, tmp_path)
    with pytest.raises(type(failure)) as caught:
        tools["workflows_workflow_status"]["handler"]({})
    assert caught.value is failure
    assert len(transport.requests) == 1


def test_disconnect_tool_never_sends_stop(
    config: MCPBridgeConfig,
    bridge: McpBridge,
    transport: RecordingTransport,
    tmp_path: Path,
) -> None:
    tools = build_workflow_tools(config, bridge, tmp_path)
    assert isinstance(tools["workflows_disconnect"]["handler"]({}), str)
    assert transport.requests == [] and transport.close_count == 1


def test_direct_figure_reply_remains_a_structured_tool_reply(
    config: MCPBridgeConfig,
    bridge: McpBridge,
    transport: RecordingTransport,
    tmp_path: Path,
) -> None:
    pixels = BytesIO()
    Image.new("RGB", (1, 1), "blue").save(pixels, format="PNG")
    transport.replies.append(
        GuiResponse(
            result={
                "run_id": "run:a",
                "name": "selected",
                "png_base64": base64.b64encode(pixels.getvalue()).decode(),
            }
        )
    )
    result = build_workflow_tools(config, bridge, tmp_path)[
        "workflows_workflow_figure"
    ]["handler"]({"run_id": "run:a", "name": "selected"})
    assert isinstance(result, ToolReply)
    assert result.data == {"run_id": "run:a", "name": "selected"}
    assert result.images[0].data == pixels.getvalue()
