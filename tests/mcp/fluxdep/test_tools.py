"""Public Fluxdep tool/stdio contracts over a recording GUI transport."""

from __future__ import annotations

import base64
import json
import sys
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
from zcu_tools.mcp.fluxdep.assembly import build_fluxdep_server

from tests.mcp.fluxdep._support import (
    Client,
    GuiError,
    GuiResponse,
    RecordingTransport,
)


class TextContent(TypedDict):
    """One text content block; text is the compact JSON receipt or error."""

    type: Literal["text"]
    text: str


class ImageContent(TypedDict):
    """One PNG block; data is ASCII base64, mimeType is image/png."""

    type: Literal["image"]
    mimeType: Literal["image/png"]
    data: str


class CallResult(TypedDict):
    """Ordered tool content; isError appears only for invocation/delivery failure."""

    content: list[TextContent | ImageContent]
    isError: NotRequired[bool]


class CallReply(TypedDict):
    """Stdio response to this test call ID, carrying one CallResult."""

    jsonrpc: Literal["2.0"]
    id: int
    result: CallResult


def invoke_stdio(
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    arguments: dict[str, object],
) -> CallReply:
    """Call one public tool through the public in-process stdio loop."""
    request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
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
        client.server.main()
        output_stream.seek(0)
        return json.loads(output_stream.read())






def test_factory_rejects_an_inconsistent_bridge_binding(
    client: Client, tmp_path: Path
) -> None:
    config = replace(client.server.bridge.config, app_name="other-app")
    with pytest.raises(ValueError, match="supplied Fluxdep config"):
        build_fluxdep_server(config, tmp_path, bridge=client.server.bridge)
    assert client.transport.requests == []
    assert client.transport.is_open












def test_gui_stale_error_preserves_native_diagnostics_without_read_or_retry(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.transport.replies.append(
        GuiResponse(error=GuiError("stale", "source changed", "state_not_seen"))
    )
    reply = invoke_stdio(client, monkeypatch, "fluxdep_spectrum_remove", {"name": "a"})
    assert reply["result"].get("isError") is True
    block = reply["result"]["content"][0]
    assert block["type"] == "text"
    text = block["text"]
    assert "stale" in text and "source changed" in text and "state_not_seen" in text
    assert len(client.transport.requests) == 1






def test_transport_timeout_has_no_reconnect_or_replay(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.transport.failure = GuiTransportTimeoutError("spectrum.remove", 6.0)
    reply = invoke_stdio(client, monkeypatch, "fluxdep_spectrum_remove", {"name": "a"})
    assert reply["result"].get("isError") is True
    assert len(client.transport.requests) == 1
    assert client.transport.close_count == 1




