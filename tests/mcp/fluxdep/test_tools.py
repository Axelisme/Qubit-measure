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
from zcu_tools.mcp.core.bridge import GuiTransportTimeoutError
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.fluxdep.assembly import build_fluxdep_server

from tests.mcp.fluxdep._support import Client, GuiError, GuiResponse


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


@pytest.mark.parametrize(
    ("method", "arguments", "params"),
    [
        ("project.info", {}, {}),
        (
            "project.setup",
            {"chip_name": "chip", "qub_name": "q"},
            {"chip_name": "chip", "qub_name": "q"},
        ),
        ("spectrum.list", {}, {}),
        ("spectrum.snapshot", {"name": "譜:a*"}, {"name": "譜:a*"}),
        (
            "spectrum.load",
            {
                "filepath": "/data/raw.hdf5",
                "spec_type": "TwoTone",
                "transpose_axes": False,
            },
            {
                "filepath": "/data/raw.hdf5",
                "spec_type": "TwoTone",
                "transpose_axes": False,
            },
        ),
        (
            "spectrum.load_processed",
            {"filepath": "/data/spectrums.hdf5"},
            {"filepath": "/data/spectrums.hdf5"},
        ),
        ("spectrum.remove", {"name": "a"}, {"name": "a"}),
        ("spectrum.set_active", {"name": None}, {}),
        ("spectrum.reset_alignment", {"name": "a"}, {"name": "a"}),
        ("spectrum.reset_points", {"name": "a"}, {"name": "a"}),
        ("selection.snapshot", {}, {}),
        ("selection.pointcloud", {}, {}),
        ("fit.result", {}, {}),
        ("fit.search", {}, {}),
        ("operation.status", {"token": None}, {}),
        ("operation.cancel", {"token": 91}, {"token": 91}),
        (
            "export.spectrums",
            {"filepath": "/out/spectrums.hdf5", "overwrite": True},
            {"filepath": "/out/spectrums.hdf5", "overwrite": True},
        ),
        (
            "fit.export_params",
            {"savepath": "/out/params.json"},
            {"savepath": "/out/params.json"},
        ),
        ("state.check", {}, {}),
    ],
)
def test_tools_forward_one_request_without_hidden_reads(
    client: Client,
    method: str,
    arguments: dict[str, object],
    params: dict[str, object],
) -> None:
    result = {"native_receipt": method}
    client.transport.replies.append(GuiResponse(result=result))
    actual = client.server.tools["fluxdep_" + method.replace(".", "_")]["handler"](
        arguments
    )
    assert actual == result
    assert len(client.transport.requests) == 1
    sent = client.transport.requests[0]
    assert sent["method"] == method
    assert sent["params"] == params


def test_fit_numeric_bounds_and_nested_transitions_preserve_json_values(
    client: Client,
) -> None:
    arguments = {
        "database_path": "/data/search.hdf5",
        "EJb": [2.5, 8],
        "ECb": [0.1, 0.6],
        "ELb": [0.2, 1.3],
        "transitions": {"main": [[0, 1], [1, 2]], "r_f": 7.2},
    }
    client.transport.replies.append(GuiResponse(result={"updated": True}))
    assert client.server.tools["fluxdep_fit_set_params"]["handler"](arguments) == {
        "updated": True
    }
    assert len(client.transport.requests) == 1
    assert client.transport.requests[0]["params"] == arguments


def test_factory_rejects_an_inconsistent_bridge_binding(
    client: Client, tmp_path: Path
) -> None:
    config = replace(client.server.bridge.config, app_name="other-app")
    with pytest.raises(ValueError, match="supplied Fluxdep config"):
        build_fluxdep_server(config, tmp_path, bridge=client.server.bridge)
    assert client.transport.requests == []
    assert client.transport.is_open


def png_bytes() -> bytes:
    """Create a valid small PNG without selecting or mutating a plotting backend."""
    buffer = BytesIO()
    with Image.new("RGB", (12, 8), color="red") as image:
        image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        ("interactive.read", {}),
        ("spectrum.interactive.open", {"name": "譜:a*", "kind": "twotone"}),
        (
            "spectrum.interactive.command",
            {
                "name": "譜:a*",
                "context_id": 12,
                "command": "stroke",
                "params": {"points": [[0.2, 1.1], [0.4, 1.2]], "width": 0.05},
            },
        ),
        ("selection.interactive.open", {}),
        ("selection.interactive.command", {"context_id": 12, "command": "apply"}),
    ],
)
def test_interactive_receipts_deliver_same_call_image_and_preserve_identity(
    client: Client, method: str, arguments: dict[str, object]
) -> None:
    image = png_bytes()
    figure = {"png_b64": base64.b64encode(image).decode("ascii"), "bytes": len(image)}
    context = {
        "context_id": 12,
        "closed": True,
        "state": {"selected_count": 2},
        "figure": figure,
    }
    effect = {"command": "finish", "changes": {"selected": -3}}
    result = {"context": context, "effect": effect}
    client.transport.replies.append(GuiResponse(result=result))
    reply = client.server.tools["fluxdep_" + method.replace(".", "_")]["handler"](
        arguments
    )
    assert isinstance(reply, ToolReply)
    assert [item.data for item in reply.images] == [image]
    assert reply.data == {
        "context": {
            **context,
            "figure": {"mime_type": "image/png", "bytes": len(image)},
        },
        "effect": effect,
    }
    assert figure["png_b64"] == base64.b64encode(image).decode("ascii")
    assert context["figure"] == figure
    assert len(client.transport.requests) == 1
    assert client.transport.requests[0]["method"] == method
    assert client.transport.requests[0]["params"] == arguments


def test_inactive_read_has_no_image_or_extra_request(client: Client) -> None:
    client.transport.replies.append(
        GuiResponse(result={"context": None, "effect": None})
    )
    reply = client.server.tools["fluxdep_interactive_read"]["handler"]({})
    assert isinstance(reply, ToolReply)
    assert reply.data == {"context": None, "effect": None}
    assert reply.images == ()
    assert len(client.transport.requests) == 1


def test_stdio_serializes_image_after_text_and_disconnects_only(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = png_bytes()
    client.transport.replies.append(
        GuiResponse(
            result={
                "context": {
                    "context_id": 9,
                    "figure": {
                        "png_b64": base64.b64encode(image).decode("ascii"),
                        "bytes": len(image),
                    },
                },
                "effect": None,
            }
        )
    )
    reply = invoke_stdio(client, monkeypatch, "fluxdep_interactive_read", {})
    content = reply["result"]["content"]
    assert [block["type"] for block in content] == ["text", "image"]
    first, second = content
    assert first["type"] == "text" and second["type"] == "image"
    assert json.loads(first["text"])["context"]["context_id"] == 9
    assert base64.b64decode(second["data"], validate=True) == image
    assert second["mimeType"] == "image/png"
    assert client.transport.close_count == 1
    assert len(client.transport.requests) == 1


@pytest.mark.parametrize(
    "encoded", ["not base64!", base64.b64encode(b"not PNG").decode("ascii")]
)
def test_image_failure_returns_tool_error_without_replay(
    client: Client, monkeypatch: pytest.MonkeyPatch, encoded: str
) -> None:
    client.transport.replies.append(
        GuiResponse(
            result={
                "context": {
                    "context_id": 12,
                    "closed": True,
                    "figure": {"png_b64": encoded, "bytes": 7},
                },
                "effect": {"command": "finish"},
            }
        )
    )
    reply = invoke_stdio(
        client,
        monkeypatch,
        "fluxdep_spectrum_interactive_command",
        {"name": "a", "context_id": 12, "command": "finish"},
    )
    assert reply["result"].get("isError") is True
    assert len(client.transport.requests) == 1
    assert client.transport.requests[0]["method"] == "spectrum.interactive.command"


def test_gui_stale_error_preserves_native_diagnostics_without_read_or_retry(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.transport.replies.append(
        GuiResponse(error=GuiError("stale", "source changed", "state_not_seen"))
    )
    reply = invoke_stdio(client, monkeypatch, "fluxdep_fit_search", {})
    assert reply["result"].get("isError") is True
    block = reply["result"]["content"][0]
    assert block["type"] == "text"
    text = block["text"]
    assert "stale" in text and "source changed" in text and "state_not_seen" in text
    assert len(client.transport.requests) == 1


@pytest.mark.parametrize(
    "result",
    [
        {"token": 91, "reason": "timeout", "outcome": None, "feedback": None},
        {
            "token": 91,
            "reason": "completed",
            "outcome": {"status": "failed", "error": "database failed"},
            "feedback": None,
        },
    ],
)
def test_await_timeout_and_failed_outcome_are_data_not_invocation_errors(
    client: Client, result: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    client.transport.replies.append(GuiResponse(result=result))
    reply = invoke_stdio(
        client, monkeypatch, "fluxdep_operation_await", {"token": 91, "timeout": 0}
    )
    block = reply["result"]["content"][0]
    assert block["type"] == "text"
    assert json.loads(block["text"]) == result
    assert not reply["result"].get("isError", False)
    assert len(client.transport.requests) == 1
    assert client.transport.requests[0]["params"] == {"token": 91, "timeout": 0.0}


def test_await_uses_declared_transport_budget(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    budgets: list[float] = []
    original = client.server.bridge.send_rpc_raw

    def record(
        method: str, params: dict[str, object], timeout_seconds: float
    ) -> dict[str, object]:
        budgets.append(timeout_seconds)
        return original(method, params, timeout_seconds)

    monkeypatch.setattr(client.server.bridge, "send_rpc_raw", record)
    client.transport.replies.append(
        GuiResponse(
            result={"token": 91, "reason": "timeout", "outcome": None, "feedback": None}
        )
    )
    client.server.tools["fluxdep_operation_await"]["handler"]({"token": 91})
    assert budgets == [36.0]
    assert client.transport.requests[0]["params"] == {"token": 91}


def test_transport_timeout_has_no_reconnect_or_replay(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.transport.failure = GuiTransportTimeoutError("fit.search", 6.0)
    reply = invoke_stdio(client, monkeypatch, "fluxdep_fit_search", {})
    assert reply["result"].get("isError") is True
    assert len(client.transport.requests) == 1
    assert client.transport.close_count == 1


def test_connect_and_launch_delegate_explicit_lifecycle_arguments(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    connects: list[tuple[int, str | None]] = []
    launches: list[tuple[Path, int, str | None, bool]] = []

    def connect(port: int, token: str | None = None) -> str:
        connects.append((port, token))
        return "connected"

    def launch(
        repo_root: Path,
        port: int,
        token: str | None = None,
        *,
        auto_connect: bool = True,
    ) -> str:
        launches.append((repo_root, port, token, auto_connect))
        return "launched"

    monkeypatch.setattr(client.server.bridge, "connect", connect)
    monkeypatch.setattr(client.server.bridge, "launch", launch)
    assert (
        client.server.tools["fluxdep_connect"]["handler"](
            {"port": 9876, "token": "test-token"}
        )
        == "connected"
    )
    assert connects == [(9876, "test-token")]
    assert (
        client.server.tools["fluxdep_launch"]["handler"](
            {"port": 9877, "auto_connect": False}
        )
        == "launched"
    )
    assert len(launches) == 1
    assert launches[0][1:] == (9877, None, False)
    assert client.transport.requests == []
