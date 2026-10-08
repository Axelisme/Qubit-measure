"""MCP stdio protocol behavior of run_stdio_loop and tool argument coercion."""

from __future__ import annotations

import base64
import io
import json
import sys
from collections.abc import Callable
from typing import Any

import pytest
from zcu_tools.gui.remote.param_spec import JsonType
from zcu_tools.mcp.core.reply import PngImage, ToolReply
from zcu_tools.mcp.core.stdio_server import (
    McpServerConfig,
    StdioLoopHooks,
    ToolTable,
    coerce_arg,
    run_stdio_loop,
)

_CONFIG = McpServerConfig(
    tool_prefix="demo_",
    server_display_name="Demo",
    server_instructions="demo instructions",
)


class _ReasonError(RuntimeError):
    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


def _tool(handler: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    return {"handler": handler, "description": "demo tool", "inputSchema": {}}


def _run(
    monkeypatch: pytest.MonkeyPatch,
    requests: list[dict[str, Any]],
    tools: ToolTable,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    stdin = io.TextIOWrapper(
        io.BytesIO("".join(json.dumps(r) + "\n" for r in requests).encode()),
        encoding="utf-8",
    )
    out = io.BytesIO()
    stdout = io.TextIOWrapper(out, encoding="utf-8", write_through=True)
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    run_stdio_loop(_CONFIG, tools, **kwargs)
    return [json.loads(line) for line in out.getvalue().decode().splitlines()]


def test_initialize_and_tools_list_describe_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replies = _run(
        monkeypatch,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ],
        {"demo_echo": _tool(lambda args: args)},
        server_version="2.3.4",
    )

    assert [r["id"] for r in replies] == [1, 2]
    info = replies[0]["result"]
    assert info["serverInfo"] == {"name": "Demo", "version": "2.3.4"}
    assert info["instructions"] == "demo instructions"
    assert replies[1]["result"]["tools"] == [
        {"name": "demo_echo", "description": "demo tool", "inputSchema": {}}
    ]


def test_tool_call_returns_compact_json_text_and_appends_reply_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replies = _run(
        monkeypatch,
        [
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": "demo_echo", "arguments": {"a": 1}},
            }
        ],
        {"demo_echo": _tool(lambda args: {"got": args})},
        hooks=StdioLoopHooks(on_each_reply=lambda: [{"type": "text", "text": "extra"}]),
    )

    assert replies == [
        {
            "jsonrpc": "2.0",
            "id": 7,
            "result": {
                "content": [
                    {"type": "text", "text": '{"got":{"a":1}}'},
                    {"type": "text", "text": "extra"},
                ]
            },
        }
    ]


def test_images_belong_to_one_reply_and_precede_hook_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images = (PngImage(b"first PNG bytes"), PngImage(b"second PNG bytes"))

    def fail(_args: dict[str, Any]) -> Any:
        raise RuntimeError("failed")

    replies = _run(
        monkeypatch,
        [
            {
                "jsonrpc": "2.0",
                "id": i,
                "method": "tools/call",
                "params": {"name": name, "arguments": {}},
            }
            for i, name in enumerate(
                ["with_images", "failure", "empty_reply", "plain_dict", "plain_text"]
            )
        ],
        {
            "with_images": _tool(lambda _: ToolReply({"value": 2}, images)),
            "failure": _tool(fail),
            "empty_reply": _tool(lambda _: ToolReply({"value": 3})),
            "plain_dict": _tool(lambda _: {"value": 4}),
            "plain_text": _tool(lambda _: "unchanged"),
        },
        hooks=StdioLoopHooks(on_each_reply=lambda: [{"type": "text", "text": "hook"}]),
    )

    hook = {"type": "text", "text": "hook"}
    assert replies[0]["result"]["content"] == [
        {"type": "text", "text": '{"value":2}'},
        *[
            {
                "type": "image",
                "mimeType": "image/png",
                "data": base64.b64encode(item.data).decode("ascii"),
            }
            for item in images
        ],
        hook,
    ]
    error = replies[1]["result"]
    assert error["isError"] is True
    assert len(error["content"]) == 1
    assert "failed" in error["content"][0]["text"]
    for reply, expected_text in zip(
        replies[2:], ['{"value":3}', '{"value":4}', "unchanged"], strict=True
    ):
        assert reply["result"]["content"] == [
            {"type": "text", "text": expected_text},
            hook,
        ]


@pytest.mark.parametrize("is_error", [False, True])
def test_partial_outcome_preserves_data_and_images_with_explicit_error_flag(
    monkeypatch: pytest.MonkeyPatch, is_error: bool
) -> None:
    data = {
        "status": "failed",
        "saved_images": [{"figure_name": "fit", "image_path": "/saved/fit.png"}],
        "error": {"phase": "image_save", "message": "second export failed"},
    }
    image = PngImage(b"confirmed preview bytes")
    replies = _run(
        monkeypatch,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "partial", "arguments": {}},
            }
        ],
        {"partial": _tool(lambda _: ToolReply(data, (image,), is_error=is_error))},
    )

    result = replies[0]["result"]
    assert result.get("isError", False) is is_error
    assert json.loads(result["content"][0]["text"]) == data
    assert result["content"][1:] == [
        {
            "type": "image",
            "mimeType": "image/png",
            "data": base64.b64encode(image.data).decode("ascii"),
        }
    ]


def test_runtime_error_reports_reason_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[str] = []

    def fail(_args: dict[str, Any]) -> Any:
        raise _ReasonError("tab is busy", "busy")

    replies = _run(
        monkeypatch,
        [
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "demo_fail", "arguments": {}},
            }
        ],
        {"demo_fail": _tool(fail)},
        hooks=StdioLoopHooks(on_error=errors.append),
    )

    result = replies[0]["result"]
    assert result["isError"] is True
    assert result["content"][0]["text"] == (
        "Error executing tool 'demo_fail': tab is busy\nreason: busy"
    )
    assert errors == ["MCP tool 'demo_fail' dispatch failed"]


def test_unexpected_error_keeps_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_args: dict[str, Any]) -> Any:
        raise KeyError("missing")

    replies = _run(
        monkeypatch,
        [
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "demo_fail", "arguments": {}},
            }
        ],
        {"demo_fail": _tool(fail)},
    )

    text = replies[0]["result"]["content"][0]["text"]
    assert text.startswith("Error executing tool 'demo_fail': 'missing'\n")
    assert "Traceback" in text


def test_unknown_tool_and_method_reply_not_found_but_notifications_do_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replies = _run(
        monkeypatch,
        [
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "demo_missing", "arguments": {}},
            },
            {"jsonrpc": "2.0", "id": 6, "method": "resources/list"},
            {"jsonrpc": "2.0", "method": "notifications/unknown"},
        ],
        {},
    )

    assert replies == [
        {
            "jsonrpc": "2.0",
            "id": 5,
            "error": {"code": -32601, "message": "Method not found: demo_missing"},
        },
        {
            "jsonrpc": "2.0",
            "id": 6,
            "error": {"code": -32601, "message": "Method not found: resources/list"},
        },
    ]


def test_hooks_run_at_start_and_on_stdin_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    replies = _run(
        monkeypatch,
        [],
        {},
        hooks=StdioLoopHooks(
            on_start=lambda: events.append("start"),
            on_cleanup=lambda: events.append("cleanup"),
        ),
    )

    assert replies == []
    assert events == ["start", "cleanup"]


def test_malformed_line_is_reported_and_the_loop_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[str] = []
    stdin = io.TextIOWrapper(
        io.BytesIO(b'not json\n{"jsonrpc":"2.0","id":9,"method":"tools/list"}\n'),
        encoding="utf-8",
    )
    out = io.BytesIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(
        sys, "stdout", io.TextIOWrapper(out, encoding="utf-8", write_through=True)
    )
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    run_stdio_loop(_CONFIG, {}, hooks=StdioLoopHooks(on_error=errors.append))

    assert errors == ["MCP loop exception"]
    assert json.loads(out.getvalue().decode())["id"] == 9


def test_coerce_arg_rejects_non_list_array() -> None:
    with pytest.raises(TypeError, match="expected list"):
        coerce_arg("1,2", JsonType.ARRAY)
