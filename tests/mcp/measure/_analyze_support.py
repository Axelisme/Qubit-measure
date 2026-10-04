"""Recording analysis fixtures and stdio delivery support."""

import base64
import io
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from threading import Event
from typing import Any

import pytest
from zcu_tools.mcp.core.stdio_server import run_stdio_loop

from ._support import MeasureClient, RpcResponder, make_client

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


AMBIGUOUS_SAVE_ERRORS = {
    "handler_timeout": {"code": "timeout", "message": "GUI handler timed out"},
    "encoding_failed": {
        "code": "internal",
        "reason": "response_encoding_failed",
        "message": "request may have executed",
    },
}


def analysis_client(
    tmp_path: Path,
    clients: list[MeasureClient],
    responder: RpcResponder | None = None,
    *,
    writeback: Mapping[str, object] | None = None,
) -> MeasureClient:
    """Build the analysis seam with a confirmed empty draft unless supplied."""
    client = make_client(tmp_path, responder)
    client.transport.replies["tab.writeback_preview"] = lambda params: {
        "ok": True,
        "result": dict(writeback)
        if writeback is not None
        else {"has_draft": False, "items": [], "destination_context": {}},
    }
    clients.append(client)
    return client


def call_stdio(
    monkeypatch: pytest.MonkeyPatch,
    client: MeasureClient,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    stdin = io.TextIOWrapper(
        io.BytesIO((json.dumps(request) + "\n").encode()), encoding="utf-8"
    )
    out = io.BytesIO()
    stdout = io.TextIOWrapper(out, encoding="utf-8", write_through=True)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdin", stdin)
        patch.setattr(sys, "stdout", stdout)
        run_stdio_loop(client.context.config, client.tools)
    return json.loads(out.getvalue(), parse_constant=_reject_json_constant)["result"]


def call_full_execution_stdio(
    monkeypatch: pytest.MonkeyPatch,
    client: MeasureClient,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Keep wire delivery checks while explicitly reading captured native detail."""
    reply = call_stdio(monkeypatch, client, name, arguments)
    data = json.loads(reply["content"][0]["text"])
    full = call_stdio(
        monkeypatch,
        client,
        "status",
        {"execution": data["execution"], "detail": "full"},
    )
    if "elapsed_s" in data:
        full_data = json.loads(full["content"][0]["text"])
        full_data["elapsed_s"] = data["elapsed_s"]
        full["content"][0]["text"] = json.dumps(
            full_data, separators=(",", ":"), allow_nan=False
        )
    reply["content"][0] = full["content"][0]
    return reply


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def stdio_data(reply: dict[str, Any]) -> dict[str, Any]:
    assert not reply.get("isError"), reply
    return json.loads(reply["content"][0]["text"], parse_constant=_reject_json_constant)


def assert_figure(reply: dict[str, Any], *, present: bool) -> Path | None:
    data = stdio_data(reply)
    if not present:
        assert data["figure"] is None
        assert len(reply["content"]) == 1
        return None
    path = Path(data["figure"])
    assert path.is_absolute()
    assert path.read_bytes() == PNG
    assert reply["content"][1:] == [
        {
            "type": "image",
            "mimeType": "image/png",
            "data": base64.b64encode(PNG).decode("ascii"),
        }
    ]
    return path


def sent_methods(client: MeasureClient) -> list[str]:
    return [
        name
        for name, _ in client.transport.sent
        if name not in ("wire.version", "rpc.catalog")
    ]


def analysis_start_reply(
    params: dict[str, Any], invalidated: list[str], *, interactive: bool = False
) -> dict[str, Any]:
    return {
        "operation_id": 71,
        "interactive": interactive,
        "params": params,
        "invalidated_on_success": invalidated,
    }


def analysis_result_reply(
    pane: str, names: list[str], params: dict[str, Any]
) -> dict[str, Any]:
    return {
        "summary": {"frequency": 5.0},
        "invalid": [],
        "params": params,
        "operation_state": {
            f"{pane}_state": {
                "figure_names": names,
                "has_figure": bool(names),
                "available": True,
            },
        },
    }


def hold_wire_reply(
    client: MeasureClient, monkeypatch: pytest.MonkeyPatch, method: str
) -> Event:
    """Model a GUI request that was dispatched but has not replied."""
    sent = Event()
    send_line = client.transport.send_line

    def send(payload):
        if payload["method"] == method:
            client.transport.sent.append((payload["method"], payload["params"]))
            sent.set()
        else:
            send_line(payload)

    monkeypatch.setattr(client.transport, "send_line", send)
    return sent


def assert_ambiguous_save_error(data: dict[str, Any], failure: str) -> None:
    if failure not in AMBIGUOUS_SAVE_ERRORS:
        return
    envelope = AMBIGUOUS_SAVE_ERRORS[failure]
    assert data["error"]["code"] == envelope["code"]
    assert data["error"]["reason"] == envelope.get("reason", "gui_handler_timeout")
    assert envelope["message"] in data["error"]["message"]


def inject_analysis_rejection(
    client: MeasureClient,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejection = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "superseded_result",
            "message": "original analysis was replaced",
        },
    }
    if failure == "writeback_rejected":
        client.transport.replies["tab.writeback_preview"] = rejection
    elif failure == "writeback_timeout":
        client.transport.replies["tab.writeback_preview"] = {
            "ok": False,
            "error": {"code": "timeout", "message": "draft read timed out"},
        }
    elif failure == "result_rejected":
        client.transport.replies["tab.get_analyze_result"] = rejection
    elif failure == "save_rejected" or failure in AMBIGUOUS_SAVE_ERRORS:
        save_error = (
            {"ok": False, "error": AMBIGUOUS_SAVE_ERRORS[failure]}
            if failure in AMBIGUOUS_SAVE_ERRORS
            else rejection
        )
        client.transport.replies["tab.save_image"] = lambda params: (
            save_error
            if params["figure_name"] == "residual"
            else {"ok": True, "result": {"image_path": "/actual/fit.png"}}
        )
    elif failure == "preview_rejected":
        client.transport.replies["tab.get_figure"] = rejection
    elif failure == "local_write":

        def reject_write(path, data):
            raise OSError("preview filesystem is full")

        monkeypatch.setattr(Path, "write_bytes", reject_write)
