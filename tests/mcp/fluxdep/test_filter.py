"""Filter decisions use GUI authority and native draft/published evidence."""

import base64
import json
from io import BytesIO, StringIO

import pytest
from PIL import Image
from zcu_tools.mcp.core.stdio_server import run_stdio_loop

from tests.mcp.fluxdep._support import Client, GuiResponse


def invoke(client: Client, command: str, **arguments: object):
    """Invoke the public stdio seam in-process; unknown tools are wire failures."""
    output = StringIO()
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "fluxdep_filter", "arguments": {"command": command, **arguments}}}
    run_stdio_loop(client.server.bridge.config, client.server.tools, input_stream=StringIO(json.dumps(request) + "\n"), output_stream=output)
    reply = json.loads(output.getvalue())
    assert "result" in reply, reply
    assert not reply["result"].get("isError", False), reply
    return reply["result"]["content"]


def native(*, closed: bool = False) -> dict[str, object]:
    """Deliberately distinct input and kept counts; per-source truth is native."""
    return {
        "context": {"context_id": 23, "kind": "selection", "closed": closed, "can_undo": not closed,
            "state": {"min_distance": 0.025, "selected": [True, True, True, False], "selected_count": 2,
                "per_spectrum": {"first": [2, 3], "second": [0, 1]}}},
        "effect": {"changes": {"points_added": 0, "points_removed": 2}},
    }


@pytest.mark.parametrize("command, image", [("stroke", False), ("undo", True), ("finish", False)])
def test_filter_composes_commands_counts_and_native_evidence(client: Client, command: str, image: bool):
    client.transport.replies.extend([GuiResponse(result=native()), GuiResponse(result=native(closed=command == "finish"))])
    png = BytesIO()
    with Image.new("RGB", (12, 8), color="blue") as producer:
        producer.save(png, format="PNG")
    if command == "finish" or image:
        client.transport.replies.append(GuiResponse(result={"png_b64": base64.b64encode(png.getvalue()).decode("ascii"), "bytes": len(png.getvalue())}))
    params = {"vertices": [[0.5, 4.5]], "width": 0.03, "mode": "erase"} if command == "stroke" else {}
    content = invoke(client, command, params=params, image=image)
    summary = json.loads(content[0]["text"])
    assert summary["stage"] == ("filtered" if command == "finish" else "filter")
    assert summary["pending"] is (command != "finish")
    assert summary["min_distance"] == 0.025
    assert summary["selected"] == ({"count": 2, "total": 4} if command == "finish" else {"count": 2, "total": 4, "changes": [0, 2]})
    assert summary["per_spectrum"] == {"first": [2, 3], "second": [0, 1]}
    if command != "finish":
        assert summary["can_undo"] is True
    expected = [("selection.interactive.open", {}), ("selection.interactive.command", {"context_id": 23, "command": command, "params": params})]
    if command == "finish" or image:
        expected.append(("selection.render", {}) if command == "finish" else ("interactive.render", {"name": "selection"}))
        assert content[1]["type"] == "image"
        assert base64.b64decode(content[1]["data"]) == png.getvalue()
    else:
        assert len(content) == 1
    assert [(request["method"], request["params"]) for request in client.transport.requests] == expected
