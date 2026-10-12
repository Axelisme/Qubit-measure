"""Sampling intents compose named native drafts and compact detector receipts."""

import base64
import json
import sys
from io import BytesIO, TextIOWrapper
from typing import Literal

import pytest
from PIL import Image
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.rpc import GuiInvocationError

from tests.mcp.fluxdep._support import Client, GuiError, GuiResponse


def capture(
    kind: Literal["onetone", "twotone"], threshold: float, *, closed: bool = False
) -> dict[str, object]:
    """Scripted app capture with independent point and mask counts."""
    state: dict[str, object] = {"threshold": threshold}
    if kind == "onetone":
        state["peak_indices"] = [1, 3, 5] if threshold == 1 else [1]
    else:
        state.update(
            sigma=1.0,
            smooth_method="wavelet",
            point_count=8 if threshold == 1 else 5,
            masked_count=20,
        )
    return {
        "context": {
            "context_id": 12,
            "kind": kind,
            "closed": closed,
            "state": state,
            "can_undo": True,
        },
        "effect": None
        if threshold == 1
        else {
            "changes": {
                "peaks_added": 0,
                "peaks_removed": 2,
                "points_added": 1,
                "points_removed": 4,
                "mask_added": 0,
                "mask_removed": 0,
            }
        },
    }


@pytest.mark.parametrize("kind", ["onetone", "twotone"])
def test_settings_report_before_after_and_positional_point_changes(
    client: Client, monkeypatch: pytest.MonkeyPatch, kind: Literal["onetone", "twotone"]
):
    command = "set_threshold" if kind == "onetone" else "set_settings"
    opened = capture(kind, 1.0)
    opened["discarded"] = "align"
    client.transport.replies.extend(
        [
            GuiResponse(result=opened),
            GuiResponse(result=capture(kind, 2.0)),
        ]
    )
    expected: dict[str, object] = {
        "name": "s",
        "stage": "sample",
        "pending": True,
        "settings": {"threshold": 2.0},
        "changes": {"threshold": [1.0, 2.0]},
        "points": {
            "count": 1 if kind == "onetone" else 5,
            "changes": [0, 2] if kind == "onetone" else [1, 4],
        },
        "discarded": "align",
    }
    if kind == "twotone":
        expected["settings"] = {
            "threshold": 2.0,
            "sigma": 1.0,
            "smooth_method": "wavelet",
        }
        expected["mask"] = {"pixels": 20, "changes": [0, 0]}
        expected["can_undo"] = True
    request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "fluxdep_sample",
            "arguments": {
                "name": "s",
                "command": command,
                "params": {"threshold": 2.0},
            },
        },
    }
    with (
        TextIOWrapper(BytesIO((json.dumps(request) + "\n").encode())) as input_stream,
        TextIOWrapper(BytesIO()) as output_stream,
    ):
        monkeypatch.setattr(sys, "stdin", input_stream)
        monkeypatch.setattr(sys, "stdout", output_stream)
        client.server.main()
        output_stream.seek(0)
        reply = json.loads(output_stream.read())
    assert reply.get("error") is None, reply
    assert not reply["result"].get("isError", False), reply
    assert json.loads(reply["result"]["content"][0]["text"]) == expected
    assert len(reply["result"]["content"]) == 1
    assert [(r["method"], r["params"]) for r in client.transport.requests] == [
        ("spectrum.interactive.open", {"name": "s", "kind": "sample"}),
        (
            "spectrum.interactive.command",
            {
                "name": "s",
                "context_id": 12,
                "command": command,
                "params": {"threshold": 2.0},
            },
        ),
    ]


@pytest.mark.parametrize(
    ("kind", "command", "image", "render_method"),
    [
        ("onetone", "finish", False, "spectrum.render"),
        ("twotone", "finish", False, "spectrum.render"),
        ("onetone", "set_threshold", True, "interactive.render"),
        ("twotone", "stroke", True, "interactive.render"),
        ("twotone", "undo", False, None),
    ],
)
def test_sampling_finish_and_image_override_use_native_evidence(
    client: Client,
    kind: Literal["onetone", "twotone"],
    command: str,
    image: bool,
    render_method: str | None,
):
    params: dict[str, object] = {}
    if command == "set_threshold":
        params = {"threshold": 2.0}
    elif command == "stroke":
        params = {
            "vertices": [[-1.0, 4.5], [1.0, 5.1]],
            "width": 0.08,
            "mode": "select",
        }
    client.transport.replies.extend(
        [
            GuiResponse(result=capture(kind, 2.0)),
            GuiResponse(result=capture(kind, 2.0, closed=command == "finish")),
        ]
    )
    if command == "finish":
        client.transport.replies.append(
            GuiResponse(
                result={
                    "points": {
                        "fluxs": [0.12345, 0.98765],
                        "freqs": [4.56789, 5.43210],
                    },
                }
            )
        )
    buffer = BytesIO()
    with Image.new("RGB", (12, 8), "red") as png:
        png.save(buffer, format="PNG")
    data = buffer.getvalue()
    if render_method:
        client.transport.replies.append(
            GuiResponse(
                result={
                    "png_b64": base64.b64encode(data).decode("ascii"),
                    "bytes": len(data),
                }
            )
        )
    value = client.server.tools["fluxdep_sample"]["handler"](
        {
            "name": "s",
            "command": command,
            "params": params,
            "image": image,
        }
    )
    reply = value if isinstance(value, ToolReply) else ToolReply(value)
    assert client.transport.requests[0]["params"] == {"name": "s", "kind": "sample"}
    assert client.transport.requests[1]["params"] == {
        "name": "s",
        "context_id": 12,
        "command": command,
        "params": params,
    }
    if command == "finish":
        assert reply.data == {
            "name": "s",
            "stage": "sampled",
            "pending": False,
            "points": {
                "count": 2,
                "flux_range": [0.123, 0.988],
                "freq_range": [4.568, 5.432],
            },
        }
        assert client.transport.requests[-2]["method"] == "spectrum.snapshot"
        assert client.transport.requests[-2]["params"] == {"name": "s"}
    else:
        assert reply.data["stage"] == "sample"
        assert reply.data["pending"] is True
        assert "changes" not in reply.data
        if kind == "twotone":
            assert reply.data["mask"] == {"pixels": 20, "changes": [0, 0]}
            assert reply.data["can_undo"] is True
    if render_method:
        assert len(reply.images) == 1
        assert reply.images[0].data == data
        assert client.transport.requests[-1]["method"] == render_method
        assert client.transport.requests[-1]["params"] == {"name": "s"}
    else:
        assert not reply.images
        assert len(client.transport.requests) == 2


@pytest.mark.parametrize(
    ("code", "message", "reason"),
    [
        ("invalid_params", "no spectrum named 'absent'", "unknown_spectrum"),
        ("precondition_failed", "one-tone picking requires an aligned spectrum", None),
    ],
)
def test_sampling_native_admission_is_not_preread_or_retried(
    client: Client,
    code: str,
    message: str,
    reason: str | None,
):
    client.transport.replies.append(GuiResponse(error=GuiError(code, message, reason)))
    with pytest.raises(GuiInvocationError) as rejected:
        client.server.tools["fluxdep_sample"]["handler"](
            {"name": "absent", "command": "finish"}
        )
    assert [(r["method"], r["params"]) for r in client.transport.requests] == [
        ("spectrum.interactive.open", {"name": "absent", "kind": "sample"}),
    ]
    assert rejected.value.code == code
    assert rejected.value.reason == reason
    assert rejected.value.message == message
