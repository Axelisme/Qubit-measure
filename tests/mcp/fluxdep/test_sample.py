"""Sampling intents compose named native drafts and compact detector receipts."""

import json
import sys
from io import BytesIO, TextIOWrapper
from typing import Literal

import pytest

from tests.mcp.fluxdep._support import Client, GuiResponse


def capture(kind: Literal["onetone", "twotone"], threshold: float) -> dict[str, object]:
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
            "closed": False,
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
            GuiResponse(
                result={"spec_type": "OneTone" if kind == "onetone" else "TwoTone"}
            ),
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
        ("spectrum.snapshot", {"name": "s"}),
        ("spectrum.interactive.open", {"name": "s", "kind": kind}),
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
