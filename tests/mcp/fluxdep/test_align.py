"""Alignment decisions compose named RPCs without retaining GUI business state."""

import base64
from io import BytesIO

import pytest
from PIL import Image
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.rpc import GuiInvocationError

from tests.mcp.fluxdep._support import Client, GuiError, GuiResponse


def native(
    half: float = 1.0,
    integer: float = 3.0,
    *,
    busy: bool = False,
    closed: bool = False,
    error: str | None = None,
) -> dict[str, object]:
    """One app-authoritative named line draft capture."""
    return {
        "context": {
            "context_id": 12,
            "closed": closed,
            "kind": "line",
            "spectrum_name": "one",
            "state": {
                "flux_half": half,
                "flux_int": integer,
                "conjugate": False,
                "magnitude_only": True,
            },
            "info": {"alignment_busy": busy, "alignment_error": error},
        },
        "effect": None,
    }


def invoke(client: Client, **args: object) -> ToolReply:
    value = client.server.tools["fluxdep_align"]["handler"]({"name": "one", **args})
    return value if isinstance(value, ToolReply) else ToolReply(value)


def test_move_line_composes_conjugate_and_reports_only_changed_lines(client: Client):
    client.transport.replies.extend(
        GuiResponse(result=result) for result in [native(), native(), native(2.0)]
    )
    result = invoke(
        client,
        command="move_line",
        params={"role": "half", "position": 2.0, "conjugate": True},
    )
    assert result.data == {
        "name": "one",
        "stage": "align",
        "pending": True,
        "lines": {"half": 2.0, "integer": 3.0, "period": 2.0},
        "changes": {"half": [1.0, 2.0]},
    }
    assert [(r["method"], r["params"]) for r in client.transport.requests] == [
        ("spectrum.interactive.open", {"name": "one", "kind": "line"}),
        (
            "spectrum.interactive.command",
            {
                "name": "one",
                "context_id": 12,
                "command": "set_conjugate",
                "params": {"enabled": True},
            },
        ),
        (
            "spectrum.interactive.command",
            {
                "name": "one",
                "context_id": 12,
                "command": "move_line",
                "params": {"role": "half", "position": 2.0},
            },
        ),
    ]
    assert not result.images


@pytest.mark.parametrize(
    "params",
    [
        {"role": "half", "position": 2.0},
        {"role": "half", "position": 2.0, "conjugate": False},
    ],
)
def test_move_line_resets_conjugate_when_omitted_or_false(
    client: Client, params: dict[str, object]
):
    client.transport.replies.extend(
        GuiResponse(result=result) for result in [native(), native(), native(2.0)]
    )
    invoke(client, command="move_line", params=params)
    assert client.transport.requests[1]["params"] == {
        "name": "one",
        "context_id": 12,
        "command": "set_conjugate",
        "params": {"enabled": False},
    }
    assert client.transport.requests[2]["params"] == {
        "name": "one",
        "context_id": 12,
        "command": "move_line",
        "params": {"role": "half", "position": 2.0},
    }


@pytest.mark.parametrize("error", [None, "alignment failed"])
def test_auto_align_polls_named_context_to_terminal(client: Client, error: str | None):
    client.transport.replies.extend(
        GuiResponse(result=result)
        for result in [
            native(),
            native(busy=True),
            native(busy=True),
            native(1.0, 4.0, error=error),
        ]
    )
    result = invoke(client, command="auto_align")
    assert result.data["lines"] == {"half": 1.0, "integer": 4.0, "period": 6.0}
    assert result.data["changes"] == {"integer": [3.0, 4.0]}
    assert result.data["auto_align"] == {
        "status": "finished" if error is None else "failed",
        "error": error,
    }
    assert [r["method"] for r in client.transport.requests] == [
        "spectrum.interactive.open",
        "spectrum.interactive.command",
        "interactive.read",
        "interactive.read",
    ]
    assert [r["params"] for r in client.transport.requests[2:]] == [
        {"name": "one", "context_id": 12},
        {"name": "one", "context_id": 12},
    ]


@pytest.mark.parametrize(
    ("command", "image", "render_method"),
    [("finish", False, "spectrum.render"), ("move_line", True, "interactive.render")],
)
def test_finish_and_image_override_return_native_evidence(
    client: Client, command: str, image: bool, render_method: str
):
    buffer = BytesIO()
    with Image.new("RGB", (12, 8), "red") as png:
        png.save(buffer, format="PNG")
    data = buffer.getvalue()
    opened = native()
    opened["discarded"] = "sample"
    client.transport.replies.extend(
        [
            GuiResponse(result=opened),
            GuiResponse(result=native(closed=command == "finish")),
            GuiResponse(
                result={
                    "png_b64": base64.b64encode(data).decode("ascii"),
                    "bytes": len(data),
                }
            ),
        ]
    )
    if command == "move_line":
        client.transport.replies.insert(1, GuiResponse(result=native()))
    result = invoke(client, command=command, image=image)
    assert result.data["pending"] is (command != "finish")
    assert result.data["stage"] == ("aligned" if command == "finish" else "align")
    assert result.data["discarded"] == "sample"
    assert len(result.images) == 1
    assert client.transport.requests[-1]["method"] == render_method
    assert client.transport.requests[-1]["params"] == {"name": "one"}


def test_align_finish_projects_native_invalidated_without_deriving_targets(client: Client):
    finished = native(closed=True)
    finished["invalidated"] = ["native points", "native mask", "fit"]
    buffer = BytesIO()
    with Image.new("RGB", (12, 8), "red") as image:
        image.save(buffer, format="PNG")
    png = buffer.getvalue()
    client.transport.replies.extend([
        GuiResponse(result=native()), GuiResponse(result=finished),
        GuiResponse(result={"png_b64": base64.b64encode(png).decode("ascii"), "bytes": len(png)}),
    ])
    reply = invoke(client, command="finish")
    assert reply.data.get("invalidated") == ["native points", "native mask", "fit"]
    assert [request["method"] for request in client.transport.requests] == [
        "spectrum.interactive.open", "spectrum.interactive.command", "spectrum.render",
    ]


def test_auto_align_reports_guarded_context_interruption(client: Client):
    client.transport.replies.extend(
        [
            GuiResponse(result=native()),
            GuiResponse(result=native(busy=True)),
            GuiResponse(
                error=GuiError(
                    code="precondition_failed",
                    message="interactive identity or target changed",
                    reason="interactive_context_changed",
                )
            ),
        ]
    )
    with pytest.raises(
        GuiInvocationError, match="auto_align interrupted.*cancelled or replaced"
    ) as refusal:
        invoke(client, command="auto_align")
    assert refusal.value.code == "precondition_failed"
    assert refusal.value.reason == "interactive_context_changed"
    assert client.transport.requests[-1]["params"] == {"name": "one", "context_id": 12}
