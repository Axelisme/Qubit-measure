"""MCP preserves primitive values for GUI-owned contract validation."""

import pytest

from tests.mcp.fluxdep._support import Client, GuiError, GuiResponse


@pytest.mark.parametrize(
    ("name", "arguments", "field"),
    [
        (
            "fluxdep_spectrum_interactive_command",
            {"name": "a", "context_id": True, "command": "undo"},
            "context_id",
        ),
        (
            "fluxdep_spectrum_interactive_command",
            {"name": "a", "context_id": 1.9, "command": "undo"},
            "context_id",
        ),
        (
            "fluxdep_spectrum_interactive_command",
            {"name": "a", "context_id": "1", "command": "undo"},
            "context_id",
        ),
        (
            "fluxdep_spectrum_interactive_command",
            {
                "name": "a",
                "context_id": 1,
                "command": "stroke",
                "params": [["points", 1]],
            },
            "params",
        ),
        ("fluxdep_spectrum_snapshot", {"name": 123}, "name"),
        (
            "fluxdep_spectrum_load",
            {
                "filepath": "/data/raw.hdf5",
                "spec_type": "OneTone",
                "transpose_axes": "false",
            },
            "transpose_axes",
        ),
        ("fluxdep_operation_await", {"token": 1, "timeout": "0.5"}, "timeout"),
    ],
)
def test_invalid_scalar_is_not_rewritten_before_gui_validation(
    client: Client, name: str, arguments: dict[str, object], field: str
) -> None:
    client.transport.replies.append(
        GuiResponse(error=GuiError("invalid_params", "wrong primitive type"))
    )
    with pytest.raises(RuntimeError, match="invalid_params.*wrong primitive type"):
        client.server.tools[name]["handler"](arguments)
    assert len(client.transport.requests) == 1
    params = client.transport.requests[0]["params"]
    assert isinstance(params, dict)
    observed = params[field]
    assert observed == arguments[field]
    assert type(observed) is type(arguments[field])
