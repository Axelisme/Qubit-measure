"""Assembly rejects ambiguous exposure; generated handlers validate arguments."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from zcu_tools.gui.remote.method_spec import McpMethodPolicy, MethodSpec
from zcu_tools.mcp.measure.assembly import build_measure_tools

from ._support import make_client


@pytest.mark.parametrize(
    ("specs", "message"),
    [
        (
            {
                "fake.method": MethodSpec(
                    1.0,
                    "",
                    mcp=McpMethodPolicy.override("missing_tool", reason="manual"),
                )
            },
            "not registered",
        ),
        (
            {
                "fake.one": MethodSpec(1.0, "", tool_name="collision"),
                "fake.two": MethodSpec(1.0, "", tool_name="collision"),
            },
            "generated MCP tool name collision",
        ),
        (
            {"fake.method": MethodSpec(1.0, "", tool_name="gui_tab_open")},
            "collide with manual tools",
        ),
        (
            {
                "fake.method": MethodSpec(
                    1.0, "", tool_name="bad", mcp=McpMethodPolicy.internal("wire-only")
                )
            },
            "cannot use MethodSpec.tool_name",
        ),
        (
            {
                "fake.method": MethodSpec(
                    1.0,
                    "",
                    tool_name="bad",
                    mcp=McpMethodPolicy.override("gui_tab_open", reason="manual"),
                )
            },
            "cannot use MethodSpec.tool_name",
        ),
    ],
)
def test_assembly_rejects_invalid_exposure_before_any_rpc(
    tmp_path: Path,
    specs: dict[str, MethodSpec],
    message: str,
) -> None:
    client = make_client(tmp_path)
    with pytest.raises(RuntimeError, match=message):
        build_measure_tools(replace(client.context, method_specs=specs))
    assert client.transport.sent == []


@pytest.mark.parametrize("arguments", [{}, {"adapter_name": None}])
def test_generated_handler_rejects_missing_required_arguments_before_rpc(
    tmp_path: Path,
    arguments: dict[str, Any],
) -> None:
    client = make_client(tmp_path)
    with pytest.raises(ValueError, match="missing 'adapter_name'"):
        client.call("gui_tab_new", arguments)
    assert client.transport.sent == []


@pytest.mark.parametrize("arguments", [{}, {"tab_id": None}, {"tab_id": "t"}])
def test_generated_optional_arguments_are_omitted_unless_provided(
    tmp_path: Path,
    arguments: dict[str, Any],
) -> None:
    client = make_client(tmp_path)
    client.transport.replies["tab.snapshot"] = {"ok": True, "result": {"tabs": []}}
    assert client.call("gui_tab_snapshot", arguments) == {"tabs": []}
    assert client.transport.sent[0] == (
        "tab.snapshot",
        {"tab_id": "t"} if arguments.get("tab_id") else {},
    )
