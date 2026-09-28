"""Assemble only the fixed measure tools owned by delivered tickets."""

from zcu_tools.mcp.core.call_log import wrap_handler
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.measure import (
    tools_device,
    tools_lifecycle,
    tools_operation,
    tools_predictor,
    tools_rpc,
    tools_screenshot,
    tools_setup,
    tools_tab,
)
from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def build_measure_tools(context: MeasureToolContext) -> ToolTable:
    """Bind a session without generating tools from GUI wire methods."""
    tools: ToolTable = {}
    for source in (
        tools_lifecycle.build_override_tools(context),
        tools_operation.build_operation_tools(context),
        tools_device.build_device_tools(context),
        tools_predictor.build_predictor_tools(context),
        tools_rpc.build_rpc_tools(context),
        tools_tab.build_tab_read_tools(context),
        tools_screenshot.build_screenshot_tools(context),
        tools_setup.build_setup_tools(context),
    ):
        for name, entry in source.items():
            if name in tools:
                raise RuntimeError(f"duplicate MCP tool {name!r}")
            tools[name] = {
                **entry,
                "handler": wrap_handler(
                    name,
                    entry["handler"],
                    redact_inputs=frozenset({"token"})
                    if name == "connect"
                    else frozenset(),
                ),
            }
    return tools
