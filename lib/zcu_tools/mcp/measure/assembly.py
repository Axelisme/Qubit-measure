"""Assemble only the fixed measure tools owned by completed tickets."""

from zcu_tools.mcp.core.bridge import ToolTable
from zcu_tools.mcp.core.call_log import wrap_handler
from zcu_tools.mcp.measure import tools_lifecycle, tools_operation, tools_rpc
from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def build_measure_tools(context: MeasureToolContext) -> ToolTable:
    """Bind a session without generating tools from GUI wire methods."""
    tools: ToolTable = {}
    for source in (
        tools_lifecycle.build_override_tools(context),
        tools_operation.build_operation_tools(context),
        tools_rpc.build_rpc_tools(context),
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
