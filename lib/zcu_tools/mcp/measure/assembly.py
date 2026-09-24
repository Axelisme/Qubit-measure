"""Assemble the measure MCP tools for one explicit session context."""

from zcu_tools.mcp.core.bridge import ToolTable, assemble_tools, generate_tools
from zcu_tools.mcp.core.call_log import wrap_handler
from zcu_tools.mcp.measure import (
    tools_cfg,
    tools_context,
    tools_debug,
    tools_device,
    tools_lifecycle,
    tools_notify,
    tools_operation,
    tools_overview,
    tools_screenshot,
    tools_soc,
    tools_tab,
)
from zcu_tools.mcp.measure.exposure import build_mcp_exposure_plan
from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def build_measure_tools(context: MeasureToolContext) -> ToolTable:
    """Return tools bound to context, rejecting invalid exposure declarations.

    Generated and manual handlers share this context's guarded session sender.
    Building another table must not change the routing of previously built tools.
    Handler results and errors retain the existing measure MCP contract.
    """
    overrides: ToolTable = {}
    for module in (
        tools_lifecycle,
        tools_overview,
        tools_cfg,
        tools_context,
        tools_operation,
        tools_device,
        tools_tab,
        tools_soc,
        tools_screenshot,
        tools_debug,
        tools_notify,
    ):
        for name, entry in module.build_override_tools(context).items():
            if name in overrides:
                raise RuntimeError(f"duplicate MCP override tool {name!r}")
            overrides[name] = dict(entry)
    exposure = build_mcp_exposure_plan(context.config, context.method_specs, overrides)
    tools = assemble_tools(
        generate_tools(
            context.config,
            dict(context.method_specs),
            exposure.non_generated_methods,
            context.send_gui_rpc,
        ),
        overrides,
        frozenset(overrides),
    )
    for name, entry in tools.items():
        entry["handler"] = wrap_handler(name, entry["handler"])
    return tools
