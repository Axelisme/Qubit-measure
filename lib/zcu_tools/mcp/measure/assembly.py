"""Assemble only the fixed measure tools owned by delivered tickets."""

from collections.abc import Sequence

from zcu_tools.mcp.core.call_log import wrap_handler
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.measure import (
    tools_lifecycle,
    tools_operation,
    tools_recipes,
    tools_rpc,
    tools_run_analyze,
    tools_setup,
    tools_tab,
    tools_writeback,
)
from zcu_tools.mcp.measure.recipe import RecipeDefinition
from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def build_measure_tools(
    context: MeasureToolContext, *, recipes: Sequence[RecipeDefinition]
) -> ToolTable:
    """Bind fixed tools and explicitly injected recipes, never catalog-generated tools.

    recipes must match the owning session declarations, including schemas and
    callbacks. A mismatch raises ValueError before GUI access. Duplicate tool names
    raise RuntimeError. Returned handlers retain this context for their lifetime.
    """
    if tuple(recipes) != context.session.recipes.definitions:
        raise ValueError("Tool recipes must match the owning session registry")
    tools: ToolTable = {}
    for source in (
        tools_lifecycle.build_override_tools(context),
        tools_operation.build_operation_tools(context),
        tools_rpc.build_rpc_tools(context),
        tools_tab.build_tab_read_tools(context),
        tools_run_analyze.build_run_analyze_tools(context),
        tools_writeback.build_writeback_tools(context),
        tools_recipes.build_recipe_tools(context, recipes=recipes),
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
