"""Bind explicitly registered Python recipes to the measure MCP tool table."""

from functools import partial
from typing import Any

from recipes import RECIPES, RecipeDefinition
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

INITIAL_WAIT_SECONDS = 300.0


def run_recipe(
    tools: MeasureToolContext, definition: RecipeDefinition, arguments: dict[str, Any]
) -> ToolReply:
    execution = tools.session.recipes.start(
        tools, definition.name, definition.run, arguments
    )
    return execution.wait(INITIAL_WAIT_SECONDS)


def build_recipe_tools(context: MeasureToolContext) -> ToolTable:
    return {
        definition.name: {
            "handler": partial(run_recipe, context, definition),
            "description": definition.description,
            "inputSchema": definition.input_schema,
        }
        for definition in RECIPES
    }
