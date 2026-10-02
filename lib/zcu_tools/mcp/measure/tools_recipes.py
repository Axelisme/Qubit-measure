"""Bind explicitly registered Python recipes to the measure MCP tool table."""

import logging
from functools import partial
from typing import Any

from recipes import RECIPES, RecipeDefinition
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.measure.recipe_context import RecipeContext, RecipeError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

logger = logging.getLogger(__name__)


def run_recipe(
    tools: MeasureToolContext, definition: RecipeDefinition, arguments: dict[str, Any]
) -> ToolReply:
    context = RecipeContext(tools, definition.name)
    try:
        definition.run(context, arguments)
    except Exception as error:
        logger.exception("Recipe %s failed", definition.name)
        context.progress.error = RecipeError(
            context.progress.phase,
            str(getattr(error, "reason", None) or "recipe_failed"),
            str(error),
            getattr(error, "code", None),
        )
        context.progress.status = "failed"
        context.progress.phase = "terminal"
    return ToolReply(context.snapshot(), is_error=context.progress.status == "failed")


def build_recipe_tools(context: MeasureToolContext) -> ToolTable:
    return {
        definition.name: {
            "handler": partial(run_recipe, context, definition),
            "description": definition.description,
            "inputSchema": definition.input_schema,
        }
        for definition in RECIPES
    }
