"""Bind explicitly registered Python recipes to the measure MCP tool table."""

from functools import partial
from typing import Any

from recipes import RECIPES, RecipeDefinition
from recipes.cfg_sources import finite_number
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.measure.execution_reply import project_execution
from zcu_tools.mcp.measure.recipe_context import RecipeContext
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

INITIAL_WAIT_SECONDS = 300.0


def _run_normalized_recipe(
    definition: RecipeDefinition, context: RecipeContext, arguments: dict[str, Any]
) -> None:
    normalized = arguments.copy()
    for name, schema in definition.input_schema["properties"].items():
        value = arguments.get(name)
        if "number" in schema["type"] and finite_number(value):
            normalized[name] = float(value)
        elif (
            "array" in schema["type"]
            and isinstance(value, list)
            and "number" in schema["items"]["type"]
        ):
            normalized[name] = [
                float(endpoint) if finite_number(endpoint) else endpoint
                for endpoint in value
            ]
    definition.run(context, normalized)


def run_recipe(
    tools: MeasureToolContext, definition: RecipeDefinition, arguments: dict[str, Any]
) -> ToolReply:
    execution = tools.session.recipes.start(
        tools, definition.name, partial(_run_normalized_recipe, definition), arguments
    )
    reply = execution.wait(INITIAL_WAIT_SECONDS)
    return ToolReply(
        {
            **project_execution(reply.data, definition=definition),
            "elapsed_s": reply.data["elapsed_s"],
        },
        reply.images,
        reply.is_error,
    )


def build_recipe_tools(context: MeasureToolContext) -> ToolTable:
    return {
        definition.name: {
            "handler": partial(run_recipe, context, definition),
            "description": definition.description
            + " Execution summaries include previews.run/primary/post as full "
            "session-only PNG path lists, separate from persistent artifacts. "
            "Use status(execution, detail=full) for captured native detail.",
            "inputSchema": definition.input_schema,
        }
        for definition in RECIPES
    }
