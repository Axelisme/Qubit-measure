"""Bind injected generator recipes and their captured-question answers."""

from collections.abc import Sequence
from functools import partial

from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.measure.execution_reply import project_execution
from zcu_tools.mcp.measure.recipe import RecipeDefinition
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

INITIAL_WAIT_SECONDS = 300.0


def run_recipe(
    tools: MeasureToolContext,
    definition: RecipeDefinition,
    arguments: dict[str, object],
) -> ToolReply:
    """Start the injected tool name with schema-validated keyword arguments.

    Invalid arguments or busy/closed admission fail before GUI work. Wait at most
    300 seconds for the next handoff; timeout leaves the generator running.
    Return captured summary/images and preserve any delivery or execution failure.
    """
    execution = tools.session.recipes.start(tools, definition.name, arguments)
    reply = execution.wait(INITIAL_WAIT_SECONDS)
    return ToolReply(
        {
            **project_execution(
                reply.data,
                definition=definition,
                recipes=tools.session.recipes.definitions,
            ),
            "elapsed_s": reply.data["elapsed_s"],
        },
        reply.images,
        reply.is_error,
    )


def answer(tools: MeasureToolContext, arguments: dict[str, object]) -> ToolReply:
    """Answer one session-local recipe question without granting write permission.

    arguments requires only recipe=<execution ID> and decision=accepted/skipped.
    Unknown IDs, repeated answers and non-question states fail without GUI writes.
    Wait up to 300 seconds for the recipe's continuation; only its explicit
    tab.accept applies anything. Return the same captured summary as done/wait.
    """
    unknown = set(arguments) - {"recipe", "decision"}
    if unknown:
        raise ValueError(f"Unexpected answer fields: {', '.join(sorted(unknown))}")
    recipe = arguments.get("recipe")
    decision = arguments.get("decision")
    if not isinstance(recipe, str) or not recipe:
        raise ValueError("recipe must be a non-empty execution ID")
    if decision not in ("accepted", "skipped"):
        raise ValueError("decision must be accepted or skipped")
    execution = tools.session.recipes.get(recipe)
    # Explicit branches narrow the dynamic JSON value to the declared decision.
    reply = execution.answer("accepted" if decision == "accepted" else "skipped")
    definition = next(
        item
        for item in tools.session.recipes.definitions
        if item.name == reply.data["recipe"]
    )
    return ToolReply(
        {
            **project_execution(
                reply.data,
                definition=definition,
                recipes=tools.session.recipes.definitions,
            ),
            "elapsed_s": reply.data["elapsed_s"],
        },
        reply.images,
        reply.is_error,
    )


def build_recipe_tools(
    context: MeasureToolContext, *, recipes: Sequence[RecipeDefinition]
) -> ToolTable:
    """Bind already validated session definitions plus the fixed answer tool.

    recipes is the same injected sequence admitted by build_measure_tools. Recipe
    names cannot collide with answer; assembly rejects every duplicate tool name.
    """
    if any(definition.name == "answer" for definition in recipes):
        raise RuntimeError("duplicate MCP tool 'answer'")
    return {
        "answer": {
            "handler": partial(answer, context),
            "description": "Answer a captured recipe writeback question by execution ID. "
            "decision is accepted or skipped, not permission or proof of a write. "
            "Only the recipe's explicit tab.accept writes the current draft. "
            "Wait up to 300 seconds for its next summary; timeout does not cancel.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "recipe": {"type": "string", "minLength": 1},
                    "decision": {"type": "string", "enum": ["accepted", "skipped"]},
                },
                "required": ["recipe", "decision"],
                "additionalProperties": False,
            },
        },
        **{
            definition.name: {
                "handler": partial(run_recipe, context, definition),
                "description": definition.description
                + " Execution summaries include previews.run/primary/post as full "
                "session-only PNG path lists, separate from persistent artifacts. "
                "Use status(execution, detail=full) for captured native detail.",
                "inputSchema": definition.input_schema,
            }
            for definition in recipes
        },
    }
