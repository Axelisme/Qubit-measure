"""Bind injected generator recipes and their captured-question answers."""

from collections.abc import Sequence
from functools import partial

from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.measure.execution_reply import project_execution
from zcu_tools.mcp.measure.recipe import RecipeDefinition
from zcu_tools.mcp.measure.recipe_api import RecipeDeclaration
from zcu_tools.mcp.measure.script_recipe_execution import ScriptRecipeExecution
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

INITIAL_WAIT_SECONDS = 300.0


def run_recipe(
    tools: MeasureToolContext,
    definition: RecipeDefinition | RecipeDeclaration,
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

    Legacy recipes take recipe=<execution ID> and decision=accepted/skipped.
    Script declarations take execution=<ID> and answer=<named Answer fields>.
    The two routes cannot be mixed. Invalid fields preserve the pending question.
    Unknown IDs, repeated answers and non-question states fail without GUI writes.
    Wait up to 300 seconds for the recipe's continuation; only its explicit
    tab.accept applies anything. Return the same captured summary as done/wait.
    """
    if "execution" in arguments or "answer" in arguments:
        if set(arguments) != {"execution", "answer"}:
            raise ValueError("Script answers require exactly execution and answer")
        key, fields = arguments["execution"], arguments["answer"]
        if not isinstance(key, str) or not key:
            raise ValueError("execution must be a non-empty execution ID")
        if not isinstance(fields, dict) or not all(
            isinstance(key, str) for key in fields
        ):
            raise ValueError("answer must be an object with string field names")
        execution = tools.session.recipes.get(key)
        if not isinstance(execution, ScriptRecipeExecution):
            raise ValueError("execution/answer requires a script recipe")
        reply = execution.answer(fields)
        return ToolReply(
            project_execution(reply.data, recipes=tools.session.recipes.definitions),
            is_error=reply.is_error,
        )
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
    if isinstance(execution, ScriptRecipeExecution):
        raise ValueError("Script recipes require execution and answer")
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
    context: MeasureToolContext,
    *,
    recipes: Sequence[RecipeDefinition | RecipeDeclaration],
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
            "description": "Answer a script question with execution and named answer fields, "
            "following status.pending_question.answer_schema, or answer a legacy "
            "writeback question with recipe and decision. Never mix routes. "
            "decision is accepted or skipped, not permission or proof of a write. "
            "Only the recipe's explicit tab.accept writes the current draft. "
            "Wait up to 300 seconds for its next summary; timeout does not cancel.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "recipe": {"type": "string", "minLength": 1},
                    "decision": {"type": "string", "enum": ["accepted", "skipped"]},
                    "execution": {"type": "string", "minLength": 1},
                    "answer": {"type": "object"},
                },
                "oneOf": [
                    {
                        "required": ["recipe", "decision"],
                        "not": {
                            "anyOf": [
                                {"required": ["execution"]},
                                {"required": ["answer"]},
                            ]
                        },
                    },
                    {
                        "required": ["execution", "answer"],
                        "not": {
                            "anyOf": [
                                {"required": ["recipe"]},
                                {"required": ["decision"]},
                            ]
                        },
                    },
                ],
                "additionalProperties": False,
            },
        },
        **{
            definition.name: {
                "handler": partial(run_recipe, context, definition),
                "description": definition.description
                + (
                    " Returns a host-owned outcome with authored report and acquisition history. "
                    "Explicit questions pause until answer(execution, answer); waiting never cancels."
                    if isinstance(definition, RecipeDeclaration)
                    else " Execution summaries retain whole-artifact Save reports, "
                    "including JSON-only analysis and every output status/error. "
                    "Current replies may include in-memory MCP images, not session PNG files. "
                    "Use status(execution, detail=full) for captured native detail."
                ),
                "inputSchema": (
                    definition.inputs.model_json_schema()
                    if isinstance(definition, RecipeDeclaration)
                    else definition.input_schema
                ),
            }
            for definition in recipes
        },
    }
