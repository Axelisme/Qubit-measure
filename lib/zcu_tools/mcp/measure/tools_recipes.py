"""Bind injected async declarations and request-scoped question answers."""

from collections.abc import Sequence
from functools import partial

from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.measure.execution_reply import project_execution
from zcu_tools.mcp.measure.recipe_api import RecipeDeclaration
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

INITIAL_WAIT_SECONDS = 300.0


def run_recipe(
    tools: MeasureToolContext,
    definition: RecipeDeclaration,
    arguments: dict[str, object],
) -> ToolReply:
    """Validate authored inputs and wait at most 300 seconds for the next handoff.

    Terminal replies contain only status/report/runs. Nonterminal replies expose
    execution and pending question/analysis for further control. Timeout leaves
    work running. Busy/closed admission fails before GUI work.
    """
    execution = tools.session.recipes.start(tools, definition.name, arguments)
    reply = execution.wait(INITIAL_WAIT_SECONDS)
    return ToolReply(
        project_execution(reply.data),
        reply.images,
        reply.is_error,
    )


def answer(tools: MeasureToolContext, arguments: dict[str, object]) -> ToolReply:
    """Answer exactly execution/request_id/answer without granting write permission.

    request_id must identify the current question; stale/absent requests return
    accepted=False, reason and pending_request_id without consuming a question.
    Invalid named answer fields raise and preserve the pending request. Accepted
    answers wait up to 300 seconds for continuation; only authored writes apply.
    """
    if set(arguments) != {"execution", "request_id", "answer"}:
        raise ValueError("Answers require exactly execution, request_id and answer")
    key = arguments["execution"]
    if not isinstance(key, str) or not key:
        raise ValueError("execution must be a non-empty execution ID")
    reply = tools.session.recipes.get(key).answer(
        {"request_id": arguments["request_id"], "answer": arguments["answer"]}
    )
    if reply.data.get("accepted") is False:
        return reply
    return ToolReply(
        project_execution(reply.data),
        reply.images,
        reply.is_error,
    )


def build_recipe_tools(
    context: MeasureToolContext,
    *,
    recipes: Sequence[RecipeDeclaration],
) -> ToolTable:
    """Generate input schemas from admitted models and bind the fixed answer tool.

    recipes must match the session declarations. Assembly rejects tool-name
    collisions; schemas and invocation validation share each inputs model.
    """
    if any(definition.name == "answer" for definition in recipes):
        raise RuntimeError("duplicate MCP tool 'answer'")
    return {
        "answer": {
            "handler": partial(answer, context),
            "description": "Answer the current question using its request_id and named answer fields from pending_question.answer_schema. A stale answer never consumes a question or writes. Only the recipe's explicit accept_writeback writes. Wait up to 300 seconds for the next handoff; timeout never cancels.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "execution": {"type": "string", "minLength": 1},
                    "request_id": {"type": "string", "minLength": 1},
                    "answer": {"type": "object"},
                },
                "required": ["execution", "request_id", "answer"],
                "additionalProperties": False,
            },
        },
        **{
            definition.name: {
                "handler": partial(run_recipe, context, definition),
                "description": definition.description
                + " Returns status, authored report and runs. Explicit questions require answer(execution, request_id, answer). Interactive analysis hands off pending(tab,stage,op). Use status(execution, detail=full) for captured owner evidence; waiting never cancels.",
                "inputSchema": definition.inputs.model_json_schema(),
            }
            for definition in recipes
        },
    }
