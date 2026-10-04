"""Explicit setup workflows over one captured GUI connection.

Mutations are sent once. Timeout never cancels them; uncertain receipts and
partial snapshots stay visible. They never launch a GUI or reconnect between steps.
"""

from functools import partial
from typing import Any, Literal, NotRequired, TypedDict

from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.measure.tool_context import MeasureToolContext


class SetupError(TypedDict, total=False):
    code: str | None
    reason: str | None
    message: str
    request_rejected: bool


class SetupStep(TypedDict):
    status: Literal["not_started", "completed", "running", "failed", "unknown"]
    error: NotRequired[SetupError]


class SetupResult(TypedDict):
    tool: str
    op: int | None
    status: str
    steps: dict[str, SetupStep]
    before: dict[str, Any]
    after: dict[str, Any]
    requested: dict[str, Any]
    operation: dict[str, Any] | None
    verification: dict[str, Any] | None
    error: SetupError | None


def simulation_initialize(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> ToolReply:
    """Initialize once, wait, then verify mock SoC, fake_flux and predictor facts."""
    raise NotImplementedError


def device_set_value(ctx: MeasureToolContext, arguments: dict[str, Any]) -> ToolReply:
    """Set only value in the declared native unit and retain actual snapshots."""
    raise NotImplementedError


def recipe_guide(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Query the registry-declared adapter guide without interpreting recipe names."""
    raise NotImplementedError


def build_setup_tools(ctx: MeasureToolContext) -> ToolTable:
    timeout_schema = {"type": "number", "minimum": 0, "maximum": 300, "default": 60.0}
    return {
        "simulation_initialize": {
            "handler": partial(simulation_initialize, ctx),
            "description": "Initialize the simulated environment through the GUI coordinator. "
            "May disconnect real devices; requires caller authorization. Reuses valid simulation. "
            "Does not apply a project, launch a GUI or connect real hardware. "
            "Wait timeout does not cancel. Inspect steps, operation and snapshots before retrying.",
            "inputSchema": {
                "type": "object",
                "properties": {"timeout": timeout_schema},
                "additionalProperties": False,
            },
        },
        "device_set_value": {
            "handler": partial(device_set_value, ctx),
            "description": "Set only a connected device working value. Unit must match its GUI "
            "snapshot; FakeDevice unit=none requires explicit native. No unit conversion, "
            "output/mode/rampstep change, retry or rollback. Reports requested and actual values "
            "without hardware tolerance. Wait timeout does not cancel; inspect partial progress.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "value": {"type": "number"},
                    "unit": {"type": "string", "minLength": 1},
                    "timeout": timeout_schema,
                },
                "required": ["name", "value", "unit"],
                "additionalProperties": False,
            },
        },
        "recipe_guide": {
            "handler": partial(recipe_guide, ctx),
            "description": "Read the native guide of a registered recipe adapter. "
            "Uses the authoritative recipe registry; does not run a recipe or mutate the GUI.",
            "inputSchema": {
                "type": "object",
                "properties": {"recipe": {"type": "string", "minLength": 1}},
                "required": ["recipe"],
                "additionalProperties": False,
            },
        },
    }
