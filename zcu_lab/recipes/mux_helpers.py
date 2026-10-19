"""Shared input schemas and reported tone fields for mux qubit recipes."""

from typing import NotRequired, TypedDict

from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    RecipeExpression,
    RecipeInputSchema,
    RecipeParameter,
)


class MuxOverride(TypedDict):
    """One logical scalar edit; integer path segments are full tone positions.

    value=None retains the GUI input. An expr object is evaluated by the owner,
    not this recipe. Plain values are explicit; expressions are derived.
    """

    path: list[str | int]
    value: RecipeParameter | RecipeExpression


class MuxLibrary(TypedDict):
    """Select key at a logical selector path; None retains the GUI selection."""

    path: list[str | int]
    key: str | None


class MuxInline(TypedDict):
    """Copy this selector's captured Spec; list ref ancestors before children."""

    path: list[str | int]


class MuxInputs(TypedDict):
    """Optional keywords shared by the seven tools; absent and None retain defaults.

    target is this Reset's mux alias. reuse_tab_id identifies an idle matching tab.
    participants contains names in execution order. library and inline contain
    ordered selector operations, followed by the atomic scalar overrides batch.
    Integer path segments always identify original complete-list positions.
    """

    target: NotRequired[str | None]
    reuse_tab_id: NotRequired[str | None]
    participants: NotRequired[list[str] | None]
    library: NotRequired[list[MuxLibrary] | None]
    inline: NotRequired[list[MuxInline] | None]
    overrides: NotRequired[list[MuxOverride] | None]


# Keep the seven recipe tools on the same typed input schema.
INPUTS: RecipeInputSchema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "target": {"type": ["string", "null"], "minLength": 1},
        "reuse_tab_id": {"type": ["string", "null"], "minLength": 1},
        "participants": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Names in execution order; omitted/null retains GUI input.",
        },
        "library": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "key"],
                "properties": {
                    "path": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": ["string", "integer"]},
                    },
                    "key": {"type": ["string", "null"], "minLength": 1},
                },
            },
        },
        "inline": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": ["string", "integer"]},
                    },
                },
            },
        },
        "overrides": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "value"],
                "properties": {
                    "path": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": ["string", "integer"]},
                    },
                    "value": {
                        "type": [
                            "string",
                            "integer",
                            "number",
                            "boolean",
                            "null",
                            "array",
                            "object",
                        ],
                        "items": {
                            "type": ["string", "integer", "number", "boolean", "null"]
                        },
                        "properties": {"expr": {"type": "string"}},
                        "required": ["expr"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    },
}


# Report pulse settings at complete-list tone positions, not participant rows.
DRIVE_PARAMETERS = (
    SummaryParameter("drive_ref", ("drive",)),
    SummaryParameter("frequency_mhz", ("drive", "freq"), "MHz"),
    SummaryParameter("gain", ("drive", "gain")),
)


DETUNE_PARAMETERS = (SummaryParameter("detune", ("detune",), "MHz"),)


ECHO_PI_PARAMETERS = (
    SummaryParameter("pi_ref", ("pi_pulse",)),
    SummaryParameter("pi_frequency_mhz", ("pi_pulse", "freq"), "MHz"),
    SummaryParameter("pi_gain", ("pi_pulse", "gain")),
)
