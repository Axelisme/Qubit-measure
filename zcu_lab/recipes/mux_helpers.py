"""Shared author operations for the seven mux qubit recipes, not a cfg engine."""

from typing import TypedDict

from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    RecipeExpression,
    RecipeFieldEdit,
    RecipeGenerator,
    RecipeInputSchema,
    RecipeParameter,
    RecipeSession,
    RecipeTab,
    ask,
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
                            "number",
                            "boolean",
                            "null",
                            "array",
                            "object",
                        ],
                        "items": {"type": ["string", "number", "boolean", "null"]},
                        "properties": {"expr": {"type": "string"}},
                        "required": ["expr"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    },
}


def prepare_mux(  # noqa: PLR0913 - All seven authors share these typed tool inputs.
    session: RecipeSession,
    adapter: str,
    *,
    target: str | None,
    reuse_tab_id: str | None,
    participants: list[str] | None,
    library: list[MuxLibrary] | None,
    inline: list[MuxInline] | None,
    overrides: list[MuxOverride] | None,
) -> RecipeTab:
    """Reset the mux tab, then select library, copy inline, and batch scalar edits.

    target overrides only this Reset's mux alias; None uses global aliases.
    reuse_tab_id names an idle tab of adapter. Caller order within library and
    inline is preserved. Ref ancestors must be copied explicitly, outer first.
    A completed selection/copy is not rolled back when a later operation fails.
    participants=None retains the Reset defaults. None override values retain
    input. Expressions stay authored and are labelled derived. All actual path,
    reference, fixed-list and cfg validation belongs to the public Form owner.
    No name lookup, range inference, calibration fallback or device access occurs.
    Owner/admission/connection failures propagate without retry or refresh.
    """
    tab = session.open_tab(
        adapter,
        aliases=None if target is None else {"mux": target},
        reuse=reuse_tab_id,
    )
    for selection in library or ():
        tab.use_library(tuple(selection["path"]), selection["key"])
    for copy in inline or ():
        tab.use_inline(tuple(copy["path"]))
    edits = [
        RecipeFieldEdit(
            ("participants",),
            None if participants is None else [name for name in participants],
        )
    ]
    edits.extend(
        RecipeFieldEdit(
            tuple(edit["path"]),
            edit["value"],
            source="derived" if isinstance(edit["value"], dict) else "explicit",
        )
        for edit in overrides or ()
    )
    tab.set_fields(edits)
    return tab


def finish_mux(tab: RecipeTab, *, post: bool = False) -> RecipeGenerator:
    """Run once, save raw, analyze Primary (and GE Post), then ask before Apply.

    Cancellation returns without starting later steps. Finish-early follows the
    owner's partial-data/save policy. Only an explicit accepted answer applies
    captured candidates; skipped writes none. Native or transport failures
    propagate at their author step. This helper never manufactures estimates.
    """
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()
    analysis = yield from run.analyze_step("primary")
    if analysis.status == "cancelled":
        return
    if post:
        analysis = yield from run.analyze_step("post")
        if analysis.status == "cancelled":
            return
    decision = yield from ask(run.propose_writeback())
    if decision.status == "cancelled":
        return
    if decision.decision == "accepted":
        tab.accept()


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
