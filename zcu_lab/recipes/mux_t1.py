"""One bounded mux recipe; numeric acquisition remains with its native owner."""

from typing import Unpack

from zcu_tools.mcp.measure.execution_reply import (
    SummaryElements,
    SummaryEstimate,
    SummaryEstimateRows,
    SummaryParameter,
)
from zcu_tools.mcp.measure.recipe import (
    RecipeDefinition,
    RecipeGenerator,
    RecipeSession,
)

from .mux_helpers import (
    DRIVE_PARAMETERS,
    INPUTS,
    MuxInputs,
    finish_mux,
    prepare_mux,
)


def mux_t1(session: RecipeSession, **inputs: Unpack[MuxInputs]) -> RecipeGenerator:
    """Run mux T1; review candidates before explicit Apply.

    session is driver-supplied. target binds the Reset mux alias; None retains
    global aliases. reuse_tab_id resets an idle mux/t1 tab.
    participants is an ordered name list; None keeps Reset defaults. library
    selects typed path/key pairs, then inline copies typed selector paths, then
    overrides sends one batch of typed scalar path/value edits. Caller order is
    preserved; list indexes are full original positions, never participant rows.
    Copy ref ancestors outer-first before editing children. None values retain
    input; expressions remain authored. No range/calibration inference occurs.
    Paths use MHz/us/native gain, matching the Form; count leaves use n.
    Shared reps/rounds/recovery and delay axes remain shared, not per-tone.
    All elements, including nonparticipants, must validate before Run.
    Defaults report retained Form inputs, not their internal generation history.
    Run saves raw and Primary; cancellation stops later steps. Finish-early
    follows owner partial-data policy. Only an accepted writeback answer applies;
    skipped writes none. Invalid inputs/cfg and native/transport failures propagate
    at their author step; successful copies are not rolled back on later failure.
    """
    tab = prepare_mux(
        session,
        "mux/t1",
        inputs,
    )
    yield from finish_mux(tab)


DEFINITION = RecipeDefinition(
    name="mux_t1",
    description="Run mux T1, save raw and analysis, and ask before Apply.",
    input_schema=INPUTS,
    run=mux_t1,
    adapter_name="mux/t1",
    summary_parameters=(
        SummaryElements("tones", ("tones",), DRIVE_PARAMETERS, source="default"),
        SummaryParameter("participants", ("participants",)),
        SummaryParameter("reps", ("reps",), None),
        SummaryParameter("rounds", ("rounds",), None),
        SummaryParameter("relax_delay", ("relax_delay",), "us"),
        SummaryParameter("delay_start", ("delays", "start"), "us"),
        SummaryParameter("delay_stop", ("delays", "stop"), "us"),
        SummaryParameter("delay_n", ("delays", "n"), None),
    ),
    summary_estimates=(
        SummaryEstimateRows(
            "tones",
            ("tones",),
            (
                SummaryEstimate(
                    "t1", ("t1",), ("t1_err",), "us", element_key=("element",)
                ),
                SummaryEstimate(
                    "t1b", ("t1b",), ("t1b_err",), "us", element_key=("element",)
                ),
            ),
        ),
    ),
)
