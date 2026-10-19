"""Measure mux Ramsey fringes, save the decay fits and review calibrations."""

from typing import Unpack

from zcu_tools.mcp.measure.execution_reply import (
    SummaryElements,
    SummaryEstimate,
    SummaryEstimateRows,
    SummaryParameter,
)
from zcu_tools.mcp.measure.recipe import (
    RecipeDefinition,
    RecipeFieldEdit,
    RecipeGenerator,
    RecipeSession,
    ask,
)

from .mux_helpers import (
    DETUNE_PARAMETERS,
    DRIVE_PARAMETERS,
    INPUTS,
    MuxInputs,
)


def mux_t2ramsey(
    session: RecipeSession, **inputs: Unpack[MuxInputs]
) -> RecipeGenerator:
    """Measure mux T2*, save raw data and fits, then ask before writeback.

    session supplies the GUI tab. target selects the mux group; reuse_tab_id
    resets an idle Ramsey tab. participants gives ordered qubit names, while
    library, inline and overrides select pulse sources and edit Form paths in
    that order. Paths use original tone positions, MHz, us and native gain;
    None retains the Form value. Copy library ancestors inline before editing
    their children. Shared delay/reps/rounds apply to every participating qubit.
    All tones must validate, including nonparticipants. Cancellation stops later
    steps; only an accepted answer applies the proposed calibrations.
    """
    # 1. Open or reset the Ramsey tab with the requested mux group.
    target = inputs.get("target")
    participants = inputs.get("participants")
    tab = session.open_tab(
        "mux/t2ramsey",
        aliases=None if target is None else {"mux": target},
        reuse=inputs.get("reuse_tab_id"),
    )

    # 2. Select pulse sources before editing their inline values.
    for selection in inputs.get("library") or ():
        tab.use_library(tuple(selection["path"]), selection["key"])
    for copy in inputs.get("inline") or ():
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
        for edit in inputs.get("overrides") or ()
    )
    tab.set_fields(edits)

    # 3. Measure and save raw IQ before fitting the Ramsey fringes.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()
    analysis = yield from run.analyze_step("primary")
    if analysis.status == "cancelled":
        return

    # 4. Present the proposed calibrations; skipping leaves the library unchanged.
    decision = yield from ask(run.propose_writeback())
    if decision.status == "cancelled":
        return
    if decision.decision == "accepted":
        tab.accept()


DEFINITION = RecipeDefinition(
    name="mux_t2ramsey",
    description="Run mux Ramsey, save raw and analysis, and ask before Apply.",
    input_schema=INPUTS,
    run=mux_t2ramsey,
    adapter_name="mux/t2ramsey",
    summary_parameters=(
        SummaryElements(
            "tones", ("tones",), DRIVE_PARAMETERS + DETUNE_PARAMETERS, source="default"
        ),
        SummaryParameter("participants", ("participants",)),
        SummaryParameter("reps", ("reps",), None),
        SummaryParameter("rounds", ("rounds",), None),
        SummaryParameter("relax_delay", ("relax_delay",), "us"),
        SummaryParameter("delay_start", ("delay", "start"), "us"),
        SummaryParameter("delay_stop", ("delay", "stop"), "us"),
        SummaryParameter("delay_n", ("delay", "n"), None),
    ),
    summary_estimates=(
        SummaryEstimateRows(
            "tones",
            ("tones",),
            (
                SummaryEstimate(
                    "t2", ("t2",), ("t2_err",), "us", element_key=("element",)
                ),
                SummaryEstimate(
                    "detune",
                    ("detune",),
                    ("detune_err",),
                    "MHz",
                    element_key=("element",),
                ),
            ),
        ),
    ),
)
