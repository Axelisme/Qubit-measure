"""Measure mux Hahn Echo, save the results and review coherence writeback."""

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
    ECHO_PI_PARAMETERS,
    INPUTS,
    MuxInputs,
)


def mux_t2echo(session: RecipeSession, **inputs: Unpack[MuxInputs]) -> RecipeGenerator:
    """Run x90 → half wait → x180 → half wait → x90 for selected mux qubits.

    session supplies the GUI connection. target binds the mux group;
    reuse_tab_id resets an idle Echo tab. participants gives ordered qubit names.
    library selects path/key pairs, inline copies references (outermost first),
    then overrides edits scalar paths in MHz/us/native gain. Integer paths use
    full tone positions, not participant rows. None retains the current input.
    The delay axis, reps, rounds and recovery are shared across all tones.
    Save raw IQ and Primary Echo fits, then ask before applying T2E writeback.
    Cancellation stops later steps; a skipped answer writes nothing.
    """
    # 1. Reset the requested tab with this call's mux group and captured sources.
    target = inputs.get("target")
    participants = inputs.get("participants")
    tab = session.open_tab(
        "mux/t2echo",
        aliases=None if target is None else {"mux": target},
        reuse=inputs.get("reuse_tab_id"),
    )

    # 2. Select calibrated pulses before copying references for local edits.
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

    # 3. Acquire and save raw IQ before fitting the Echo decay or fringes.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()
    analysis = yield from run.analyze_step("primary")
    if analysis.status == "cancelled":
        return

    # 4. Let the user review T2E candidates; only acceptance applies the draft.
    decision = yield from ask(run.propose_writeback())
    if decision.status == "cancelled":
        return
    if decision.decision == "accepted":
        tab.accept()


DEFINITION = RecipeDefinition(
    name="mux_t2echo",
    description="Run mux Echo, save raw and analysis, and ask before Apply.",
    input_schema=INPUTS,
    run=mux_t2echo,
    adapter_name="mux/t2echo",
    summary_parameters=(
        SummaryElements(
            "tones",
            ("tones",),
            DRIVE_PARAMETERS + DETUNE_PARAMETERS + ECHO_PI_PARAMETERS,
            source="default",
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
