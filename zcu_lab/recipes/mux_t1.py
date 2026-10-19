"""Measure simultaneous T1 decays and ask before writing calibrations."""

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
    DRIVE_PARAMETERS,
    INPUTS,
    MuxInputs,
)


def mux_t1(session: RecipeSession, **inputs: Unpack[MuxInputs]) -> RecipeGenerator:
    """Excite selected qubits, scan a shared wait, and fit their T1 decays.

    session supplies the measurement tab. target selects the mux group;
    reuse_tab_id resets an idle mux/t1 tab instead of opening a new one.
    participants lists qubit names in measurement order. library selects pulse
    or readout entries, inline copies selections for editing, and overrides sets
    scalar fields in MHz, us or native gain. Integer paths address the complete
    tone list, not participant rows; copy parent references before children.
    Omitted/None values retain Form inputs, and expressions stay expressions.

    Save raw IQ and Primary fits, then ask before applying T1 calibrations to
    the captured qubits. Skipping writes nothing. Cancellation stops later
    steps; invalid settings and acquisition/save failures propagate.
    """
    # 1. Open or reset the tab with this mux group's pulse/readout defaults.
    target = inputs.get("target")
    participants = inputs.get("participants")
    tab = session.open_tab(
        "mux/t1",
        aliases=None if target is None else {"mux": target},
        reuse=inputs.get("reuse_tab_id"),
    )

    # 2. Select library entries before copying and editing their fields.
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

    # 3. Run pi pulses -> shared delay -> mux readout, then save raw IQ.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()

    # 4. Fit each qubit's decay on the requested delay axis and save Primary.
    analysis = yield from run.analyze_step("primary")
    if analysis.status == "cancelled":
        return

    # 5. Apply the current calibration draft only after an accepted answer.
    decision = yield from ask(run.propose_writeback())
    if decision.status == "cancelled":
        return
    if decision.decision == "accepted":
        tab.accept()


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
