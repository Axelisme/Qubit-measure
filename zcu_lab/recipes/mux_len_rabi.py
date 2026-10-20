"""Measure mux length Rabi, fit pulse lengths, and review calibration updates."""

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


def mux_len_rabi(
    session: RecipeSession, **inputs: Unpack[MuxInputs]
) -> RecipeGenerator:
    """Sweep simultaneous qubit pulse lengths and fit pi/pi2 calibrations.

    session is driver-supplied. target binds the Reset mux alias; None retains
    global aliases. reuse_tab_id resets an idle mux/len_rabi tab.
    participants is an ordered name list; None keeps Reset defaults. library
    selects typed path/key pairs, then inline copies typed selector paths, then
    overrides sends one batch of typed scalar path/value edits. Caller order is
    preserved; list indexes are full original positions, never participant rows.
    Copy ref ancestors outer-first before editing children. None values retain
    input; expressions remain authored. No range/calibration inference occurs.
    Paths use MHz/us/native gain, matching the Form; count leaves use n.
    Shared reps/rounds/recovery and delay axes remain shared, not per-tone.
    All elements require equal sweep counts; Len Rabi supports const/flat_top only.
    Run saves raw data and Primary fits, then asks whether to apply the proposed
    calibrations. Accept writes all current draft items; skip writes nothing.
    Cancellation stops later steps; invalid inputs and native/transport failures
    propagate. Completed input edits are not rolled back after a later failure.
    """
    # 1. Open or reset the tab with the requested mux binding.
    target = inputs.get("target")
    participants = inputs.get("participants")
    tab = session.open_tab(
        "mux/len_rabi",
        aliases=None if target is None else {"mux": target},
        reuse=inputs.get("reuse_tab_id"),
    )

    # 2. Select sources before copying references and editing their scalar leaves.
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

    # 3. Acquire and save raw IQ, then fit and save Primary Rabi results.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()
    analysis = yield from run.analyze_step("primary")
    if analysis.status == "cancelled":
        return

    # 4. Ask before applying any calibration; skipping leaves the library unchanged.
    decision = yield from ask(run.propose_writeback())
    if decision.status == "cancelled":
        return
    if decision.decision == "accepted":
        tab.accept()


DEFINITION = RecipeDefinition(
    name="mux_len_rabi",
    description="Run mux length Rabi, save raw and analysis, and ask before Apply.",
    input_schema=INPUTS,
    run=mux_len_rabi,
    adapter_name="mux/len_rabi",
    summary_parameters=(
        SummaryElements(
            "tones",
            ("tones",),
            DRIVE_PARAMETERS
            + (
                SummaryParameter("start", ("sweep", "start"), "us"),
                SummaryParameter("stop", ("sweep", "stop"), "us"),
                SummaryParameter("points", ("sweep", "n")),
            ),
            source="default",
        ),
        SummaryParameter("participants", ("participants",)),
        SummaryParameter("reps", ("reps",), None),
        SummaryParameter("rounds", ("rounds",), None),
        SummaryParameter("relax_delay", ("relax_delay",), "us"),
    ),
    summary_estimates=(
        SummaryEstimateRows(
            "tones",
            ("tones",),
            (
                SummaryEstimate(
                    "pi_len",
                    ("pi_len",),
                    ("pi_len_err",),
                    "us",
                    element_key=("element",),
                ),
                SummaryEstimate(
                    "pi2_len",
                    ("pi2_len",),
                    ("pi2_len_err",),
                    "us",
                    element_key=("element",),
                ),
                SummaryEstimate(
                    "rabi_f",
                    ("rabi_f",),
                    ("rabi_f_err",),
                    "MHz",
                    element_key=("element",),
                ),
            ),
        ),
    ),
)
