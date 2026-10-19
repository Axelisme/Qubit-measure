"""Measure mux probe-off/on IQ clouds, fit GE separation, and review calibration."""

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


def mux_singleshot_ge(
    session: RecipeSession, **inputs: Unpack[MuxInputs]
) -> RecipeGenerator:
    """Compare simultaneous probe-off/on shots and review each qubit's GE fit.

    session supplies the GUI connection. Optional inputs select the mux target,
    reusable tab, ordered participants, library modules, inline copies and scalar
    overrides. Paths use full tone positions, MHz, us and native gain; None keeps
    the Form input. Copy referenced ancestors before editing their children.
    Raw shots are saved, then Primary fits separation and Post estimates radius
    and confusion. Only an accepted answer applies the current calibration draft;
    skipping writes nothing. Cancellation stops later steps; invalid inputs and
    acquisition or analysis failures propagate without undoing completed edits.
    """
    # 1. Open or reset with this mux group as the source of tone defaults.
    target = inputs.get("target")
    participants = inputs.get("participants")
    tab = session.open_tab(
        "mux/singleshot_ge",
        aliases=None if target is None else {"mux": target},
        reuse=inputs.get("reuse_tab_id"),
    )

    # 2. Select modules, then copy references before editing their scalar leaves.
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

    # 3. Keep every shot in the paired preparations and save the raw IQ first.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()

    # 4. Fit GE separation before estimating radius and confusion from that fit.
    analysis = yield from run.analyze_step("primary")
    if analysis.status == "cancelled":
        return
    analysis = yield from run.analyze_step("post")
    if analysis.status == "cancelled":
        return

    # 5. Present calibration candidates; only explicit acceptance writes them.
    decision = yield from ask(run.propose_writeback())
    if decision.status == "cancelled":
        return
    if decision.decision == "accepted":
        tab.accept()


DEFINITION = RecipeDefinition(
    name="mux_singleshot_ge",
    description="Run mux single-shot GE, save raw and analysis, and ask before Apply.",
    input_schema=INPUTS,
    run=mux_singleshot_ge,
    adapter_name="mux/singleshot_ge",
    summary_parameters=(
        SummaryElements("tones", ("tones",), DRIVE_PARAMETERS, source="default"),
        SummaryParameter("participants", ("participants",)),
        SummaryParameter("reps", ("reps",), None),
        SummaryParameter("rounds", ("rounds",), None),
        SummaryParameter("relax_delay", ("relax_delay",), "us"),
        SummaryParameter("shots", ("shots",), None),
    ),
    summary_estimates=(
        SummaryEstimateRows(
            "tones",
            ("tones",),
            (
                SummaryEstimate(
                    "fidelity", ("fidelity",), None, None, element_key=("element",)
                ),
                SummaryEstimate(
                    "theta", ("theta",), None, None, element_key=("element",)
                ),
                SummaryEstimate(
                    "threshold", ("threshold",), None, None, element_key=("element",)
                ),
                SummaryEstimate(
                    "ge_s", ("ge_s",), None, None, element_key=("element",)
                ),
                SummaryEstimate(
                    "ge_radius", ("ge_radius",), None, None, element_key=("element",)
                ),
            ),
        ),
    ),
)
