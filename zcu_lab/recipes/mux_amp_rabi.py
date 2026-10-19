"""Measure mux Rabi oscillations versus gain and review pi/pi2 pulse updates."""

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


def mux_amp_rabi(
    session: RecipeSession, **inputs: Unpack[MuxInputs]
) -> RecipeGenerator:
    """Measure Rabi oscillations, fit pi/pi2 gains and ask before writing updates.

    session opens a mux/amp_rabi tab; target selects its mux group, or None keeps
    global aliases. reuse_tab_id resets an idle tab. participants gives the ordered
    qubit names, or None keeps the Form defaults. library selects path/key pairs;
    inline copies selected modules before overrides applies scalar path/value edits.
    Paths index the complete tone list, not participant rows. Copy referenced
    ancestors outer-first. None values retain inputs; expressions remain authored.
    Frequencies use MHz, times us and gains normalized units; sweep counts use n.
    No scan range is inferred. All tones must validate, including nonparticipants.
    Raw data and Primary analysis are saved before the writeback question.
    Only an accepted answer applies the current pi/pi2 pulse, slot and pi_product
    proposals; skipping writes nothing. Cancellation stops later steps, while
    finish-early retains partial data. Invalid inputs and operation errors propagate.
    """
    # Capture this group's Form defaults without changing the global aliases.
    target = inputs.get("target")
    participants = inputs.get("participants")
    tab = session.open_tab(
        "mux/amp_rabi",
        aliases=None if target is None else {"mux": target},
        reuse=inputs.get("reuse_tab_id"),
    )

    # Select sources before copying referenced modules for scalar edits.
    for selection in inputs.get("library") or ():
        tab.use_library(tuple(selection["path"]), selection["key"])
    for copy in inputs.get("inline") or ():
        tab.use_inline(tuple(copy["path"]))

    # Apply participants and overrides together, retaining omitted Form values.
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

    # Acquire and save raw IQ before fitting the gain oscillations.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()

    # Primary analysis saves the fit and proposes pi/pi2 pulse calibrations.
    analysis = yield from run.analyze_step("primary")
    if analysis.status == "cancelled":
        return

    # Preview never writes: apply only after an explicit accepted answer.
    decision = yield from ask(run.propose_writeback())
    if decision.status == "cancelled":
        return
    if decision.decision == "accepted":
        tab.accept()


DEFINITION = RecipeDefinition(
    name="mux_amp_rabi",
    description="Run mux amplitude Rabi, save raw and analysis, and ask before Apply.",
    input_schema=INPUTS,
    run=mux_amp_rabi,
    adapter_name="mux/amp_rabi",
    summary_parameters=(
        SummaryElements(
            "tones",
            ("tones",),
            DRIVE_PARAMETERS
            + (
                SummaryParameter("start", ("sweep", "start"), None),
                SummaryParameter("stop", ("sweep", "stop"), None),
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
                    "pi_gain",
                    ("pi_gain",),
                    ("pi_gain_err",),
                    None,
                    element_key=("element",),
                ),
                SummaryEstimate(
                    "pi2_gain",
                    ("pi2_gain",),
                    ("pi2_gain_err",),
                    None,
                    element_key=("element",),
                ),
            ),
        ),
    ),
)
