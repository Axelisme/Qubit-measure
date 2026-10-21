"""Measure mux qubit spectra and review frequency and linewidth writeback."""

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


def mux_twotone_spectrum(
    session: RecipeSession, **inputs: Unpack[MuxInputs]
) -> RecipeGenerator:
    """Measure selected qubit spectra, fit them and ask before writing calibration.

    session opens a mux/twotone_spectrum tab. target selects the mux group; reuse_tab_id
    resets an idle tab. participants gives qubit names in measurement order.
    library selects path/key pairs, inline copies references outer-first, then
    overrides edits scalar paths in one batch. Integer path segments are full
    tone positions, not participant rows. Values use MHz/us/native gain; sweep
    counts use n. Omitted or None inputs retain Form defaults, including authored
    expressions. All tones must validate, even those not participating.
    Raw data and Primary analysis are saved. Cancellation stops later steps;
    skipping writeback changes no calibration. Operation errors propagate.
    """
    # 1. Open or reset with the requested mux group before editing any values.
    target = inputs.get("target")
    participants = inputs.get("participants")
    tab = session.open_tab(
        "mux/twotone_spectrum",
        aliases=None if target is None else {"mux": target},
        reuse=inputs.get("reuse_tab_id"),
    )

    # 2. Select sources, then copy references before changing their scalar leaves.
    for selection in inputs.get("library") or ():
        tab.use_library(tuple(selection["path"]), selection["key"])
    for copy in inputs.get("inline") or ():
        tab.use_inline(tuple(copy["path"]))

    # 3. Keep participant selection and scalar overrides in one edit batch.
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

    # 4. Save acquired spectra before fitting; cancellation stops later steps.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()

    # 5. Fit each qubit spectrum and save the Primary analysis.
    analysis = yield from run.analyze_step("primary")
    if analysis.status == "cancelled":
        return

    # 6. Offer calibration updates, but write only after explicit acceptance.
    decision = yield from ask(run.propose_writeback())
    if decision.status == "cancelled":
        return
    if decision.decision == "accepted":
        tab.accept()


DEFINITION = RecipeDefinition(
    name="mux_twotone_spectrum",
    description="Run mux two-tone spectrum, save raw and analysis, and ask before Apply.",
    input_schema=INPUTS,
    run=mux_twotone_spectrum,
    adapter_name="mux/twotone_spectrum",
    summary_parameters=(
        SummaryElements(
            "tones",
            ("tones",),
            DRIVE_PARAMETERS
            + (
                SummaryParameter("start", ("sweep", "start"), "MHz"),
                SummaryParameter("stop", ("sweep", "stop"), "MHz"),
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
                    "freq", ("freq",), ("freq_err",), "MHz", element_key=("element",)
                ),
                SummaryEstimate(
                    "fwhm", ("fwhm",), ("fwhm_err",), "MHz", element_key=("element",)
                ),
            ),
        ),
    ),
)
