"""Twotone spectroscopy across the shared coil, with one native picker per Q.

Recipe n → frequency.expts; flux_n → flux Expts.
"""

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import FieldEdit, RecipeSession, recipe

from .helpers import report_number
from .mux_flux_helpers import (
    MuxFluxPickSummary,
    MuxFluxReport,
    MuxFluxToneReport,
    MuxTwotoneFluxInputs,
    flux_range_edits,
    global_flux_edits,
    mux_flux_analysis_updates,
    mux_flux_final_edits,
)
from .mux_helpers import ToneRangeInputs


class MuxTwotoneFluxGlobalInputs(MuxTwotoneFluxInputs):
    """One native coil range shared by all participants.

    flux uses the captured group source; null keeps Form bounds.
    flux_n is the common positive DC point count.
    """

    flux: ToneRangeInputs | None = Field(
        None,
        description="Native coil start/stop; null retains the Form range. No unit conversion.",
    )


@recipe(
    "mux_twotone_flux_global",
    description="Map mux twotone spectra over global DC flux, pick each Q, ask before selected writeback, then Save.",
)
async def mux_twotone_flux_global(
    session: RecipeSession, inputs: MuxTwotoneFluxGlobalInputs
) -> MuxFluxReport:
    """Acquire twotone maps over global DC values, then report per-Q flux lines.

    target binds the Reset mux alias; reuse_tab_id resets an idle matching tab.
    Select library, copy inline sources outer-first, apply ranges/counts, then
    overrides to the flat cfg. Null retains Form inputs. MHz frequency and
    native DC coordinates are never converted. All full tones must validate,
    even nonparticipants. Missing settings stop before acquisition with owner
    paths and reasons. n sets the shared frequency Expts; flux_n sets DC Expts.

    Analyze selected Qs in participant order using original tone indices.
    Each tracked analysis hands off through PendingAnalysis/tab_interact/done.
    Optional flux_half/flux_integer maps seed lines without changing the Run.
    answer only selects writeback proposals, never picker commands.
    Each Q proposes one complete group coil_flux.<Q> record.
    Accept only selected items and use the confirmed receipt for the report.
    Expected analysis failure preserves earlier rows and saves without asking
    for the failed Q. Other errors and root cancellation propagate. Once Run
    returns, finally saves raw and completed artifacts; per-output Save status
    is in runs, not implied by completion. No calibration is inferred here.
    """
    tab = session.open_tab(
        "mux/twotone_flux_global",
        aliases=None if inputs.target is None else {"mux": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    for selection in inputs.library or ():
        tab.select(selection.path, ref=selection.key)
    for copy in inputs.inline or ():
        tab.select(copy.path, inline=True)
    edits = flux_range_edits(tab, inputs.sweeps, ("frequency", "tones"))
    edits.append(FieldEdit(path=("frequency", "expts"), value=inputs.n))
    edits.extend(global_flux_edits(inputs.flux, inputs.flux_n))
    edits.extend(mux_flux_final_edits(inputs))
    tab.apply(edits, require_valid=True)
    updates = mux_flux_analysis_updates(tab, inputs)

    await tab.run()
    tones: list[MuxFluxToneReport] = []
    try:
        for update in updates:
            analysis = await tab.analyze(
                "primary", result=MuxFluxPickSummary, interactive=True, updates=update
            )
            if analysis.result is None:
                return MuxFluxReport(tones=tones, analysis_failure=analysis.reason)
            decision = await tab.ask_writeback()
            applied = False
            if decision.accepted:
                receipt = tab.accept_writeback(decision.items)
                receipt.raise_for_error()
                applied = bool(receipt.items)
            picked = analysis.result
            tones.append(
                MuxFluxToneReport(
                    index=picked.tone,
                    name=picked.qubit,
                    flux_half=report_number(picked.flx_half),
                    flux_integer=report_number(picked.flx_int),
                    flux_period=report_number(picked.flx_period),
                    writeback_applied=applied,
                )
            )
        return MuxFluxReport(tones=tones)
    finally:
        tab.save()


DEFINITION = mux_twotone_flux_global
