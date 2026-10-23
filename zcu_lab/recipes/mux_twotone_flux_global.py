"""Twotone spectroscopy across the shared coil, with one native picker per Q.

Recipe n → frequency.expts; flux_n → flux Expts.
"""

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    FieldEdit,
    RecipeSession,
    recipe,
    report_number,
)

from zcu_lab.v2._support.measure.mux_flux_adapter import MuxFluxPickSummary

from .mux_flux_helpers import (
    MuxFluxReport,
    MuxFluxToneReport,
    MuxTwotoneFluxInputs,
    flux_range_edits,
    global_flux_edits,
    mux_flux_analysis_updates,
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
    """Twotone spectroscopy across the shared coil, with one native picker per Q.

    Select library then inline sources; null keeps captured Form values.
    Overrides run last. Only participant drive/sweep settings are required;
    readout identities remain required for every tone.
    n sets shared frequency Expts; flux_n sets DC Expts (zipped for local lines).
    Picker flux_half/flux_integer maps seed native DC lines by captured Q name.
    Analyze each participant in acquisition order, ask, then accept selected items.
    Save raw and completed artifacts after Run, including expected fit failures.
    Operation errors propagate without rolling back earlier edits or saves.
    """
    tab = session.open_recipe_tab("mux/twotone_flux_global", inputs, target_alias="mux")
    tab.select_sources(inputs)
    edits = flux_range_edits(tab, inputs.sweeps, ("frequency", "tones"))
    edits.append(FieldEdit(path=("frequency", "expts"), value=inputs.n))
    edits.extend(global_flux_edits(inputs.flux, inputs.flux_n))
    tab.apply_inputs(
        inputs,
        {
            ("flux_unit",): inputs.flux_unit,
        },
        edits=[
            FieldEdit(
                path=("participants",),
                value=None if inputs.participants is None else [*inputs.participants],
            ),
            *edits,
        ],
    )

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
