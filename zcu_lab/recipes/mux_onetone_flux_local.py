"""Onetone spectroscopy across zipped local DC lines, with one native picker per Q.

Recipe n → frequency.df Expts; flux_n → flux_expts.
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
    MuxOnetoneFluxInputs,
    flux_range_edits,
    mux_flux_analysis_updates,
    onetone_flux_edits,
)
from .mux_helpers import ToneRangeInputs


class MuxOnetoneFluxLocalInputs(MuxOnetoneFluxInputs):
    """Native DC ranges for every original tone, including nonparticipants.

    flux_sweeps must cover the complete list; null keeps Form bounds.
    flux_n is shared by zipped local lines, not a Cartesian product.
    """

    flux_sweeps: list[ToneRangeInputs] | None = Field(
        None,
        description="Native local DC start/stop for every tone in original order; null retains ranges. No per-tone point count.",
    )


@recipe(
    "mux_onetone_flux_local",
    description="Map mux onetone spectra over local DC flux, pick each Q, ask before selected writeback, then Save.",
)
async def mux_onetone_flux_local(
    session: RecipeSession, inputs: MuxOnetoneFluxLocalInputs
) -> MuxFluxReport:
    """Onetone spectroscopy across zipped local DC lines, with one native picker per Q.

    Select library then inline sources; null keeps captured Form values.
    Overrides run last. Only participant drive/sweep settings are required;
    readout identities remain required for every tone.
    n sets shared frequency Expts; flux_n sets DC Expts (zipped for local lines).
    Picker flux_half/flux_integer maps seed native DC lines by captured Q name.
    Analyze each participant in acquisition order, ask, then accept selected items.
    Save raw and completed artifacts after Run, including expected fit failures.
    Operation errors propagate without rolling back earlier edits or saves.
    """
    tab = session.open_recipe_tab("mux/onetone_flux_local", inputs, target_alias="mux")
    tab.select_sources(inputs)
    edits = onetone_flux_edits(inputs)
    edits.extend(flux_range_edits(tab, inputs.flux_sweeps, ("tones",)))
    edits.append(FieldEdit(path=("flux_expts",), value=inputs.flux_n))
    tab.apply_inputs(
        inputs,
        {
            ("participants",): inputs.participants,
            ("flux_unit",): inputs.flux_unit,
        },
        edits=edits,
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


DEFINITION = mux_onetone_flux_local
