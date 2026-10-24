"""Onetone spectroscopy across zipped local DC lines, with one native picker per Q.

Recipe n → frequency.df Expts; flux_n → flux_expts.
"""

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    RecipeSession,
    recipe,
    report_number,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)

from zcu_lab.v2._support.measure.mux_flux_adapter import MuxFluxPickSummary

from .mux_flux_helpers import (
    MuxFluxReport,
    MuxFluxToneReport,
    MuxOnetoneFluxInputs,
    mux_flux_analysis_updates,
)
from .mux_helpers import ToneRangeInputs


class MuxOnetoneFluxLocalInputs(MuxOnetoneFluxInputs):
    """Native DC ranges for every original tone, including nonparticipants.

    flux_sweeps must cover the complete list; omission keeps Form bounds; null clears them.
    flux_n is shared by zipped local lines, not a Cartesian product.
    """

    flux_sweeps: list[ToneRangeInputs] | None | Keep = Field(
        default=KEEP,
        description="Native local DC start/stop for every tone in original order; omission keeps ranges; null clears their bounds. No per-tone point count.",
    )


@recipe(
    "mux_onetone_flux_local",
    description="Map mux onetone spectra over local DC flux, pick each Q, ask before selected writeback, then Save.",
)
async def mux_onetone_flux_local(
    session: RecipeSession, inputs: MuxOnetoneFluxLocalInputs
) -> MuxFluxReport:
    """Onetone spectroscopy across zipped local DC lines, with one native picker per Q.

    Open captures Form sources; omitted inputs keep values and null clears them.
    Overrides run last. Only participant drive/sweep settings are required;
    readout identities remain required for every tone.
    n sets shared frequency Expts; flux_n sets DC Expts (zipped for local lines).
    Picker flux_half/flux_integer maps seed native DC lines by captured Q name.
    Analyze each participant in acquisition order, ask, then accept selected items.
    Save raw and completed artifacts after Run, including expected fit failures.
    Operation errors propagate without rolling back earlier edits or saves.
    """
    tab = session.open_tab(
        "mux/onetone_flux_local",
        target={"mux": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.set("participants", inputs.participants)
    if isinstance(inputs.df, ToneRangeInputs):
        tab.set("frequency.df.start", inputs.df.start)
        tab.set("frequency.df.stop", inputs.df.stop)
    elif inputs.df is None:
        tab.set("frequency.df.start", None)
        tab.set("frequency.df.stop", None)
    tab.set("frequency.df.n", inputs.n)
    if not isinstance(inputs.flux_sweeps, Keep):
        input_tones = tab.read("tones").value
        if not isinstance(input_tones, list):
            raise RecipeNeedsParameters(
                (MissingParameter("tones", "Full tone list is unavailable."),)
            )
        if inputs.flux_sweeps is not None and len(inputs.flux_sweeps) != len(
            input_tones
        ):
            raise RecipeNeedsParameters(
                (
                    MissingParameter(
                        "flux_sweeps",
                        f"Expected {len(input_tones)} ranges in original tone order; missing indices {list(range(len(inputs.flux_sweeps), len(input_tones)))}; extra indices {list(range(len(input_tones), len(inputs.flux_sweeps)))}.",
                    ),
                )
            )
        for index in range(len(input_tones)):
            tab.set(
                f"tones.{index}.sweep",
                None if inputs.flux_sweeps is None else inputs.flux_sweeps[index],
            )
    tab.set("flux_expts", inputs.flux_n)
    tab.set("flux_unit", inputs.flux_unit)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

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
