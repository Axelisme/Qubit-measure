"""Twotone spectroscopy across the shared coil, with one native picker per Q.

Recipe n → frequency.expts; flux_n → flux Expts.
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
    MuxTwotoneFluxInputs,
    mux_flux_analysis_updates,
)
from .mux_helpers import ToneRangeInputs


class MuxTwotoneFluxGlobalInputs(MuxTwotoneFluxInputs):
    """One native coil range shared by all participants.

    flux uses the captured group source; omission keeps Form bounds; null clears them.
    flux_n is the common positive DC point count.
    """

    flux: ToneRangeInputs | None | Keep = Field(
        default=KEEP,
        description="Native coil start/stop; omission keeps the Form range; null clears its bounds. No unit conversion.",
    )


@recipe(
    "mux_twotone_flux_global",
    description="Map mux twotone spectra over global DC flux, pick each Q, ask before selected writeback, then Save.",
)
async def mux_twotone_flux_global(
    session: RecipeSession, inputs: MuxTwotoneFluxGlobalInputs
) -> MuxFluxReport:
    """Twotone spectroscopy across the shared coil, with one native picker per Q.

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
        "mux/twotone_flux_global",
        target={"mux": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.set(
        "participants",
        [*inputs.participants]
        if isinstance(inputs.participants, list)
        else inputs.participants,
    )
    if not isinstance(inputs.sweeps, Keep):
        input_tones = tab.read(
            (
                "frequency",
                "tones",
            )
        ).value
        if not isinstance(input_tones, list):
            raise RecipeNeedsParameters(
                (MissingParameter("frequency.tones", "Full tone list is unavailable."),)
            )
        if inputs.sweeps is not None and len(inputs.sweeps) != len(input_tones):
            raise RecipeNeedsParameters(
                (
                    MissingParameter(
                        "sweeps",
                        f"Expected {len(input_tones)} ranges in original tone order; missing indices {list(range(len(inputs.sweeps), len(input_tones)))}; extra indices {list(range(len(input_tones), len(inputs.sweeps)))}.",
                    ),
                )
            )
        for index in range(len(input_tones)):
            tab.set(
                f"frequency.tones.{index}.sweep",
                None if inputs.sweeps is None else inputs.sweeps[index],
            )
    tab.set("frequency.expts", inputs.n)
    if isinstance(inputs.flux, ToneRangeInputs):
        tab.set("flux.start", inputs.flux.start)
        tab.set("flux.stop", inputs.flux.stop)
    elif inputs.flux is None:
        tab.set("flux.start", None)
        tab.set("flux.stop", None)
    tab.set("flux.n", inputs.flux_n)
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


DEFINITION = mux_twotone_flux_global
