"""Measure mux qubit spectra and review frequency and linewidth writeback.

Recipe n → expts: one shared scan point count for every tone.
"""

from typing import Literal

from pydantic import Field, StrictInt
from zcu_tools.mcp.measure.recipe_api import (
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
    report_number,
)

from zcu_lab.v2._support.measure.mux_results import MuxSummaryFailure
from zcu_lab.v2.mux.twotone_spectrum.gui import MuxTwotoneSpectrumSummary

from .mux_helpers import (
    MuxToneReport,
    ToneRangeInputs,
    mux_sweep_edits,
)


class MuxTwotoneSpectrumInputs(RecipeInputs):
    """Experiment inputs; null retains captured Form values, overrides apply last."""

    participants: list[StrictInt] | None = Field(
        default=None,
        description="Unique original tone indices in acquisition order; null keeps the captured Form selection.",
    )
    sweeps: list[ToneRangeInputs] | None = Field(
        default=None,
        description="Start/stop for every tone in original order, including nonparticipants; null keeps ranges. No per-tone n.",
    )
    n: int | None = Field(
        default=None,
        strict=True,
        gt=0,
        description="Positive hardware scan point count shared by all tones; null keeps Expts.",
    )

    model_type: Literal["lor", "sinc"] | None = Field(
        None,
        description="Frequency fit model: Lorentzian or sinc. Null retains analysis settings.",
    )
    plot_fit: bool | None = Field(
        None,
        description="Draw the fitted frequency curve. Null retains analysis settings.",
    )


class MuxTwotoneSpectrumToneReport(MuxToneReport):
    """One captured tone's estimates and one-sigma errors, or its fit failure.

    Nullable estimates retain absent or nonfinite native summary values as null.
    Identity uses the original tone position, never the participant row offset.
    """

    freq: float | None = Field(
        default=None, description="Fitted transition frequency in MHz."
    )
    freq_err: float | None = Field(
        default=None, description="One-sigma transition frequency uncertainty in MHz."
    )
    fwhm: float | None = Field(
        default=None, description="Fitted transition linewidth in MHz."
    )
    fwhm_err: float | None = Field(
        default=None, description="One-sigma linewidth uncertainty in MHz."
    )


class MuxTwotoneSpectrumReport(RecipeReport):
    """Native ordered estimates/failures and the confirmed writeback outcome."""

    tones: list[MuxTwotoneSpectrumToneReport] = Field(
        description="Native ordered fit estimates or failures with original tone identities."
    )


@recipe(
    "mux_twotone_spectrum",
    description="Run mux two-tone spectrum, save raw and analysis, and ask before Apply.",
)
async def mux_twotone_spectrum(
    session: RecipeSession, inputs: MuxTwotoneSpectrumInputs
) -> MuxTwotoneSpectrumReport:
    """Measure mux qubit spectra and review frequency and linewidth writeback.

    Select library then inline sources; null keeps captured Form values.
    Overrides run last. Only participant drive/sweep settings are required;
    readout identities remain required for every tone.
    n sets shared Expts; sweeps covers all original tones with start/stop only.
    Frequency uses MHz, length us and gain normalized units.
    Analyze with the supplied model_type, plot_fit, then ask before writeback.
    Save raw and completed artifacts after Run, including expected fit failures.
    Operation errors propagate without rolling back earlier edits or saves.
    """
    tab = session.open_recipe_tab("mux/twotone_spectrum", inputs, target_alias="mux")

    tab.select_sources(inputs)
    tab.apply_inputs(
        inputs,
        {("participants",): inputs.participants},
        edits=mux_sweep_edits(tab, inputs.sweeps, inputs.n),
    )

    # Acquire before fitting; the finally block saves raw and artifacts.
    await tab.run()
    try:
        fit = await tab.analyze(
            "primary",
            updates={
                key: value
                for key, value in {
                    "model_type": inputs.model_type,
                    "plot_fit": inputs.plot_fit,
                }.items()
                if value is not None
            },
            result=MuxTwotoneSpectrumSummary,
        )
        if fit.result is None:
            return MuxTwotoneSpectrumReport(
                tones=[],
                writeback_applied=False,
                analysis_failure=fit.reason,
            )

        # Asking does not write; only an accepted answer applies calibration items.
        answer = await tab.ask_writeback()
        writeback_applied = False
        if answer.accepted:
            receipt = tab.accept_writeback(answer.items)
            receipt.raise_for_error()
            writeback_applied = bool(receipt.items)

        tones: list[MuxTwotoneSpectrumToneReport] = []
        for outcome in fit.result.tones:
            name = outcome.name
            if isinstance(outcome, MuxSummaryFailure):
                tones.append(
                    MuxTwotoneSpectrumToneReport(
                        index=outcome.position, name=name, failure=outcome.failure
                    )
                )
            else:
                tones.append(
                    MuxTwotoneSpectrumToneReport(
                        index=outcome.position,
                        name=name,
                        freq=report_number(outcome.freq),
                        freq_err=report_number(outcome.freq_err),
                        fwhm=report_number(outcome.fwhm),
                        fwhm_err=report_number(outcome.fwhm_err),
                    )
                )
        return MuxTwotoneSpectrumReport(
            tones=tones, writeback_applied=writeback_applied
        )
    finally:
        tab.save()


DEFINITION = mux_twotone_spectrum
