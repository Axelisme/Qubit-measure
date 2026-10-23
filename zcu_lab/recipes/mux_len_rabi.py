"""Measure mux length Rabi, fit pulse lengths, and review calibration updates.

Recipe n → expts: one shared scan point count for every tone.
"""

from pydantic import Field, StrictInt
from zcu_tools.mcp.measure.recipe_api import (
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
    report_number,
)

from zcu_lab.v2._support.measure.mux_results import MuxSummaryFailure
from zcu_lab.v2.mux.len_rabi.gui import MuxLenRabiSummary

from .mux_helpers import (
    MuxToneReport,
    ToneRangeInputs,
    mux_sweep_edits,
)


class MuxLenRabiInputs(RecipeInputs):
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

    decay: bool | None = Field(
        None,
        description="Fit a decaying oscillation envelope. Null retains analysis settings.",
    )
    fit_phase: bool | None = Field(
        None,
        description="Fit the oscillation phase offset. Null retains analysis settings.",
    )


class MuxLenRabiToneReport(MuxToneReport):
    """One captured tone's estimates and one-sigma errors, or its fit failure.

    Nullable estimates retain absent or nonfinite native summary values as null.
    Identity uses the original tone position, never the participant row offset.
    """

    pi_len: float | None = Field(
        default=None, description="Fitted pi total pulse length in us."
    )
    pi_len_err: float | None = Field(
        default=None, description="One-sigma pi length uncertainty in us."
    )
    pi2_len: float | None = Field(
        default=None, description="Fitted pi/2 total pulse length in us."
    )
    pi2_len_err: float | None = Field(
        default=None, description="One-sigma pi/2 length uncertainty in us."
    )
    rabi_f: float | None = Field(
        default=None, description="Rabi oscillation frequency in MHz."
    )
    rabi_f_err: float | None = Field(
        default=None, description="One-sigma Rabi frequency uncertainty in MHz."
    )


class MuxLenRabiReport(RecipeReport):
    """Native ordered estimates/failures and the confirmed writeback outcome."""

    tones: list[MuxLenRabiToneReport] = Field(
        description="Native ordered fit estimates or failures with original tone identities."
    )


@recipe(
    "mux_len_rabi",
    description="Run mux length Rabi, save raw and analysis, and ask before Apply.",
)
async def mux_len_rabi(
    session: RecipeSession, inputs: MuxLenRabiInputs
) -> MuxLenRabiReport:
    """Measure mux length Rabi, fit pulse lengths, and review calibration updates.

    Select library then inline sources; null keeps captured Form values.
    Overrides run last. Only participant drive/sweep settings are required;
    readout identities remain required for every tone.
    n sets shared Expts; sweeps covers all original tones with start/stop only.
    Frequency uses MHz, length us and gain normalized units.
    Analyze with the supplied decay, fit_phase, then ask before writeback.
    Save raw and completed artifacts after Run, including expected fit failures.
    Operation errors propagate without rolling back earlier edits or saves.
    """
    tab = session.open_recipe_tab("mux/len_rabi", inputs, target_alias="mux")

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
                    "decay": inputs.decay,
                    "fit_phase": inputs.fit_phase,
                }.items()
                if value is not None
            },
            result=MuxLenRabiSummary,
        )
        if fit.result is None:
            return MuxLenRabiReport(
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

        tones: list[MuxLenRabiToneReport] = []
        for outcome in fit.result.tones:
            name = outcome.name
            if isinstance(outcome, MuxSummaryFailure):
                tones.append(
                    MuxLenRabiToneReport(
                        index=outcome.position, name=name, failure=outcome.failure
                    )
                )
            else:
                tones.append(
                    MuxLenRabiToneReport(
                        index=outcome.position,
                        name=name,
                        pi_len=report_number(outcome.pi_len),
                        pi_len_err=report_number(outcome.pi_len_err),
                        pi2_len=report_number(outcome.pi2_len),
                        pi2_len_err=report_number(outcome.pi2_len_err),
                        rabi_f=report_number(outcome.rabi_f),
                        rabi_f_err=report_number(outcome.rabi_f_err),
                    )
                )
        return MuxLenRabiReport(tones=tones, writeback_applied=writeback_applied)
    finally:
        tab.save()


DEFINITION = mux_len_rabi
