"""Measure mux Rabi oscillations versus gain and review pi/pi2 pulse updates.

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
from zcu_lab.v2.mux.amp_rabi.gui import MuxAmpRabiSummary

from .mux_helpers import (
    MuxToneReport,
    ToneRangeInputs,
    mux_sweep_edits,
)


class MuxAmpRabiInputs(RecipeInputs):
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

    skip: int | None = Field(
        None,
        description="Points to skip before fitting the gain oscillation. Null retains analysis settings.",
    )


class MuxAmpRabiToneReport(MuxToneReport):
    """One captured tone's estimates and one-sigma errors, or its fit failure.

    Nullable estimates retain absent or nonfinite native summary values as null.
    Identity uses the original tone position, never the participant row offset.
    """

    pi_gain: float | None = Field(
        default=None, description="Fitted pi gain in normalized generator units."
    )
    pi_gain_err: float | None = Field(
        default=None, description="One-sigma pi gain uncertainty."
    )
    pi2_gain: float | None = Field(
        default=None, description="Fitted pi/2 gain in normalized generator units."
    )
    pi2_gain_err: float | None = Field(
        default=None, description="One-sigma pi/2 gain uncertainty."
    )


class MuxAmpRabiReport(RecipeReport):
    """Native ordered estimates/failures and the confirmed writeback outcome."""

    tones: list[MuxAmpRabiToneReport] = Field(
        description="Native ordered fit estimates or failures with original tone identities."
    )


@recipe(
    "mux_amp_rabi",
    description="Run mux amplitude Rabi, save raw and analysis, and ask before Apply.",
)
async def mux_amp_rabi(
    session: RecipeSession, inputs: MuxAmpRabiInputs
) -> MuxAmpRabiReport:
    """Measure mux Rabi oscillations versus gain and review pi/pi2 pulse updates.

    Select library then inline sources; null keeps captured Form values.
    Overrides run last. Only participant drive/sweep settings are required;
    readout identities remain required for every tone.
    n sets shared Expts; sweeps covers all original tones with start/stop only.
    Frequency uses MHz, length us and gain normalized units.
    Analyze with the supplied skip, then ask before writeback.
    Save raw and completed artifacts after Run, including expected fit failures.
    Operation errors propagate without rolling back earlier edits or saves.
    """
    tab = session.open_recipe_tab("mux/amp_rabi", inputs, target_alias="mux")

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
                for key, value in {"skip": inputs.skip}.items()
                if value is not None
            },
            result=MuxAmpRabiSummary,
        )
        if fit.result is None:
            return MuxAmpRabiReport(
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

        tones: list[MuxAmpRabiToneReport] = []
        for outcome in fit.result.tones:
            name = outcome.name
            if isinstance(outcome, MuxSummaryFailure):
                tones.append(
                    MuxAmpRabiToneReport(
                        index=outcome.position, name=name, failure=outcome.failure
                    )
                )
            else:
                tones.append(
                    MuxAmpRabiToneReport(
                        index=outcome.position,
                        name=name,
                        pi_gain=report_number(outcome.pi_gain),
                        pi_gain_err=report_number(outcome.pi_gain_err),
                        pi2_gain=report_number(outcome.pi2_gain),
                        pi2_gain_err=report_number(outcome.pi2_gain_err),
                    )
                )
        return MuxAmpRabiReport(tones=tones, writeback_applied=writeback_applied)
    finally:
        tab.save()


DEFINITION = mux_amp_rabi
