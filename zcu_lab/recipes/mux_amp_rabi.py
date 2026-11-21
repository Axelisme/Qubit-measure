"""Measure mux Rabi oscillations versus gain and review pi/pi2 pulse updates.

Recipe n → expts: one shared scan point count for every tone.
"""

from typing import Annotated

from pydantic import Field, StrictInt
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    Overrides,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    ReuseTabId,
    SingleTarget,
    recipe,
    report_number,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)

from zcu_lab.v2.mux.amp_rabi.gui import MuxAmpRabiSummary
from zcu_lab.v2.support.measure.mux_results import MuxSummaryFailure

from .mux_helpers import (
    MuxFitToneReport,
    ToneRangeInputs,
)


class MuxAmpRabiInputs(RecipeInputs):
    """Omitted Form inputs retain values; null clears them; overrides apply last."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    participants: list[StrictInt] | None | Keep = Field(
        default=KEEP,
        description="Sets Participants: original tone indices (zero-based) for drives and readout, in run order.",
    )
    sweeps: list[ToneRangeInputs] | None | Keep = Field(
        default=KEEP,
        description="Sets Tones › Gain › Start and Stop (a.u.) for every original tone, including nonparticipants.",
    )
    n: Annotated[int, Field(strict=True, gt=0)] | None | Keep = Field(
        default=KEEP,
        description="Sets Expts (count), the shared number of gain points.",
    )

    skip: int | None | Keep = Field(
        default=KEEP,
        description="Sets Analysis › Skip points (count), the number of leading gain points excluded from the fit.",
    )


class MuxAmpRabiToneReport(MuxFitToneReport):
    """One captured tone's estimates and one-sigma errors, or its fit failure.

    Nullable estimates retain absent or nonfinite native summary values as null.
    Identity uses the original tone position, never the participant row offset.
    """

    pi_gain: float | None = Field(
        default=None,
        description="Fitted pi gain as a DAC full-scale fraction in a.u., within [-1, 1].",
    )
    pi_gain_err: float | None = Field(
        default=None, description="One-sigma pi gain uncertainty."
    )
    pi2_gain: float | None = Field(
        default=None,
        description="Fitted pi/2 gain as a DAC full-scale fraction in a.u., within [-1, 1].",
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

    Open captures Form sources; omitted inputs keep values and null clears them.
    Overrides run last. Only participant drive/sweep settings are required;
    readout identities remain required for every tone.
    n sets shared Expts; sweeps covers all original tones with start/stop only.
    Frequency uses MHz, length µs and gain in a.u. within [-1, 1].
    Analyze with the supplied skip, then ask before writeback.
    Save raw and completed artifacts after Run, including expected fit failures.
    Operation errors propagate without rolling back earlier edits or saves.
    """
    tab = session.open_tab(
        "mux/amp_rabi", target={"mux": inputs.target}, reuse=inputs.reuse_tab_id
    )
    tab.set("participants", inputs.participants)
    if not isinstance(inputs.sweeps, Keep):
        input_tones = tab.read("tones").value
        if not isinstance(input_tones, list):
            raise RecipeNeedsParameters(
                (MissingParameter("tones", "Full tone list is unavailable."),)
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
                f"tones.{index}.gain",
                None if inputs.sweeps is None else inputs.sweeps[index],
            )
    tab.set("expts", inputs.n)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        fit = await tab.analyze(
            "primary",
            updates={"skip": inputs.skip},
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
        for outcome in fit.result.outcomes:
            name = None if fit.result.names is None else fit.result.names[outcome.index]
            if isinstance(outcome, MuxSummaryFailure):
                tones.append(
                    MuxAmpRabiToneReport(
                        index=outcome.index, name=name, failure=outcome.reason
                    )
                )
            else:
                tones.append(
                    MuxAmpRabiToneReport(
                        index=outcome.index,
                        name=name,
                        fit_quality=outcome.quality,
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
