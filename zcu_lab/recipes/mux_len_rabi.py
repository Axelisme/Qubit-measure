"""Measure mux length Rabi, fit pulse lengths, and review calibration updates.

Recipe n → expts: one shared scan point count for every tone.
"""

from typing import Annotated

from pydantic import Field, StrictInt
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
    report_number,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)

from zcu_lab.v2._support.measure.mux_results import MuxSummaryFailure
from zcu_lab.v2.mux.len_rabi.gui import MuxLenRabiSummary

from .mux_helpers import (
    MuxToneReport,
    ToneRangeInputs,
)


class MuxLenRabiInputs(RecipeInputs):
    """Omitted Form inputs retain values; null clears them; overrides apply last."""

    participants: list[StrictInt] | None | Keep = Field(
        default=KEEP,
        description="Unique original tone indices in acquisition order; omission keeps the Form selection; null clears it.",
    )
    sweeps: list[ToneRangeInputs] | None | Keep = Field(
        default=KEEP,
        description="Start/stop for every tone in original order, including nonparticipants; omission keeps ranges; null clears their bounds. No per-tone n.",
    )
    n: Annotated[int, Field(strict=True, gt=0)] | None | Keep = Field(
        default=KEEP,
        description="Positive hardware scan point count shared by all tones; omission keeps Expts; null clears it.",
    )

    decay: bool | None | Keep = Field(
        default=KEEP,
        description="Fit a decaying oscillation envelope. Omission keeps analysis settings; null is passed to the analysis owner.",
    )
    fit_phase: bool | None | Keep = Field(
        default=KEEP,
        description="Fit the oscillation phase offset. Omission keeps analysis settings; null is passed to the analysis owner.",
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

    Open captures Form sources; omitted inputs keep values and null clears them.
    Overrides run last. Only participant drive/sweep settings are required;
    readout identities remain required for every tone.
    n sets shared Expts; sweeps covers all original tones with start/stop only.
    Frequency uses MHz, length us and gain normalized units.
    Analyze with the supplied decay, fit_phase, then ask before writeback.
    Save raw and completed artifacts after Run, including expected fit failures.
    Operation errors propagate without rolling back earlier edits or saves.
    """
    tab = session.open_tab(
        "mux/len_rabi", target={"mux": inputs.target}, reuse=inputs.reuse_tab_id
    )
    tab.set(
        "participants",
        [*inputs.participants]
        if isinstance(inputs.participants, list)
        else inputs.participants,
    )
    if not isinstance(inputs.sweeps, Keep):
        input_tones = tab.read(("tones",)).value
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
                f"tones.{index}.sweep",
                None if inputs.sweeps is None else inputs.sweeps[index],
            )
    tab.set("expts", inputs.n)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        fit = await tab.analyze(
            "primary",
            updates={
                "decay": inputs.decay,
                "fit_phase": inputs.fit_phase,
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
