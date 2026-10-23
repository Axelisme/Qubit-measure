"""Measure mux Rabi oscillations versus gain and review pi/pi2 pulse updates."""

from pydantic import BaseModel, Field
from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.measure.recipe_api import (
    FieldEdit,
    RecipeSession,
    recipe,
)

from .helpers import report_number
from .mux_helpers import (
    MuxInline,
    MuxLibrary,
    MuxSummaryFailure,
    MuxSummaryTone,
    MuxToneReport,
    ToneRangeInputs,
    mux_overrides,
    mux_sweep_edits,
)


class MuxAmpRabiInputs(FormModel):
    """Choose the mux group and ordered Form edits without inferring calibrations.

    Optional fields default to null. Null keeps Form inputs, uses global mux
    aliases and opens a new tab. The edit lists run in library, inline,
    sweeps/n, overrides order; participants precedes sweeps/n.
    """

    target: str | None = Field(
        default=None, description="Mux Reset alias; null uses global aliases."
    )
    reuse_tab_id: str | None = Field(
        default=None, description="Idle matching tab to reset; null opens a tab."
    )
    participants: list[str] | None = Field(
        default=None,
        description="Qubit names in execution order; null keeps the GUI value.",
    )
    library: list[MuxLibrary] | None = Field(
        default=None, description="Ordered module selections; null keeps the GUI value."
    )
    inline: list[MuxInline] | None = Field(
        default=None,
        description="Ordered captured-module copies, ancestors first; null keeps the GUI value.",
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
    overrides: list[FieldEdit] | None = Field(
        default=None,
        description="Ordered scalar edits at full tone positions; null keeps the GUI value.",
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


class MuxAmpRabiReport(FormModel):
    """Report per-tone fits in native analysis order and the writeback decision.

    tones retains failed rows with their reasons. writeback_applied is true only
    after acceptance has returned; skipping the question writes nothing.
    """

    tones: list[MuxAmpRabiToneReport] = Field(
        description="Native ordered fit estimates or failures with original tone identities."
    )
    writeback_applied: bool = Field(
        default=False, description="Whether accepted calibration items were applied."
    )

    analysis_failure: str | None = Field(
        default=None,
        description="Analysis failure reason; null when analysis completed.",
    )


class MuxAmpRabiFitSummary(MuxSummaryTone):
    """Author view of owner summary fields; no missing values are invented."""

    pi_gain: float | None = Field(
        description="Owner pi_gain in gain; null retains an unavailable estimate."
    )

    pi_gain_err: float | None = Field(
        description="Owner pi_gain_err in gain; null retains an unavailable estimate."
    )

    pi2_gain: float | None = Field(
        description="Owner pi2_gain in gain; null retains an unavailable estimate."
    )

    pi2_gain_err: float | None = Field(
        description="Owner pi2_gain_err in gain; null retains an unavailable estimate."
    )


class MuxAmpRabiSummary(BaseModel):
    """Owner tone rows in analysis order, retaining successes and failures."""

    tones: list[MuxAmpRabiFitSummary | MuxSummaryFailure]


@recipe(
    "mux_amp_rabi",
    description="Run mux amplitude Rabi, save raw and analysis, and ask before Apply.",
)
async def mux_amp_rabi(
    session: RecipeSession, inputs: MuxAmpRabiInputs
) -> MuxAmpRabiReport:
    """Measure simultaneous gain oscillations and report pi/pi2 pulse gains.

    target binds this Reset's mux alias; reuse_tab_id resets an idle matching
    tab. library selects modules, inline copies captures outer-first, then
    participants and overrides edit the Form. Integer paths use full original
    tone positions, not participant rows. Null keeps GUI input. Frequencies use
    MHz, times us and gains normalized units. sweeps supplies start/stop for
    every tone in original order; a length mismatch raises before applying edits.
    n sets the shared Expts; per-tone n is rejected (only start/stop are accepted).
    Null keeps ranges/count. overrides runs last and may replace these edits.
    The GUI scans each drive gain without inferring a range. All tones must
    validate, including nonparticipants.
    Analyze Primary, then ask before applying calibration items.
    Save raw data and completed artifacts in the Run finalizer.
    Return ordered estimates and the writeback decision after completion.
    Expected top-level analysis failures return a reason without writeback.
    Invalid inputs and other operation failures propagate;
    completed edits and saves are not rolled back after a later failure.
    """
    # Capture this group's defaults without changing global aliases.
    tab = session.open_tab(
        "mux/amp_rabi",
        aliases=None if inputs.target is None else {"mux": inputs.target},
        reuse=inputs.reuse_tab_id,
    )

    # Select sources before copying captures and editing original tone positions.
    for selection in inputs.library or ():
        tab.select(selection.path, ref=selection.key)
    for copy in inputs.inline or ():
        tab.select(copy.path, inline=True)
    edits = [
        FieldEdit(
            path=("participants",),
            value=None
            if inputs.participants is None
            else [name for name in inputs.participants],
        )
    ]
    edits.extend(mux_sweep_edits(tab, inputs.sweeps, inputs.n))
    edits.extend(mux_overrides(inputs.overrides))
    tab.apply(edits, require_valid=True)

    # Acquire before fitting; the finally block saves raw and artifacts.
    await tab.run()
    try:
        fit = await tab.analyze("primary", result=MuxAmpRabiSummary)
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
