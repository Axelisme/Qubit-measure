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
    mux_overrides,
)


class MuxAmpRabiInputs(FormModel):
    """Choose the mux group and ordered Form edits without inferring calibrations.

    Optional fields default to null. Null keeps Form inputs, uses global mux
    aliases and opens a new tab. The edit lists run in library, inline,
    overrides order; participants precedes overrides.
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
    MHz, times us and gains normalized units; sweep counts use n.
    The GUI scans each drive gain without inferring a range. All tones must
    validate, including nonparticipants.
    Analyze Primary, then ask before applying calibration items.
    Save raw data and completed artifacts in the Run finalizer.
    Return ordered estimates and the writeback decision after completion.
    Invalid inputs and operation failures propagate;
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
    edits.extend(mux_overrides(inputs.overrides))
    tab.apply(edits)

    # Acquire before fitting; the finally block saves raw and artifacts.
    await tab.run()
    try:
        fit = await tab.analyze("primary", result=MuxAmpRabiSummary)

        # Asking does not write; only an accepted answer applies calibration items.
        answer = await tab.ask_writeback()
        if answer.accepted:
            tab.accept_writeback(answer.items)

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
        return MuxAmpRabiReport(tones=tones, writeback_applied=answer.accepted)
    finally:
        tab.save()


DEFINITION = mux_amp_rabi
