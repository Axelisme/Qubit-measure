"""Measure mux length Rabi, fit pulse lengths, and review calibration updates."""

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


class MuxLenRabiInputs(FormModel):
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


class MuxLenRabiReport(FormModel):
    """Report per-tone fits in native analysis order and the writeback decision.

    tones retains failed rows with their reasons. writeback_applied is true only
    after acceptance has returned; skipping the question writes nothing.
    """

    tones: list[MuxLenRabiToneReport] = Field(
        description="Native ordered fit estimates or failures with original tone identities."
    )
    writeback_applied: bool = Field(
        default=False, description="Whether accepted calibration items were applied."
    )


class MuxLenRabiFitSummary(MuxSummaryTone):
    """Author view of owner summary fields; no missing values are invented."""

    pi_len: float | None = Field(
        description="Owner pi_len in us; null retains an unavailable estimate."
    )

    pi_len_err: float | None = Field(
        description="Owner pi_len_err in us; null retains an unavailable estimate."
    )

    pi2_len: float | None = Field(
        description="Owner pi2_len in us; null retains an unavailable estimate."
    )

    pi2_len_err: float | None = Field(
        description="Owner pi2_len_err in us; null retains an unavailable estimate."
    )

    rabi_f: float | None = Field(
        description="Owner rabi_f in MHz; null retains an unavailable estimate."
    )

    rabi_f_err: float | None = Field(
        description="Owner rabi_f_err in MHz; null retains an unavailable estimate."
    )


class MuxLenRabiSummary(BaseModel):
    """Owner tone rows in analysis order, retaining successes and failures."""

    tones: list[MuxLenRabiFitSummary | MuxSummaryFailure]


@recipe(
    "mux_len_rabi",
    description="Run mux length Rabi, save raw and analysis, and ask before Apply.",
)
async def mux_len_rabi(
    session: RecipeSession, inputs: MuxLenRabiInputs
) -> MuxLenRabiReport:
    """Sweep simultaneous pulse lengths and report pi/pi2 length calibrations.

    target binds this Reset's mux alias; reuse_tab_id resets an idle matching
    tab. library selects modules, inline copies captures outer-first, then
    participants and overrides edit the Form. Integer paths use full original
    tone positions, not participant rows. Null keeps GUI input. Frequencies use
    MHz, times us and gains normalized units; sweep counts use n.
    The GUI uses const or flat_top drives with equal sweep counts. No range
    or calibration is inferred. Reps, rounds and recovery remain shared.
    Analyze Primary, then ask before applying calibration items.
    Save raw data and completed artifacts in the Run finalizer.
    Return ordered estimates and the writeback decision after completion.
    Invalid inputs and operation failures propagate;
    completed edits and saves are not rolled back after a later failure.
    """
    # Capture this group's defaults without changing global aliases.
    tab = session.open_tab(
        "mux/len_rabi",
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
        fit = await tab.analyze("primary", result=MuxLenRabiSummary)

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
