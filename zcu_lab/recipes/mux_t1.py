"""Measure simultaneous T1 decays and ask before writing calibrations."""

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


class MuxT1Inputs(FormModel):
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


class MuxT1ToneReport(MuxToneReport):
    """One captured tone's estimates and one-sigma errors, or its fit failure.

    Nullable estimates retain absent or nonfinite native summary values as null.
    Identity uses the original tone position, never the participant row offset.
    """

    t1: float | None = Field(default=None, description="Fitted decay time in us.")
    t1_err: float | None = Field(
        default=None, description="One-sigma decay time uncertainty in us."
    )
    t1b: float | None = Field(
        default=None, description="Optional second decay time in us."
    )
    t1b_err: float | None = Field(
        default=None, description="Optional second decay uncertainty in us."
    )


class MuxT1Report(FormModel):
    """Report per-tone fits in native analysis order and the writeback decision.

    tones retains failed rows with their reasons. writeback_applied is true only
    after acceptance has returned; skipping the question writes nothing.
    """

    tones: list[MuxT1ToneReport] = Field(
        description="Native ordered fit estimates or failures with original tone identities."
    )
    writeback_applied: bool = Field(
        default=False, description="Whether accepted calibration items were applied."
    )


class MuxT1FitSummary(MuxSummaryTone):
    """Author view of owner summary fields; no missing values are invented."""

    t1: float | None = Field(
        description="Owner t1 in us; null retains an unavailable estimate."
    )

    t1_err: float | None = Field(
        description="Owner t1_err in us; null retains an unavailable estimate."
    )

    t1b: float | None = Field(
        description="Owner t1b in us; null retains an unavailable estimate."
    )

    t1b_err: float | None = Field(
        description="Owner t1b_err in us; null retains an unavailable estimate."
    )


class MuxT1Summary(BaseModel):
    """Owner tone rows in analysis order, retaining successes and failures."""

    tones: list[MuxT1FitSummary | MuxSummaryFailure]


@recipe(
    "mux_t1", description="Run mux T1, save raw and analysis, and ask before Apply."
)
async def mux_t1(session: RecipeSession, inputs: MuxT1Inputs) -> MuxT1Report:
    """Excite selected qubits, scan a shared wait, and report their T1 decays.

    target binds this Reset's mux alias; reuse_tab_id resets an idle matching
    tab. library selects modules, inline copies captures outer-first, then
    participants and overrides edit the Form. Integer paths use full original
    tone positions, not participant rows. Null keeps GUI input. Frequencies use
    MHz, times us and gains normalized units; sweep counts use n.
    The GUI runs pi pulses, a shared delay and mux readout. No missing pulse
    calibration or delay range is supplied by this recipe.
    Analyze Primary, then ask before applying calibration items.
    Save raw data and completed artifacts in the Run finalizer.
    Return ordered estimates and the writeback decision after completion.
    Invalid inputs and operation failures propagate;
    completed edits and saves are not rolled back after a later failure.
    """
    # Capture this group's defaults without changing global aliases.
    tab = session.open_tab(
        "mux/t1",
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
        fit = await tab.analyze("primary", result=MuxT1Summary)

        # Asking does not write; only an accepted answer applies calibration items.
        answer = await tab.ask_writeback()
        writeback_applied = False
        if answer.accepted:
            receipt = tab.accept_writeback(answer.items)
            receipt.raise_for_error()
            writeback_applied = bool(receipt.items)

        tones: list[MuxT1ToneReport] = []
        for outcome in fit.result.tones:
            name = outcome.name
            if isinstance(outcome, MuxSummaryFailure):
                tones.append(
                    MuxT1ToneReport(
                        index=outcome.position, name=name, failure=outcome.failure
                    )
                )
            else:
                tones.append(
                    MuxT1ToneReport(
                        index=outcome.position,
                        name=name,
                        t1=report_number(outcome.t1),
                        t1_err=report_number(outcome.t1_err),
                        t1b=report_number(outcome.t1b),
                        t1b_err=report_number(outcome.t1b_err),
                    )
                )
        return MuxT1Report(tones=tones, writeback_applied=writeback_applied)
    finally:
        tab.save()


DEFINITION = mux_t1
