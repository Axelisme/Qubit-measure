"""Measure mux Hahn Echo, save the results and review coherence writeback."""

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


class MuxT2EchoInputs(FormModel):
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
    participants: list[int] | None = Field(
        default=None,
        description="Unique original tone indices in acquisition order; null keeps the captured Form selection.",
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


class MuxT2EchoToneReport(MuxToneReport):
    """One captured tone's estimates and one-sigma errors, or its fit failure.

    Nullable estimates retain absent or nonfinite native summary values as null.
    Identity uses the original tone position, never the participant row offset.
    """

    t2: float | None = Field(
        default=None, description="Fitted Echo coherence time in us."
    )
    t2_err: float | None = Field(
        default=None, description="One-sigma Echo time uncertainty in us."
    )
    detune: float | None = Field(
        default=None, description="Fitted fringe rate in MHz; zero for decay fits."
    )
    detune_err: float | None = Field(
        default=None, description="One-sigma fringe rate uncertainty in MHz."
    )


class MuxT2EchoReport(FormModel):
    """Report per-tone fits in native analysis order and the writeback decision.

    tones retains failed rows with their reasons. writeback_applied is true only
    after acceptance has returned; skipping the question writes nothing.
    """

    tones: list[MuxT2EchoToneReport] = Field(
        description="Native ordered fit estimates or failures with original tone identities."
    )
    writeback_applied: bool = Field(
        default=False, description="Whether accepted calibration items were applied."
    )

    analysis_failure: str | None = Field(
        default=None,
        description="Analysis failure reason; null when analysis completed.",
    )


class MuxT2EchoFitSummary(MuxSummaryTone):
    """Author view of owner summary fields; no missing values are invented."""

    t2: float | None = Field(
        description="Owner t2 in us; null retains an unavailable estimate."
    )

    t2_err: float | None = Field(
        description="Owner t2_err in us; null retains an unavailable estimate."
    )

    detune: float | None = Field(
        description="Owner detune in MHz; null retains an unavailable estimate."
    )

    detune_err: float | None = Field(
        description="Owner detune_err in MHz; null retains an unavailable estimate."
    )


class MuxT2EchoSummary(BaseModel):
    """Owner tone rows in analysis order, retaining successes and failures."""

    tones: list[MuxT2EchoFitSummary | MuxSummaryFailure]


@recipe(
    "mux_t2echo",
    description="Run mux Echo, save raw and analysis, and ask before Apply.",
)
async def mux_t2echo(
    session: RecipeSession, inputs: MuxT2EchoInputs
) -> MuxT2EchoReport:
    """Measure mux Hahn Echo and report coherence times and fringe rates.

    target binds this Reset's mux alias; reuse_tab_id resets an idle matching
    tab. library selects modules, inline copies captures outer-first, then
    participants and overrides edit the Form. Integer paths use full original
    tone positions, not participant rows. Null keeps GUI input. Frequencies use
    MHz, times us and gains normalized units; sweep counts use n.
    The GUI runs x90, half wait, x180, half wait and x90 with shared delays.
    No missing pulse calibration or delay range is supplied by this recipe.
    Analyze Primary, then ask before applying calibration items.
    Save raw data and completed artifacts in the Run finalizer.
    Return ordered estimates and the writeback decision after completion.
    Expected top-level analysis failures return a reason without writeback.
    Invalid inputs and other operation failures propagate;
    completed edits and saves are not rolled back after a later failure.
    """
    # Capture this group's defaults without changing global aliases.
    tab = session.open_tab(
        "mux/t2echo",
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
            value=None if inputs.participants is None else list(inputs.participants),
        )
    ]
    edits.extend(mux_overrides(inputs.overrides))
    tab.apply(edits, require_valid=True)

    # Acquire before fitting; the finally block saves raw and artifacts.
    await tab.run()
    try:
        fit = await tab.analyze("primary", result=MuxT2EchoSummary)
        if fit.result is None:
            return MuxT2EchoReport(
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

        tones: list[MuxT2EchoToneReport] = []
        for outcome in fit.result.tones:
            name = outcome.name
            if isinstance(outcome, MuxSummaryFailure):
                tones.append(
                    MuxT2EchoToneReport(
                        index=outcome.position, name=name, failure=outcome.failure
                    )
                )
            else:
                tones.append(
                    MuxT2EchoToneReport(
                        index=outcome.position,
                        name=name,
                        t2=report_number(outcome.t2),
                        t2_err=report_number(outcome.t2_err),
                        detune=report_number(outcome.detune),
                        detune_err=report_number(outcome.detune_err),
                    )
                )
        return MuxT2EchoReport(tones=tones, writeback_applied=writeback_applied)
    finally:
        tab.save()


DEFINITION = mux_t2echo
