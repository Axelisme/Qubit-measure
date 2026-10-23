"""Measure mux probe-off/on IQ clouds, fit GE separation, and review calibration."""

from pydantic import BaseModel, Field, StrictInt
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


class MuxSingleshotGEInputs(FormModel):
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
    participants: list[StrictInt] | None = Field(
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


class MuxGEToneReport(MuxToneReport):
    """One captured qubit's GE separation estimates, or its Primary fit failure.

    Scalars use the GE owner's normalized ADC coordinates; theta is in radians.
    Absent or nonfinite values remain null. Identity uses the original tone
    position, never the participant row offset.
    """

    fidelity: float | None = Field(
        default=None, description="GE assignment fidelity, dimensionless."
    )
    theta: float | None = Field(
        default=None, description="IQ rotation angle in radians."
    )
    threshold: float | None = Field(
        default=None, description="GE decision threshold in normalized ADC units."
    )
    ge_s: float | None = Field(
        default=None, description="GE cloud width in normalized ADC units."
    )


class MuxGEPostToneReport(MuxToneReport):
    """One captured qubit's selected radius, or its Post analysis failure.

    ge_radius uses normalized ADC units; absent or nonfinite values remain null.
    Rows keep the Post analysis order independently of Primary outcomes.
    """

    ge_radius: float | None = Field(
        default=None,
        description="Selected GE acceptance radius in normalized ADC units.",
    )


class MuxSingleshotGEReport(FormModel):
    """Report GE calibration scalars, Post radii and the writeback decision.

    tones and post_tones preserve native row order, original identities and
    failures for each stage. writeback_applied is true only after accepted
    calibration items have been applied. This is not a joint confusion matrix.
    """

    tones: list[MuxGEToneReport] = Field(
        description="Primary GE estimates or failures in native order."
    )
    post_tones: list[MuxGEPostToneReport] = Field(
        description="Post radius estimates or failures in native order."
    )
    writeback_applied: bool = Field(
        default=False, description="Whether accepted calibration items were applied."
    )

    analysis_failure: str | None = Field(
        default=None,
        description="Analysis failure reason; null when analysis completed.",
    )


class MuxGEFitSummary(MuxSummaryTone):
    """Author view of owner summary fields; no missing values are invented."""

    fidelity: float | None = Field(
        description="Owner fidelity in dimensionless; null retains an unavailable estimate."
    )

    theta: float | None = Field(
        description="Owner theta in rad; null retains an unavailable estimate."
    )

    threshold: float | None = Field(
        description="Owner threshold in ADC units; null retains an unavailable estimate."
    )

    ge_s: float | None = Field(
        description="Owner ge_s in ADC units; null retains an unavailable estimate."
    )


class MuxGESummary(BaseModel):
    """Owner tone rows in analysis order, retaining successes and failures."""

    tones: list[MuxGEFitSummary | MuxSummaryFailure]


class MuxGEPostFitSummary(MuxSummaryTone):
    """Author view of owner summary fields; no missing values are invented."""

    ge_radius: float | None = Field(
        description="Owner ge_radius in ADC units; null retains an unavailable estimate."
    )


class MuxGEPostSummary(BaseModel):
    """Owner Post tone rows in their own native analysis order."""

    tones: list[MuxGEPostFitSummary | MuxSummaryFailure]


@recipe(
    "mux_singleshot_ge",
    description="Run mux single-shot GE, save raw and analysis, and ask before Apply.",
)
async def mux_singleshot_ge(
    session: RecipeSession, inputs: MuxSingleshotGEInputs
) -> MuxSingleshotGEReport:
    """Compare simultaneous probe-off/on shots and report each qubit's GE fit.

    target binds this Reset's mux alias; reuse_tab_id resets an idle matching
    tab. library selects modules, inline copies captures outer-first, then
    participants and overrides edit the Form. Paths use full original tone
    positions, MHz, us and native gain. Null keeps GUI input. No missing pulse
    calibration is supplied. Copy referenced ancestors before their children.
    Fit Primary separation, then estimate Post radius and confusion.
    Ask before applying calibration items. Save raw shots and completed artifacts
    in the Run finalizer. Return ordered GE scalars, radii and the writeback
    decision. Skipping writes nothing. Expected analysis failures return a reason
    without writeback, retaining Primary estimates if only Post failed. Invalid
    inputs and other operation failures propagate without undoing edits or saves.
    """
    # Capture this group's defaults without changing global aliases.
    tab = session.open_tab(
        "mux/singleshot_ge",
        aliases=None if inputs.target is None else {"mux": inputs.target},
        reuse=inputs.reuse_tab_id,
    )

    # Select modules before copying captures and editing original tone positions.
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

    # Acquire paired shots before fitting GE separation.
    await tab.run()
    try:
        primary = await tab.analyze("primary", result=MuxGESummary)
        if primary.result is None:
            return MuxSingleshotGEReport(
                tones=[],
                post_tones=[],
                writeback_applied=False,
                analysis_failure=primary.reason,
            )

        # Post uses the adopted Primary fit for radius and confusion diagnostics.
        post = await tab.analyze("post", result=MuxGEPostSummary)
        if post.result is None:
            return MuxSingleshotGEReport(
                tones=_primary_tone_reports(primary.result),
                post_tones=[],
                writeback_applied=False,
                analysis_failure=post.reason,
            )

        # Asking does not write; only an accepted answer applies calibration items.
        answer = await tab.ask_writeback()
        writeback_applied = False
        if answer.accepted:
            receipt = tab.accept_writeback(answer.items)
            receipt.raise_for_error()
            writeback_applied = bool(receipt.items)

        return MuxSingleshotGEReport(
            tones=_primary_tone_reports(primary.result),
            post_tones=_post_tone_reports(post.result),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


def _primary_tone_reports(result: MuxGESummary) -> list[MuxGEToneReport]:
    """Project native Primary outcomes in order, keeping identities and failures."""
    tones: list[MuxGEToneReport] = []
    for outcome in result.tones:
        name = outcome.name
        if isinstance(outcome, MuxSummaryFailure):
            tones.append(
                MuxGEToneReport(
                    index=outcome.position, name=name, failure=outcome.failure
                )
            )
        else:
            tones.append(
                MuxGEToneReport(
                    index=outcome.position,
                    name=name,
                    fidelity=report_number(outcome.fidelity),
                    theta=report_number(outcome.theta),
                    threshold=report_number(outcome.threshold),
                    ge_s=report_number(outcome.ge_s),
                )
            )
    return tones


def _post_tone_reports(result: MuxGEPostSummary) -> list[MuxGEPostToneReport]:
    """Project native Post radii separately, without pairing them to Primary rows."""
    post_tones: list[MuxGEPostToneReport] = []
    for outcome in result.tones:
        name = outcome.name
        if isinstance(outcome, MuxSummaryFailure):
            post_tones.append(
                MuxGEPostToneReport(
                    index=outcome.position, name=name, failure=outcome.failure
                )
            )
        else:
            post_tones.append(
                MuxGEPostToneReport(
                    index=outcome.position,
                    name=name,
                    ge_radius=report_number(outcome.ge_radius),
                )
            )
    return post_tones


DEFINITION = mux_singleshot_ge
