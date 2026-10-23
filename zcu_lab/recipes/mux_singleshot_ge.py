"""Measure mux probe-off/on IQ clouds, fit GE separation, and review calibration."""

from typing import Literal

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

from zcu_lab.v2._support.measure.mux_results import MuxSummaryFailure
from zcu_lab.v2.mux.singleshot_ge.gui import MuxGEPostSummary, MuxGESummary

from .mux_helpers import (
    MuxToneReport,
)


class MuxSingleshotGEInputs(RecipeInputs):
    """Omitted Form inputs retain values; null clears them; overrides apply last."""

    participants: list[StrictInt] | None | Keep = Field(
        default=KEEP,
        description="Unique original tone indices in acquisition order; omission keeps the Form selection; null clears it.",
    )

    initial_state: Literal["ground", "excited"] | None | Keep = Field(
        default=KEEP,
        description="Prepared state for GE separation. Omission keeps analysis settings; null is passed to the analysis owner.",
    )
    backend: Literal["pca", "center"] | None | Keep = Field(
        default=KEEP,
        description="Rotation and threshold fit backend. Omission keeps analysis settings; null is passed to the analysis owner.",
    )
    logscale: bool | None | Keep = Field(
        default=KEEP,
        description="Plot histograms on a logarithmic scale. Omission keeps analysis settings; null is passed to the analysis owner.",
    )
    align_t1: bool | None | Keep = Field(
        default=KEEP,
        description="Align the GE model with captured T1. Omission keeps analysis settings; null is passed to the analysis owner.",
    )
    length_ratio: float | None | Keep = Field(
        default=KEEP,
        description="Integration length ratio for GE fitting. Omission keeps analysis settings; null is passed to the analysis owner.",
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


class MuxSingleshotGEReport(RecipeReport):
    """Native ordered estimates/failures and the confirmed writeback outcome."""

    tones: list[MuxGEToneReport] = Field(
        description="Primary GE estimates or failures in native order."
    )
    post_tones: list[MuxGEPostToneReport] = Field(
        description="Post radius estimates or failures in native order."
    )


@recipe(
    "mux_singleshot_ge",
    description="Run mux single-shot GE, save raw and analysis, and ask before Apply.",
)
async def mux_singleshot_ge(
    session: RecipeSession, inputs: MuxSingleshotGEInputs
) -> MuxSingleshotGEReport:
    """Measure mux probe-off/on IQ clouds, fit GE separation, and review calibration.

    Open captures Form sources; omitted inputs keep values and null clears them.
    Overrides run last. Only participant drives are required;
    readout identities remain required for every tone.
    Set the IQ-cloud sample count with a shots override; there is no sweep n.
    Indices are original tone positions.
    Fit Primary GE separation and Post radius, then ask before writeback.
    Save raw and completed artifacts after Run, including expected fit failures.
    Operation errors propagate without rolling back earlier edits or saves.
    """
    tab = session.open_tab(
        "mux/singleshot_ge", target={"mux": inputs.target}, reuse=inputs.reuse_tab_id
    )
    tab.set(
        "participants",
        [*inputs.participants]
        if isinstance(inputs.participants, list)
        else inputs.participants,
    )
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        primary = await tab.analyze(
            "primary",
            updates={
                key: value
                for key, value in {
                    "initial_state": inputs.initial_state,
                    "backend": inputs.backend,
                    "logscale": inputs.logscale,
                    "align_t1": inputs.align_t1,
                    "length_ratio": inputs.length_ratio,
                }.items()
                if not isinstance(value, Keep)
            },
            result=MuxGESummary,
        )
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
