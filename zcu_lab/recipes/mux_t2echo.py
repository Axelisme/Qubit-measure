"""Measure mux Hahn Echo, save the results and review coherence writeback."""

from typing import Literal

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

from zcu_lab.v2.support.measure.mux_results import (
    MuxCoherenceSummary,
    MuxSummaryFailure,
)

from .mux_helpers import (
    MuxFitToneReport,
)


class MuxT2EchoInputs(RecipeInputs):
    """Omitted Form inputs retain values; null clears them; overrides apply last."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    participants: list[StrictInt] | None | Keep = Field(
        default=KEEP,
        description="Sets Participants (zero-based tone positions in Run order); null requires an override before Run.",
    )

    fit_method: Literal["fringe", "decay"] | None | Keep = Field(
        default=KEEP,
        description="Sets Fit method (fringe or decay). Omit to keep it; Analyze rejects null.",
    )
    fit_phase: bool | None | Keep = Field(
        default=KEEP,
        description="Sets Fit phase offset (fringe only). Omit to keep it; Analyze rejects null after Run.",
    )


class MuxT2EchoToneReport(MuxFitToneReport):
    """One captured tone's estimates and one-sigma errors, or its fit failure.

    Nullable estimates retain absent or nonfinite native summary values as null.
    Identity uses the original tone position, never the participant row offset.
    """

    t2: float | None = Field(default=None, description="T2 echo (µs).")
    t2_err: float | None = Field(
        default=None, description="T2 echo standard error (µs)."
    )
    detune: float | None = Field(
        default=None, description="Fitted fringe rate (MHz); zero for decay fits."
    )
    detune_err: float | None = Field(
        default=None,
        description="Fitted fringe rate standard error (MHz); zero for decay fits.",
    )


class MuxT2EchoReport(RecipeReport):
    """Native ordered estimates/failures and the confirmed writeback outcome."""

    tones: list[MuxT2EchoToneReport] = Field(
        description="Native ordered fit estimates or failures with original tone identities."
    )


@recipe(
    "mux_t2echo",
    description="Run mux Echo, save raw and analysis, and ask before Apply.",
)
async def mux_t2echo(
    session: RecipeSession, inputs: MuxT2EchoInputs
) -> MuxT2EchoReport:
    """Measure mux Hahn Echo, save the results and review coherence writeback.

    Open captures Form sources; omitted inputs keep values and null clears them.
    Overrides run last. Only participant drive/sweep settings are required;
    readout identities remain required for every tone.
    Use overrides for Delay › Start/Stop (µs) and Delay › Expts (count); n sets Expts.
    Analyze with fit_method, fit_phase, then ask before writeback.
    Save raw and completed artifacts after Run, including expected fit failures.
    Operation errors propagate without rolling back earlier edits or saves.
    """
    tab = session.open_tab(
        "mux/t2echo", target={"mux": inputs.target}, reuse=inputs.reuse_tab_id
    )
    tab.set("participants", inputs.participants)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        fit = await tab.analyze(
            "primary",
            updates={
                "fit_method": inputs.fit_method,
                "fit_phase": inputs.fit_phase,
            },
            result=MuxCoherenceSummary,
        )
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
        for outcome in fit.result.outcomes:
            name = None if fit.result.names is None else fit.result.names[outcome.index]
            if isinstance(outcome, MuxSummaryFailure):
                tones.append(
                    MuxT2EchoToneReport(
                        index=outcome.index, name=name, failure=outcome.reason
                    )
                )
            else:
                tones.append(
                    MuxT2EchoToneReport(
                        index=outcome.index,
                        name=name,
                        fit_quality=outcome.quality,
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
