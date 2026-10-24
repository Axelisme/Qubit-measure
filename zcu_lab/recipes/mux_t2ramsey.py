"""Measure mux Ramsey fringes, save the decay fits and review calibrations."""

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

from zcu_lab.v2.support.measure.mux_results import (
    MuxCoherenceSummary,
    MuxSummaryFailure,
)

from .mux_helpers import (
    MuxToneReport,
)


class MuxT2RamseyInputs(RecipeInputs):
    """Omitted Form inputs retain values; null clears them; overrides apply last."""

    participants: list[StrictInt] | None | Keep = Field(
        default=KEEP,
        description="Unique original tone indices in acquisition order; omission keeps the Form selection; null clears it.",
    )

    fit_fringe: bool | None | Keep = Field(
        default=KEEP,
        description="Fit oscillations instead of a pure decay. Omission keeps analysis settings; null is passed to the analysis owner.",
    )
    fit_phase: bool | None | Keep = Field(
        default=KEEP,
        description="Fit the phase offset when fitting fringes. Omission keeps analysis settings; null is passed to the analysis owner.",
    )


class MuxT2RamseyToneReport(MuxToneReport):
    """One captured tone's estimates and one-sigma errors, or its fit failure.

    Nullable estimates retain absent or nonfinite native summary values as null.
    Identity uses the original tone position, never the participant row offset.
    """

    t2: float | None = Field(
        default=None, description="Fitted Ramsey coherence time T2* in us."
    )
    t2_err: float | None = Field(
        default=None, description="One-sigma Ramsey time uncertainty in us."
    )
    detune: float | None = Field(
        default=None, description="Fitted Ramsey fringe rate in MHz."
    )
    detune_err: float | None = Field(
        default=None, description="One-sigma fringe rate uncertainty in MHz."
    )


class MuxT2RamseyReport(RecipeReport):
    """Native ordered estimates/failures and the confirmed writeback outcome."""

    tones: list[MuxT2RamseyToneReport] = Field(
        description="Native ordered fit estimates or failures with original tone identities."
    )


@recipe(
    "mux_t2ramsey",
    description="Run mux Ramsey, save raw and analysis, and ask before Apply.",
)
async def mux_t2ramsey(
    session: RecipeSession, inputs: MuxT2RamseyInputs
) -> MuxT2RamseyReport:
    """Measure mux Ramsey fringes, save the decay fits and review calibrations.

    Open captures Form sources; omitted inputs keep values and null clears them.
    Overrides run last. Only participant drive/sweep settings are required;
    readout identities remain required for every tone.
    Shared sweep n is set through overrides; indices are original tone positions.
    Analyze with fit_fringe, fit_phase, then ask before writeback.
    Save raw and completed artifacts after Run, including expected fit failures.
    Operation errors propagate without rolling back earlier edits or saves.
    """
    tab = session.open_tab(
        "mux/t2ramsey", target={"mux": inputs.target}, reuse=inputs.reuse_tab_id
    )
    tab.set("participants", inputs.participants)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        fit = await tab.analyze(
            "primary",
            updates={
                "fit_fringe": inputs.fit_fringe,
                "fit_phase": inputs.fit_phase,
            },
            result=MuxCoherenceSummary,
        )
        if fit.result is None:
            return MuxT2RamseyReport(
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

        tones: list[MuxT2RamseyToneReport] = []
        for outcome in fit.result.outcomes:
            name = None if fit.result.names is None else fit.result.names[outcome.index]
            if isinstance(outcome, MuxSummaryFailure):
                tones.append(
                    MuxT2RamseyToneReport(
                        index=outcome.index, name=name, failure=outcome.reason
                    )
                )
            else:
                tones.append(
                    MuxT2RamseyToneReport(
                        index=outcome.index,
                        name=name,
                        t2=report_number(outcome.t2),
                        t2_err=report_number(outcome.t2_err),
                        detune=report_number(outcome.detune),
                        detune_err=report_number(outcome.detune_err),
                    )
                )
        return MuxT2RamseyReport(tones=tones, writeback_applied=writeback_applied)
    finally:
        tab.save()


DEFINITION = mux_t2ramsey
