"""Measure mux Ramsey fringes, save the decay fits and review calibrations."""

from pydantic import Field, StrictInt
from zcu_tools.mcp.measure.recipe_api import (
    FieldEdit,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
    report_number,
)

from zcu_lab.v2._support.measure.mux_results import (
    MuxCoherenceSummary,
    MuxSummaryFailure,
)

from .mux_helpers import (
    MuxToneReport,
)


class MuxT2RamseyInputs(RecipeInputs):
    """Experiment inputs; null retains captured Form values, overrides apply last."""

    participants: list[StrictInt] | None = Field(
        default=None,
        description="Unique original tone indices in acquisition order; null keeps the captured Form selection.",
    )

    fit_fringe: bool | None = Field(
        None,
        description="Fit oscillations instead of a pure decay. Null retains analysis settings.",
    )
    fit_phase: bool | None = Field(
        None,
        description="Fit the phase offset when fitting fringes. Null retains analysis settings.",
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

    Select library then inline sources; null keeps captured Form values.
    Overrides run last. Only participant drive/sweep settings are required;
    readout identities remain required for every tone.
    Shared sweep n is set through overrides; indices are original tone positions.
    Analyze with fit_fringe, fit_phase, then ask before writeback.
    Save raw and completed artifacts after Run, including expected fit failures.
    Operation errors propagate without rolling back earlier edits or saves.
    """
    tab = session.open_recipe_tab("mux/t2ramsey", inputs, target_alias="mux")

    tab.select_sources(inputs)
    tab.apply_inputs(
        inputs,
        {},
        edits=[
            FieldEdit(
                path=("participants",),
                value=None if inputs.participants is None else [*inputs.participants],
            )
        ],
    )

    # Acquire before fitting; the finally block saves raw and artifacts.
    await tab.run()
    try:
        fit = await tab.analyze(
            "primary",
            updates={
                key: value
                for key, value in {
                    "fit_fringe": inputs.fit_fringe,
                    "fit_phase": inputs.fit_phase,
                }.items()
                if value is not None
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
        for outcome in fit.result.tones:
            name = outcome.name
            if isinstance(outcome, MuxSummaryFailure):
                tones.append(
                    MuxT2RamseyToneReport(
                        index=outcome.position, name=name, failure=outcome.failure
                    )
                )
            else:
                tones.append(
                    MuxT2RamseyToneReport(
                        index=outcome.position,
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
