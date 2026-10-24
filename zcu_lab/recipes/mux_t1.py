"""Measure simultaneous T1 decays and ask before writing calibrations."""

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

from zcu_lab.v2.mux.t1.gui import MuxT1Summary
from zcu_lab.v2.support.measure.mux_results import MuxSummaryFailure

from .mux_helpers import (
    MuxToneReport,
)


class MuxT1Inputs(RecipeInputs):
    """Omitted Form inputs retain values; null clears them; overrides apply last."""

    participants: list[StrictInt] | None | Keep = Field(
        default=KEEP,
        description="Unique original tone indices in acquisition order; omission keeps the Form selection; null clears it.",
    )

    dual_exp: bool | None | Keep = Field(
        default=KEEP,
        description="Fit two decay components instead of one. Omission keeps analysis settings; null is passed to the analysis owner.",
    )
    skip: int | None | Keep = Field(
        default=KEEP,
        description="Points to skip before fitting the decay. Omission keeps analysis settings; null is passed to the analysis owner.",
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


class MuxT1Report(RecipeReport):
    """Native ordered estimates/failures and the confirmed writeback outcome."""

    tones: list[MuxT1ToneReport] = Field(
        description="Native ordered fit estimates or failures with original tone identities."
    )


@recipe(
    "mux_t1", description="Run mux T1, save raw and analysis, and ask before Apply."
)
async def mux_t1(session: RecipeSession, inputs: MuxT1Inputs) -> MuxT1Report:
    """Measure simultaneous T1 decays and ask before writing calibrations.

    Open captures Form sources; omitted inputs keep values and null clears them.
    Overrides run last. Only participant drive/sweep settings are required;
    readout identities remain required for every tone.
    Shared sweep n is set through overrides; indices are original tone positions.
    Analyze with dual_exp, skip, then ask before writeback.
    Save raw and completed artifacts after Run, including expected fit failures.
    Operation errors propagate without rolling back earlier edits or saves.
    """
    tab = session.open_tab(
        "mux/t1", target={"mux": inputs.target}, reuse=inputs.reuse_tab_id
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
                "dual_exp": inputs.dual_exp,
                "skip": inputs.skip,
            },
            result=MuxT1Summary,
        )
        if fit.result is None:
            return MuxT1Report(
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

        tones: list[MuxT1ToneReport] = []
        for outcome in fit.result.outcomes:
            name = None if fit.result.names is None else fit.result.names[outcome.index]
            if isinstance(outcome, MuxSummaryFailure):
                tones.append(
                    MuxT1ToneReport(
                        index=outcome.index, name=name, failure=outcome.reason
                    )
                )
            else:
                tones.append(
                    MuxT1ToneReport(
                        index=outcome.index,
                        name=name,
                        t1=report_number(outcome.fit.t1),
                        t1_err=report_number(outcome.fit.t1_err),
                        t1b=report_number(outcome.fit.t1b),
                        t1b_err=report_number(outcome.fit.t1b_err),
                    )
                )
        return MuxT1Report(tones=tones, writeback_applied=writeback_applied)
    finally:
        tab.save()


DEFINITION = mux_t1
