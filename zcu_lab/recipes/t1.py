"""Measure excited-state relaxation versus delay to estimate T1.

Recipe points → expts: scan sample counts map to the GUI sweep Expts field.
"""

from typing import Literal

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    FieldEdit,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
    report_number,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)

from zcu_lab.v2.twotone.time_domain.t1.gui import T1Summary

from .helpers import (
    FrequencyCalibration,
    choose_frequency,
    require_pi,
    select_or_disable,
)


class T1Inputs(RecipeInputs):
    """Inputs for t1; all units and null policies are declared per field."""

    readout_ref: str | None = Field(
        None,
        description="Readout library key; null keeps the GUI selection. Frequency uses a valid captured library value, then resonator calibration.",
    )
    use_reset: str | Literal["off"] = Field(
        "off",
        description="Reset library key, or off to disable reset. Omission disables reset; null is rejected.",
    )
    pi_ref: str | None = Field(
        None,
        description="Calibrated pi-pulse library key; null keeps the GUI selection, which must be a valid reference.",
    )
    max_delay_us: float | None = Field(
        None, description="Delay sweep stop in us; null keeps the GUI stop and start."
    )
    points: int | None = Field(
        None, description="Sweep sample count; null keeps the GUI count when optional."
    )
    reps: int | None = Field(
        None, description="Acquisition repetition count; null keeps the GUI value."
    )
    rounds: int | None = Field(
        None, description="Averaging round count; null keeps the GUI value."
    )

    dual_exp: bool | None = Field(
        None,
        description="Dual exponential; null keeps the GUI analysis value. Analysis only.",
    )
    skip: int | None = Field(
        None,
        description="Skip leading points; null keeps the GUI analysis value. Analysis only.",
    )


class T1Report(RecipeReport):
    """T1 results and failure reason; cancellation returns no report."""

    t1_us: float | None = Field(
        description="Relaxation time in us; null means not estimable."
    )
    t1_err_us: float | None = Field(
        description="Relaxation uncertainty in us; null means not estimable."
    )
    t1b_us: float | None = Field(
        description="Second relaxation time in us; null without a second fit or when not estimable."
    )
    t1b_err_us: float | None = Field(
        description="Second relaxation uncertainty in us; null without a second fit or when not estimable."
    )


@recipe(
    "t1",
    description="Run one calibrated T1 delay sweep and save raw and Primary analysis.",
)
async def t1(session: RecipeSession, inputs: T1Inputs) -> T1Report:
    """Measure T1 from the decay of the qubit's excited-state population.

    After optional reset, apply a calibrated pi pulse, wait the swept delay, and
    read out the qubit. Fit the decay to estimate the relaxation
    time, and ask before applying the proposed T1; skipping writes nothing.

    points → expts sets the sweep sample count.
    Analysis controls: dual_exp, skip. Null keeps each GUI value.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_recipe_tab("twotone/t1", inputs, target_alias="qubit")
    select_or_disable(tab, ("modules", "reset"), inputs.use_reset)
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    tab.select_sources(inputs)

    edits: list[FieldEdit] = []
    missing: list[MissingParameter] = []
    frequency = choose_frequency(
        tab,
        ("modules", "readout", "freq"),
        calibration=FrequencyCalibration(
            path="$resonator.freq", required="readout_ref"
        ),
    )
    missing.extend(frequency.missing)
    if frequency.edit is not None:
        tab.select(("modules", "readout"), inline=True)
        edits.append(frequency.edit)
    if (
        parameter := require_pi(
            tab, ("modules", "pi_pulse"), inputs.pi_ref, required="pi_ref"
        )
    ) is not None:
        missing.append(parameter)
    if missing:
        raise RecipeNeedsParameters(tuple(missing))
    tab.apply_inputs(
        inputs,
        {
            ("sweep", "length", "stop"): inputs.max_delay_us,
            ("sweep", "length", "n"): inputs.points,
            ("reps",): inputs.reps,
            ("rounds",): inputs.rounds,
        },
        edits=edits,
    )

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=T1Summary,
            updates={
                key: value
                for key, value in {
                    "dual_exp": inputs.dual_exp,
                    "skip": inputs.skip,
                }.items()
                if value is not None
            },
        )
        if analysis.result is None:
            return T1Report(
                t1_us=None,
                t1_err_us=None,
                t1b_us=None,
                t1b_err_us=None,
                writeback_applied=False,
                analysis_failure=analysis.reason,
            )

        decision = await tab.ask_writeback()
        writeback_applied = False
        if decision.accepted:
            receipt = tab.accept_writeback(decision.items)
            receipt.raise_for_error()
            writeback_applied = bool(receipt.items)

        return T1Report(
            t1_us=report_number(analysis.result.t1),
            t1_err_us=report_number(analysis.result.t1_err),
            t1b_us=report_number(analysis.result.t1b),
            t1b_err_us=report_number(analysis.result.t1b_err),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = t1
