"""Measure excited-state relaxation versus delay to estimate T1."""

from typing import Literal

from pydantic import BaseModel, Field
from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)

from .helpers import (
    FrequencyCalibration,
    apply_frequency,
    choose_frequency,
    report_number,
    require_pi,
    select_or_disable,
)


class T1Inputs(FormModel):
    """Inputs for t1; all units and null policies are declared per field."""

    target: str | None = Field(
        None,
        description="Qubit alias for this Reset; null uses the GUI aliases without changing focus.",
    )
    reuse_tab_id: str | None = Field(
        None,
        description="Idle matching tab to reset; null opens a new tab. May be combined with target.",
    )
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


class T1Report(FormModel):
    """Completed t1 results; cancellation returns no report."""

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
    writeback_applied: bool = Field(
        description="Whether the selected calibration proposal was explicitly accepted and applied."
    )


class T1Summary(BaseModel):
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


@recipe(
    "t1",
    description="Run one calibrated T1 delay sweep and save raw and Primary analysis.",
)
async def t1(session: RecipeSession, inputs: T1Inputs) -> T1Report:
    """Measure T1 from the decay of the qubit's excited-state population.

    After optional reset, apply a calibrated pi pulse, wait the swept delay, and
    read out the qubit. Fit the decay to estimate the relaxation
    time, and ask before applying the proposed T1; skipping writes nothing.

    Inputs declare native units and GUI-retention policies. The session supplies
    fixed Form sources. Return the typed report after completion.
    The runtime owns cancellation. Owner, acquisition, save and analysis errors
    propagate. Save raw data and completed artifacts once in the Run finalizer.
    Missing calibration raises RecipeNeedsParameters before Run.
    """
    # Select the captured sources before editing measurement fields.
    tab = session.open_tab(
        "twotone/t1",
        aliases=None if inputs.target is None else {"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    select_or_disable(tab, ("modules", "reset"), inputs.use_reset)
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    missing: list[MissingParameter] = []
    frequency = choose_frequency(
        tab,
        ("modules", "readout", "freq"),
        session,
        calibration=FrequencyCalibration(
            path="$resonator.freq", required="readout_ref"
        ),
    )
    missing.extend(frequency.missing)
    if frequency.edit is not None:
        tab.select(("modules", "readout"), inline=True)
        apply_frequency(tab, frequency.edit)
    if (
        parameter := require_pi(
            tab, ("modules", "pi_pulse"), inputs.pi_ref, required="pi_ref"
        )
    ) is not None:
        missing.append(parameter)
    if missing:
        raise RecipeNeedsParameters(tuple(missing))
    tab.set(("sweep", "length", "stop"), inputs.max_delay_us)
    tab.set(("sweep", "length", "n"), inputs.points)
    tab.set(("reps",), inputs.reps)
    tab.set(("rounds",), inputs.rounds)

    # Acquire before analysis; the finally block saves raw and artifacts.
    await tab.run()
    try:
        # Analyze the acquisition with the GUI parameters.
        analysis = await tab.analyze("primary", result=T1Summary)

        # Write only the proposal items selected in the answer.
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
