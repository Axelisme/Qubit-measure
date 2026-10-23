"""Measure spin-echo decay versus total delay to estimate refocused coherence.

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

from zcu_lab.v2.twotone.time_domain.t2echo.gui import T2EchoSummary

from .helpers import (
    FrequencyCalibration,
    choose_frequency,
    require_pi,
    select_or_disable,
)


class T2EchoInputs(RecipeInputs):
    """Inputs for t2echo; all units and null policies are declared per field."""

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
    pi2_ref: str | None = Field(
        None,
        description="Calibrated pi/2-pulse library key; null keeps the GUI selection, which must be a valid reference.",
    )
    max_delay_us: float | None = Field(
        None,
        description="Total echo-delay stop in us, not half-delay; null keeps the GUI sweep.",
    )
    detune_ratio: float | None = Field(
        None,
        description="Final-pulse phase-ramp ratio, dimensionless; null keeps the GUI value.",
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

    fit_method: Literal["fringe", "decay"] | None = Field(
        None,
        description="Fit method; null keeps the GUI analysis value. Analysis only.",
    )
    fit_phase: bool | None = Field(
        None,
        description="Fit phase offset (fringe only); null keeps the GUI analysis value. Analysis only.",
    )


class T2EchoReport(RecipeReport):
    """T2 Echo results and failure reason; cancellation returns no report."""

    t2echo_us: float | None = Field(
        description="Refocused coherence time in us; null means not estimable."
    )
    t2echo_err_us: float | None = Field(
        description="Refocused coherence uncertainty in us; null means not estimable."
    )


@recipe("t2echo", description="Run one calibrated Echo total-delay sweep.")
async def t2echo(session: RecipeSession, inputs: T2EchoInputs) -> T2EchoReport:
    """Measure refocused qubit coherence with a spin-echo delay sweep.

    After optional reset, apply pi/2, wait half the total delay, apply pi, wait
    the other half, then apply a phase-shifted pi/2 pulse and read out.
    Fit the decay to estimate echo T2, with optional fringe fitting.
    Ask before applying the proposed calibration, then save raw IQ and analysis.
    Skipping writes nothing.

    points → expts sets the sweep sample count.
    Analysis controls: fit_method, fit_phase. Null keeps each GUI value.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_recipe_tab("twotone/t2echo", inputs, target_alias="qubit")
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
    if (
        parameter := require_pi(
            tab, ("modules", "pi2_pulse"), inputs.pi2_ref, required="pi2_ref"
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
            ("extras", "detune_ratio"): inputs.detune_ratio,
        },
        edits=edits,
    )
    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=T2EchoSummary,
            updates={
                key: value
                for key, value in {
                    "fit_method": inputs.fit_method,
                    "fit_phase": inputs.fit_phase,
                }.items()
                if value is not None
            },
        )
        if analysis.result is None:
            return T2EchoReport(
                t2echo_us=None,
                t2echo_err_us=None,
                writeback_applied=False,
                analysis_failure=analysis.reason,
            )
        decision = await tab.ask_writeback()
        writeback_applied = False
        if decision.accepted:
            receipt = tab.accept_writeback(decision.items)
            receipt.raise_for_error()
            writeback_applied = bool(receipt.items)
        return T2EchoReport(
            t2echo_us=report_number(analysis.result.t2e),
            t2echo_err_us=report_number(analysis.result.t2e_err),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = t2echo
