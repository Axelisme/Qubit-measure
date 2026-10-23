"""Measure Ramsey fringes versus delay to estimate free-evolution coherence.

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

from zcu_lab.v2.twotone.time_domain.t2ramsey.gui import T2RamseySummary

from .helpers import (
    FrequencyCalibration,
    choose_frequency,
    require_pi,
    select_or_disable,
)


class T2RamseyInputs(RecipeInputs):
    """Inputs for t2ramsey; all units and null policies are declared per field."""

    readout_ref: str | None = Field(
        None,
        description="Readout library key; null keeps the GUI selection. Frequency uses a valid captured library value, then resonator calibration.",
    )
    use_reset: str | Literal["off"] = Field(
        "off",
        description="Reset library key, or off to disable reset. Omission disables reset; null is rejected.",
    )
    pi2_ref: str | None = Field(
        None,
        description="Calibrated pi/2-pulse library key; null keeps the GUI selection, which must be a valid reference.",
    )
    max_delay_us: float | None = Field(
        None, description="Delay sweep stop in us; null keeps the GUI stop and start."
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

    fit_fringe: bool | None = Field(
        None,
        description="Fit fringe; null keeps the GUI analysis value. Analysis only.",
    )
    fit_phase: bool | None = Field(
        None,
        description="Fit phase offset (fringe only); null keeps the GUI analysis value. Analysis only.",
    )


class T2RamseyReport(RecipeReport):
    """T2 Ramsey results and failure reason; cancellation returns no report."""

    t2ramsey_us: float | None = Field(
        description="Free-evolution coherence time in us; null means not estimable."
    )
    t2ramsey_err_us: float | None = Field(
        description="Free-evolution coherence uncertainty in us; null means not estimable."
    )
    detune_mhz: float | None = Field(
        description="Fitted fringe frequency in MHz; zero for decay-only fitting; null means not estimable."
    )
    fit_fringe: bool = Field(
        description="Whether the fringe model was fitted; false when analysis failed."
    )


@recipe("t2ramsey", description="Run one calibrated Ramsey delay sweep.")
async def t2ramsey(session: RecipeSession, inputs: T2RamseyInputs) -> T2RamseyReport:
    """Measure Ramsey fringes and their decay during free qubit evolution.

    After optional reset, apply pi/2, wait the swept delay, then apply a
    phase-shifted pi/2 pulse and read out. Analyze before fitting the decay
    to estimate Ramsey T2; fringe fitting also estimates detuning. Ask before
    applying the proposed calibration; skipping writes nothing. Analysis keeps
    the GUI parameters, which fit a free fringe phase by default. Turn off Fit
    phase offset in the tab to use a fixed phase instead.

    points → expts sets the sweep sample count.
    Analysis controls: fit_fringe, fit_phase. Null keeps each GUI value.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_recipe_tab("twotone/t2ramsey", inputs, target_alias="qubit")
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
            result=T2RamseySummary,
            updates={
                key: value
                for key, value in {
                    "fit_fringe": inputs.fit_fringe,
                    "fit_phase": inputs.fit_phase,
                }.items()
                if value is not None
            },
        )
        if analysis.result is None:
            return T2RamseyReport(
                t2ramsey_us=None,
                t2ramsey_err_us=None,
                detune_mhz=None,
                fit_fringe=False,
                writeback_applied=False,
                analysis_failure=analysis.reason,
            )

        decision = await tab.ask_writeback()
        writeback_applied = False
        if decision.accepted:
            receipt = tab.accept_writeback(decision.items)
            receipt.raise_for_error()
            writeback_applied = bool(receipt.items)

        return T2RamseyReport(
            t2ramsey_us=report_number(analysis.result.t2r),
            t2ramsey_err_us=report_number(analysis.result.t2r_err),
            detune_mhz=report_number(analysis.result.detune),
            fit_fringe=analysis.result.fit_fringe,
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = t2ramsey
