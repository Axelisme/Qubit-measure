"""Measure CPMG decay across positive refocusing-pulse counts.

Recipe points controls the pulse-count grid, not a time-sweep expts field.
length_points → length_expts: the GUI Time points input is the shared time-window count.
"""

from math import isfinite
from typing import Literal

from pydantic import Field
from zcu_tools.gui.form_adapter.log_points import (
    PositiveExplicitIntegerPoints,
    PositiveLogIntegerPoints,
)
from zcu_tools.mcp.measure.recipe_api import (
    FieldEdit,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)

from zcu_lab.v2.twotone.time_domain.cpmg.core import CPMG_SweepCfg
from zcu_lab.v2.twotone.time_domain.cpmg.gui import CPMGSummary

from .helpers import (
    FrequencyCalibration,
    choose_frequency,
    require_pi,
    select_or_disable,
)


def _positive_integer(name: str, value: int | None, minimum: int = 1) -> None:
    if value is not None and (type(value) is not int or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum} or null")


def _prepare_counts(
    max_n: int | None, points: int | None, times: list[int] | None
) -> list[int] | None:
    _positive_integer("max_n", max_n)
    _positive_integer("points", points)
    if (max_n is None) != (points is None):
        raise ValueError("max_n and points must be supplied together")
    if times is not None and max_n is not None:
        raise ValueError("Use max_n/points or times, not both")
    if times is not None:
        return CPMG_SweepCfg.model_validate({"times": times}).times
    if max_n is not None and points is not None:
        return PositiveLogIntegerPoints(max=max_n, points=points).values
    return None


def _validate_time_window(
    minimum: float | None, maximum: float | None, points: int | None
) -> None:
    for name, value in (("min_length_us", minimum), ("max_length_us", maximum)):
        if value is not None and (not isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive and finite or null")
    if minimum is not None and maximum is not None and (maximum <= minimum):
        raise ValueError("max_length_us must exceed min_length_us")
    _positive_integer("length_points", points, minimum=2)


class CPMGInputs(RecipeInputs):
    """Inputs for cpmg; all units and null policies are declared per field."""

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
    max_n: int | None = Field(
        None,
        description="Maximum positive refocusing count for a geometric count grid; supply together with points, or null for GUI counts.",
    )
    points: int | None = Field(
        None,
        description="Positive geometric-count sample request; supply with max_n. Rounding and deduplication may reduce the count; max_n=1 permits one point.",
    )
    times: list[int] | None = Field(
        None,
        description="Explicit positive refocusing counts; mutually exclusive with max_n/points. Null keeps the GUI counts.",
    )
    min_length_us: float | None = Field(
        None,
        description="Shared minimum total evolution time in us; null keeps the GUI value.",
    )
    max_length_us: float | None = Field(
        None,
        description="Shared maximum total evolution time in us; null keeps the GUI value.",
    )
    length_points: int | None = Field(
        None,
        description="Shared total-time sample count, at least two; null keeps the GUI value.",
    )
    detune_ratio: float | None = Field(
        None,
        description="Final-pulse phase-ramp ratio, dimensionless; null keeps the GUI value.",
    )
    reps: int | None = Field(
        None, description="Acquisition repetition count; null keeps the GUI value."
    )
    rounds: int | None = Field(
        None, description="Averaging round count; null keeps the GUI value."
    )

    fit_fringe: bool | None = Field(
        None,
        description="Fit fringes; null keeps the GUI analysis value. Analysis only.",
    )
    t2r: float | None = Field(
        None,
        description="Ramsey T2 reference (us); null keeps the GUI analysis value. Analysis only.",
    )
    t1: float | None = Field(
        None,
        description="T1 reference (us); null keeps the GUI analysis value. Analysis only.",
    )


class CPMGReport(RecipeReport):
    """CPMG results and failure reason; cancellation returns no report."""

    counts: list[int] = Field(
        description="Measured refocusing counts, or requested counts after analysis failure; empty when GUI counts are unavailable."
    )
    t2_us: list[float | None] = Field(
        description="Fitted coherence times in us; rejected rows are null."
    )
    t2_err_us: list[float | None] = Field(
        description="Coherence uncertainties in us; rejected rows are null."
    )


@recipe(
    "cpmg", description="Measure CPMG coherence over positive pi-pulse counts and time."
)
async def cpmg(session: RecipeSession, inputs: CPMGInputs) -> CPMGReport:
    """Measure coherence versus CPMG pulse count and total evolution time.

    Apply pi/2, n refocusing pi pulses, a final pi/2 projection and readout.
    Fit T_CPMG for each count, then save raw IQ and completed artifacts.
    No calibration is written.

    points sets the pulse-count grid; length_points → length_expts sets the shared time count.
    Analysis controls: fit_fringe, t2r, t1. Null keeps each GUI value.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    for name, value in (
        ("reuse_tab_id", inputs.reuse_tab_id),
        ("readout_ref", inputs.readout_ref),
        ("use_reset", inputs.use_reset),
        ("pi_ref", inputs.pi_ref),
        ("pi2_ref", inputs.pi2_ref),
    ):
        if value is not None and (not value.strip()):
            raise ValueError(f"{name} must be non-empty or null")
    counts = _prepare_counts(inputs.max_n, inputs.points, inputs.times)
    _validate_time_window(
        inputs.min_length_us, inputs.max_length_us, inputs.length_points
    )
    _positive_integer("reps", inputs.reps)
    _positive_integer("rounds", inputs.rounds)
    if inputs.detune_ratio is not None and (not isfinite(inputs.detune_ratio)):
        raise ValueError("detune_ratio must be finite or null")
    tab = session.open_recipe_tab("twotone/cpmg", inputs, target_alias="qubit")
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
    edits.append(
        FieldEdit(
            path=("sweep", "times"),
            value=None
            if counts is None
            else PositiveExplicitIntegerPoints(values=counts),
        )
    )
    edits.extend(
        (
            FieldEdit(path=("extras", "min_length_us"), value=inputs.min_length_us),
            FieldEdit(path=("extras", "max_length_us"), value=inputs.max_length_us),
            FieldEdit(path=("extras", "length_points"), value=inputs.length_points),
        )
    )
    tab.apply_inputs(
        inputs,
        {
            ("detune_ratio",): inputs.detune_ratio,
            ("reps",): inputs.reps,
            ("rounds",): inputs.rounds,
        },
        edits=edits,
    )

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=CPMGSummary,
            updates={
                key: value
                for key, value in {
                    "fit_fringe": inputs.fit_fringe,
                    "t2r": inputs.t2r,
                    "t1": inputs.t1,
                }.items()
                if value is not None
            },
        )
        if analysis.result is None:
            failure_counts = [] if counts is None else counts
            return CPMGReport(
                counts=failure_counts,
                t2_us=[None] * len(failure_counts),
                t2_err_us=[None] * len(failure_counts),
                analysis_failure=analysis.reason,
            )
        return CPMGReport(
            counts=analysis.result.ns,
            t2_us=analysis.result.t2s,
            t2_err_us=analysis.result.t2errs,
        )
    finally:
        tab.save()


DEFINITION = cpmg
