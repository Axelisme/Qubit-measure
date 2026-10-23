"""Measure CPMG decay across positive refocusing-pulse counts.

Recipe points controls the pulse-count grid, not a time-sweep expts field.
length_points → length_expts: the GUI Time points input is the shared time-window count.
"""

from pydantic import Field
from zcu_tools.gui.form_adapter.log_points import (
    PositiveExplicitIntegerPoints,
    PositiveLogIntegerPoints,
)
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
)

from zcu_lab.v2.twotone.time_domain.cpmg.gui import CPMGSummary


class CPMGInputs(RecipeInputs):
    """Inputs for cpmg; all units and null policies are declared per field."""

    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Readout library key; omission retains the Form value; null clears/disables it. Sources are chosen by the Form.",
    )
    use_reset: str | None | Keep = Field(
        KEEP,
        description="Reset module option; omission retains the Form selection, null disables it.",
    )
    pi_ref: str | None | Keep = Field(
        KEEP,
        description="Calibrated pi-pulse library key; omission retains the Form value; null clears/disables it.",
    )
    pi2_ref: str | None | Keep = Field(
        KEEP,
        description="Calibrated pi/2-pulse library key; omission retains the Form value; null clears/disables it.",
    )
    max_n: int | None | Keep = Field(
        KEEP,
        description="Maximum positive refocusing count for a geometric count grid; supply together with points. Omission retains the count grid; null clears it.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Positive geometric-count sample request; supply with max_n. Rounding and deduplication may reduce the count; max_n=1 permits one point.",
    )
    times: list[int] | None | Keep = Field(
        KEEP,
        description="Explicit positive refocusing counts; mutually exclusive with max_n/points. Omission retains the Form counts; null clears them.",
    )
    min_length_us: float | None | Keep = Field(
        KEEP,
        description="Shared minimum total evolution time in us; omission retains the Form value; null clears/disables it.",
    )
    max_length_us: float | None | Keep = Field(
        KEEP,
        description="Shared maximum total evolution time in us; omission retains the Form value; null clears/disables it.",
    )
    length_points: int | None | Keep = Field(
        KEEP,
        description="Shared total-time sample count, at least two; omission retains the Form value; null clears/disables it.",
    )
    detune_ratio: float | None | Keep = Field(
        KEEP,
        description="Final-pulse phase-ramp ratio, dimensionless; omission retains the Form value; null clears/disables it.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Acquisition repetition count; omission retains the Form value; null clears/disables it.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Averaging round count; omission retains the Form value; null clears/disables it.",
    )

    fit_fringe: bool | None | Keep = Field(
        KEEP,
        description="Fit fringes; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    t2r: float | None | Keep = Field(
        KEEP,
        description="Ramsey T2 reference (us); omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    t1: float | None | Keep = Field(
        KEEP,
        description="T1 reference (us); omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
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
    Analysis controls: fit_fringe, t2r, t1. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "twotone/cpmg",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("modules.reset", inputs.use_reset)
    tab.select("modules.readout", inputs.readout_ref)
    tab.select("modules.pi_pulse", inputs.pi_ref)
    tab.select("modules.pi2_pulse", inputs.pi2_ref)
    counts: list[int] | None = None
    if not isinstance(inputs.times, Keep):
        if not isinstance(inputs.max_n, Keep) or not isinstance(inputs.points, Keep):
            raise ValueError("Use max_n/points or times, not both")
        if inputs.times is None:
            tab.set("sweep.times", None)
        else:
            counts = inputs.times
            tab.set("sweep.times", PositiveExplicitIntegerPoints(values=counts))
    elif not isinstance(inputs.max_n, Keep) or not isinstance(inputs.points, Keep):
        if isinstance(inputs.max_n, Keep) or isinstance(inputs.points, Keep):
            raise ValueError("max_n and points must be supplied together")
        if inputs.max_n is None or inputs.points is None:
            tab.set("sweep.times", None)
        else:
            count_grid = PositiveLogIntegerPoints(
                max=inputs.max_n, points=inputs.points
            )
            counts = count_grid.values
            tab.set("sweep.times", count_grid)
    tab.set("extras.min_length_us", inputs.min_length_us)
    tab.set("extras.max_length_us", inputs.max_length_us)
    tab.set("extras.length_points", inputs.length_points)
    tab.set("detune_ratio", inputs.detune_ratio)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

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
                if not isinstance(value, Keep)
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
