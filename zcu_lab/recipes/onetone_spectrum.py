"""Resonator frequency sweep, spectral fit and reviewed calibration."""

from zcu_tools.mcp.measure.execution_reply import SummaryEstimate, SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    RecipeDefinition,
    RecipeGenerator,
    RecipeNeedsParameters,
    RecipeSession,
    ask,
)

from .helpers import (
    FrequencySweepCalibration,
    retain_scalar,
    set_frequency_sweep,
)


def onetone_spectrum(  # noqa: PLR0913 - Typed keywords preserve the public tool inputs.
    session: RecipeSession,
    *,
    target: str | None = None,
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    center_mhz: float | None = None,
    span_mhz: float | None = None,
    gain: float | None = None,
    points: int | None = None,
    reps: int | None = None,
    rounds: int | None = None,
) -> RecipeGenerator:
    """Sweep readout frequency to locate the resonator response.

    At each frequency, apply the readout pulse and average its IQ signal.
    Primary fits the spectrum to estimate resonator frequency and linewidth.
    Save raw data and analysis, then ask before applying the calibration.

    session is driver-supplied. target overrides the qubit alias for this Reset;
    None uses global aliases. reuse_tab_id resets a named idle onetone/freq tab.
    readout_ref selects a library module; None retains its GUI selection.
    center_mhz and positive span_mhz use MHz; omitted values use resonator
    frequency and three linewidths for the full span. gain is native amplitude,
    points is the sweep count, and reps/rounds control averaging.
    Other None values retain GUI defaults. Blank references or invalid ranges
    fail; missing calibration reports needs_parameters. Cfg/native failures
    propagate. Cancellation stops later steps; skipped writeback changes nothing.
    """
    # 1. Validate inputs before selecting the measurement sources.
    for name, value in (("reuse_tab_id", reuse_tab_id), ("readout_ref", readout_ref)):
        if value is not None and not value.strip():
            raise ValueError(f"{name} must be a non-empty string or null")
    if span_mhz is not None and span_mhz <= 0:
        raise ValueError("span_mhz must be positive")

    # 2. Open/reset the tab and select the pulse libraries.
    tab = session.open_tab(
        "onetone/freq",
        aliases=None if target is None else {"qubit": target},
        reuse=reuse_tab_id,
    )
    tab.use_library(("modules", "readout"), readout_ref)

    # 3. Set the scan range and averaging for the resonator response.
    missing = set_frequency_sweep(
        tab,
        ("sweep", "freq"),
        calibration=FrequencySweepCalibration(
            "$resonator.freq", "$resonator.kappa", 3.0
        ),
        center_mhz=center_mhz,
        span_mhz=span_mhz,
        expts=points,
    )
    if missing:
        raise RecipeNeedsParameters(missing)
    if gain is not None:
        tab.use_inline(("modules", "readout"))
        tab.set(("modules", "readout", "gain"), gain)
    else:
        retain_scalar(tab, ("modules", "readout", "gain"))
    tab.set(("reps",), reps)
    tab.set(("rounds",), rounds)

    # 4. Acquire and save the raw IQ sweep.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()

    # 5. Fit the resonator frequency and linewidth.
    analysis = yield from run.analyze_step("primary")
    if analysis.status == "cancelled":
        return

    # 6. Ask before applying the proposed calibration.
    decision = yield from ask(run.propose_writeback())
    if decision.status == "cancelled":
        return
    if decision.decision == "accepted":
        tab.accept()


DEFINITION = RecipeDefinition(
    name="onetone_spectrum",
    description="Run one onetone spectrum and save raw data and Primary analysis.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target": {"type": ["string", "null"], "minLength": 1},
            **{
                name: {"type": ["string", "null"], "minLength": 1}
                for name in ("reuse_tab_id", "readout_ref")
            },
            **{
                name: {"type": ["number", "null"]}
                for name in ("center_mhz", "span_mhz")
            },
            **{name: {"type": ["integer", "null"]} for name in ("reps", "rounds")},
            "gain": {"type": ["number", "null"]},
            "points": {"type": ["integer", "null"]},
        },
    },
    run=onetone_spectrum,
    adapter_name="onetone/freq",
    summary_parameters=(
        SummaryParameter("frequency_sweep", ("sweep", "freq"), "MHz"),
        SummaryParameter("gain", ("modules", "readout", "gain")),
        SummaryParameter("readout_ref", ("modules", "readout")),
        SummaryParameter("reps", ("reps",)),
        SummaryParameter("rounds", ("rounds",)),
    ),
    summary_estimates=(
        SummaryEstimate("freq", ("freq",), unit="MHz"),
        SummaryEstimate("fwhm", ("fwhm",), unit="MHz"),
    ),
)
