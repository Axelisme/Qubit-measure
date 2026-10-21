"""Resonator frequency/gain map with raw save and Run preview only."""

from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    RecipeDefinition,
    RecipeGenerator,
    RecipeNeedsParameters,
    RecipeSession,
)

from .helpers import (
    FrequencySweepCalibration,
    retain_scalar,
    set_frequency_sweep,
)


# tolerated: Public frequency/gain survey keywords must match the handwritten input_schema.
def onetone_spectrum_over_power(  # noqa: PLR0913
    session: RecipeSession,
    *,
    target: str | None = None,
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    center_mhz: float | None = None,
    span_mhz: float | None = None,
    freq_points: int | None = None,
    gain_points: int | None = None,
    gain_range: list[float] | None = None,
    reps: int | None = None,
    rounds: int | None = None,
) -> RecipeGenerator:
    """Map the resonator response versus readout frequency and gain.

    At each gain, sweep readout frequency and average IQ to show how the
    resonance changes with drive amplitude. Save the raw map and capture its
    Run preview only; this survey performs no analysis or writeback.

    session is driver-supplied. target overrides the qubit alias for this Reset;
    None uses global aliases. reuse_tab_id resets an idle onetone/power_dep tab.
    readout_ref selects a library module; None retains its GUI selection.
    center_mhz and positive span_mhz use MHz; omitted values use resonator
    frequency and three linewidths for the full span. gain_range has two native
    amplitude endpoints; freq_points/gain_points count each sweep axis.
    reps/rounds control averaging. Other None values retain GUI defaults.
    Missing calibration reports needs_parameters; invalid inputs/cfg/native/PNG
    failures propagate. A cancelled Run stops before save and preview.
    """
    # 1. Validate inputs before selecting the measurement sources.
    for name, value in (("reuse_tab_id", reuse_tab_id), ("readout_ref", readout_ref)):
        if value is not None and not value.strip():
            raise ValueError(f"{name} must be a non-empty string or null")
    if span_mhz is not None and span_mhz <= 0:
        raise ValueError("span_mhz must be positive")

    # 2. Open/reset the tab and select the pulse libraries.
    tab = session.open_tab(
        "onetone/power_dep",
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
        expts=freq_points,
    )
    if missing:
        raise RecipeNeedsParameters(missing)
    tab.set_sweep(
        ("sweep", "gain"),
        start=None if gain_range is None else gain_range[0],
        stop=None if gain_range is None else gain_range[1],
        expts=gain_points,
    )
    retain_scalar(tab, ("modules", "readout", "gain"))
    tab.set(("reps",), reps)
    tab.set(("rounds",), rounds)

    # 4. Acquire and save the raw IQ sweep.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()

    # 5. Capture the map preview without fitting or changing calibration.
    run.preview()


DEFINITION = RecipeDefinition(
    name="onetone_spectrum_over_power",
    description="Run one frequency/gain survey, save raw and return its Run preview. No analysis or writeback.",
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
            "freq_points": {"type": ["integer", "null"]},
            "gain_points": {"type": ["integer", "null"]},
            "gain_range": {
                "type": ["array", "null"],
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
    },
    run=onetone_spectrum_over_power,
    adapter_name="onetone/power_dep",
    summary_parameters=(
        SummaryParameter("frequency_sweep", ("sweep", "freq"), "MHz"),
        SummaryParameter("gain", ("modules", "readout", "gain")),
        SummaryParameter("readout_ref", ("modules", "readout")),
        SummaryParameter("reps", ("reps",)),
        SummaryParameter("rounds", ("rounds",)),
        SummaryParameter("gain_sweep", ("sweep", "gain")),
    ),
    summary_estimates=(),
)
