"""Resonator frequency/flux map and interactive flux calibration."""

from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    MissingParameter,
    RecipeDefinition,
    RecipeGenerator,
    RecipeNeedsParameters,
    RecipeSession,
    ask,
)

from .helpers import (
    FrequencySweepCalibration,
    retain_scalar,
    set_flux_sweep,
    set_frequency_sweep,
    use_flux_device,
)


# tolerated: Public frequency/flux survey keywords must match the handwritten input_schema.
def onetone_spectrum_over_flux(  # noqa: PLR0913
    session: RecipeSession,
    *,
    target: str | None = None,
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    center_mhz: float | None = None,
    span_mhz: float | None = None,
    gain: float | None = None,
    flux_device: str,
    flux_unit: str,
    freq_points: int | None = None,
    flux_points: int,
    flux_range: list[float],
    reps: int | None = None,
    rounds: int | None = None,
) -> RecipeGenerator:
    """Map the resonator spectrum across a native flux range.

    At each flux value, sweep the readout frequency and average IQ.
    Primary lets the user pick half- and integer-flux lines to estimate flux
    calibration. Save the raw map and analysis, then ask before applying it.

    session is driver-supplied. target overrides the qubit alias for this Reset;
    None uses global aliases. reuse_tab_id resets an idle onetone/flux_dep tab.
    readout_ref selects a library module; None retains its GUI selection.
    center_mhz and positive span_mhz use MHz; omitted values use resonator
    frequency and two linewidths for the full span. gain is native amplitude;
    freq_points counts frequencies and reps/rounds control averaging.
    Required flux_range has two distinct device-native endpoints; flux_points
    must be >=2. Required flux_device names a registered device; flux_unit must
    exactly match its cached unit, including native. No coordinates are converted
    or devices connected. Other None values retain GUI defaults.
    Missing frequency/linewidth reports needs_parameters; invalid device/unit
    fails before Run. Invalid inputs/cfg/native failures propagate.
    Cancellation stops later steps; skipped writeback changes nothing.
    """
    # 1. Validate inputs before selecting the measurement sources.
    if (
        not flux_device
        or not flux_device.strip()
        or not flux_unit
        or not flux_unit.strip()
    ):
        raise ValueError("flux_device and flux_unit must be non-empty strings")
    for name, value in (
        ("reuse_tab_id", reuse_tab_id),
        ("readout_ref", readout_ref),
    ):
        if value is not None and not value.strip():
            raise ValueError(f"{name} must be a non-empty string or null")
    if span_mhz is not None and span_mhz <= 0:
        raise ValueError("span_mhz must be positive")

    # 2. Open/reset the tab and select the pulse libraries.
    tab = session.open_tab(
        "onetone/flux_dep",
        aliases=None if target is None else {"qubit": target},
        reuse=reuse_tab_id,
    )
    tab.use_library(("modules", "readout"), readout_ref)

    # 3. Set frequency and native flux axes before acquisition.
    # Report independent missing sources together before Run.
    missing: list[MissingParameter] = []
    missing.extend(
        set_frequency_sweep(
            tab,
            ("sweep", "freq"),
            calibration=FrequencySweepCalibration(
                "$resonator.freq", "$resonator.kappa", 2.0
            ),
            center_mhz=center_mhz,
            span_mhz=span_mhz,
            expts=freq_points,
        )
    )
    use_flux_device(tab, flux_device, unit=flux_unit)
    set_flux_sweep(
        tab,
        ("sweep", "flux"),
        start=flux_range[0],
        stop=flux_range[1],
        expts=flux_points,
    )
    if missing:
        raise RecipeNeedsParameters(tuple(missing))
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

    # 5. Let the user select half- and integer-flux lines.
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
    name="onetone_spectrum_over_flux",
    description="Run one frequency/flux survey with explicit native flux range/count. Save raw and Primary analysis. flux_unit must match the cached device snapshot.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["flux_device", "flux_unit", "flux_range", "flux_points"],
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
            "flux_device": {"type": "string", "minLength": 1},
            "flux_unit": {
                "type": "string",
                "minLength": 1,
                "description": "Required exact cached-snapshot unit assertion without conversion, including explicit native coordinates.",
            },
            "freq_points": {"type": ["integer", "null"]},
            "flux_points": {"type": "integer"},
            "flux_range": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
    },
    run=onetone_spectrum_over_flux,
    adapter_name="onetone/flux_dep",
    summary_parameters=(
        SummaryParameter("frequency_sweep", ("sweep", "freq"), "MHz"),
        SummaryParameter("gain", ("modules", "readout", "gain")),
        SummaryParameter("readout_ref", ("modules", "readout")),
        SummaryParameter("reps", ("reps",)),
        SummaryParameter("rounds", ("rounds",)),
        SummaryParameter("flux_sweep", ("sweep", "flux")),
        SummaryParameter("flux_device", ("flux_dev",)),
        SummaryParameter("flux_unit", ("flux_unit",)),
    ),
    summary_estimates=(),
)
