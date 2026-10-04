"""Frequency/flux survey with explicit calibration and writeback handoff."""

from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    MissingParameter,
    RecipeDefinition,
    RecipeGenerator,
    RecipeNeedsParameters,
    RecipeRun,
    RecipeSession,
)


def onetone_spectrum_over_flux(  # noqa: PLR0913 - Preserve typed tool keywords.
    session: RecipeSession,
    *,
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    center_mhz: float | None = None,
    span_mhz: float | None = None,
    gain: float | None = None,
    flux_device: str | None = None,
    flux_unit: str | None = None,
    freq_points: int | None = None,
    flux_points: int | None = None,
    flux_range: list[float] | None = None,
    reps: int | None = None,
    rounds: int | None = None,
) -> RecipeGenerator:
    """Save one frequency/flux survey and Primary images, then ask before writing.

    session is driver-supplied. reuse_tab_id resets an idle onetone/flux_dep tab.
    readout_ref selects a library; None retains the selection. center/span use MHz
    and default to resonator/linewidth calibration. flux_range has two distinct
    device-native endpoints, or None for the GUI's calibrated range. flux_device
    defaults to the observed flux device; flux_unit asserts units without conversion.
    FakeDevice requires explicit native units. Other None values retain GUI defaults.
    All missing frequency/device/range sources share one needs_parameters handoff.
    Invalid cfg/native failures throw at their author step. Cancelled yields return;
    finish_early saves usable data. Accepted answer triggers explicit current-draft
    writeback; skipped does not write.
    """
    for name, value in (
        ("reuse_tab_id", reuse_tab_id),
        ("readout_ref", readout_ref),
        ("flux_device", flux_device),
        ("flux_unit", flux_unit),
    ):
        if value is not None and not value.strip():
            raise ValueError(f"{name} must be a non-empty string or null")
    if span_mhz is not None and span_mhz <= 0:
        raise ValueError("span_mhz must be positive")
    tab = session.open_tab("onetone/flux_dep", reuse=reuse_tab_id)
    tab.use_library("modules.readout", readout_ref)
    # Report independent missing sources together before admitting a Run.
    missing: list[MissingParameter] = []
    try:
        tab.set_frequency_sweep(
            "sweep.freq",
            calibration="resonator",
            center_mhz=center_mhz,
            span_mhz=span_mhz,
            expts=freq_points,
        )
    except RecipeNeedsParameters as error:
        missing.extend(error.missing)
    try:
        tab.use_flux_device(flux_device, unit=flux_unit)
    except RecipeNeedsParameters as error:
        missing.extend(error.missing)
    try:
        tab.set_flux_sweep(
            "sweep.flux",
            start=None if flux_range is None else flux_range[0],
            stop=None if flux_range is None else flux_range[1],
            expts=flux_points,
        )
    except RecipeNeedsParameters as error:
        missing.extend(error.missing)
    if missing:
        raise RecipeNeedsParameters(tuple(missing))
    tab.set("modules.readout.pulse_cfg.gain", gain)
    tab.set("reps", reps)
    tab.set("rounds", rounds)
    run, status = yield tab.run()
    if status == "cancelled":
        return
    assert isinstance(run, RecipeRun)
    run.save_raw()
    _, status = yield run.analyze("primary")
    if status == "cancelled":
        return
    decision, status = yield run.propose_writeback()
    if status == "cancelled":
        return
    if decision == "accepted":
        tab.accept()


DEFINITION = RecipeDefinition(
    name="onetone_spectrum_over_flux",
    description="Run one frequency/flux survey, save raw and Primary analysis. Physical units must match; FakeDevice requires flux_unit=native.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
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
            "flux_device": {"type": ["string", "null"], "minLength": 1},
            "flux_unit": {
                "type": ["string", "null"],
                "minLength": 1,
                "description": "Optional device-unit assertion. FakeDevice requires explicit native coordinates.",
            },
            "freq_points": {"type": ["integer", "null"]},
            "flux_points": {"type": ["integer", "null"]},
            "flux_range": {
                "type": ["array", "null"],
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
    },
    run=onetone_spectrum_over_flux,
    adapter_name="onetone/flux_dep",
    summary_parameters=(
        SummaryParameter("center_mhz", "center_mhz", "MHz"),
        SummaryParameter("span_mhz", "span_mhz", "MHz"),
        SummaryParameter("frequency_sweep", "sweep.freq", "MHz"),
        SummaryParameter("gain", "modules.readout.pulse_cfg.gain"),
        SummaryParameter("readout_ref", "modules.readout"),
        SummaryParameter("reps", "reps"),
        SummaryParameter("rounds", "rounds"),
        SummaryParameter("flux_sweep", "sweep.flux"),
        SummaryParameter("flux_device", "dev.flux_dev"),
    ),
    summary_estimates=(),
)
