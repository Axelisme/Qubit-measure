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

from .helpers import (
    FrequencySweepCalibration,
    set_flux_sweep,
    set_frequency_sweep,
    use_flux_device,
)


def onetone_spectrum_over_flux(  # noqa: PLR0913 - Preserve typed tool keywords.
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
    """Save one frequency/flux survey and Primary images, then ask before writing.

    target binds the new tab's primary role without changing GUI focus.
    Omission uses focus. Do not pass target when reusing an existing tab.

    session is driver-supplied. reuse_tab_id resets an idle onetone/flux_dep tab.
    readout_ref selects a library; None retains the selection. center/span use MHz
    and default to resonator/linewidth calibration. flux_range has two distinct
    device-native endpoints and is required along with flux_points >=2.
    flux_device is a required registered device name. flux_unit must match its
    cached snapshot unit exactly, including explicitly requested native. These
    inputs reject null; no coordinates are converted and no device is connected.
    Other None values retain GUI defaults. Missing frequency/linewidth sources
    share one needs_parameters handoff; invalid device/unit fails before Run.
    Invalid cfg/native failures throw at their author step. Cancelled yields return;
    finish_early saves usable data. Accepted answer triggers explicit current-draft
    writeback; skipped does not write.
    """
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
    tab = session.open_tab(
        "onetone/flux_dep",
        aliases=None if target is None else {"qubit": target},
        reuse=reuse_tab_id,
    )
    tab.use_library("modules.readout", readout_ref)
    # Report independent missing sources together before admitting a Run.
    missing: list[MissingParameter] = []
    missing.extend(
        set_frequency_sweep(
            tab,
            "sweep.freq",
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
        "sweep.flux",
        start=flux_range[0],
        stop=flux_range[1],
        expts=flux_points,
    )
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
        SummaryParameter("frequency_sweep", "sweep.freq", "MHz"),
        SummaryParameter("gain", "modules.readout.pulse_cfg.gain"),
        SummaryParameter("readout_ref", "modules.readout"),
        SummaryParameter("reps", "reps"),
        SummaryParameter("rounds", "rounds"),
        SummaryParameter("flux_sweep", "sweep.flux"),
        SummaryParameter("flux_device", "dev.flux_dev"),
        SummaryParameter("flux_unit", "flux_unit"),
    ),
    summary_estimates=(),
)
