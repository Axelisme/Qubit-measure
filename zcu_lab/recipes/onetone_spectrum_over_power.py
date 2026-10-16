"""Frequency/gain survey with raw save and original Run preview only."""

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


def onetone_spectrum_over_power(  # noqa: PLR0913 - Preserve typed tool keywords.
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
    """Run/save one power survey and capture its Run PNG, with no analysis/write.

    target binds the new tab's primary role without changing GUI focus.
    Omission uses focus. Do not pass target when reusing an existing tab.

    session is driver-supplied. reuse_tab_id resets an idle onetone/power_dep tab.
    readout_ref selects a library; None retains the selection. center/span use MHz
    and default to resonator/linewidth calibration. gain_range contains two native
    amplitude endpoints; None retains the GUI range. Counts/averages retain GUI
    defaults when None. Missing calibration ends needs_parameters; cfg/native/PNG
    failures throw at their author step. Cancelled Run returns; finish_early saves
    usable data. Later cancellation does not suppress the ordinary save/preview.
    """
    for name, value in (("reuse_tab_id", reuse_tab_id), ("readout_ref", readout_ref)):
        if value is not None and not value.strip():
            raise ValueError(f"{name} must be a non-empty string or null")
    if span_mhz is not None and span_mhz <= 0:
        raise ValueError("span_mhz must be positive")
    tab = session.open_tab(
        "onetone/power_dep",
        aliases=None if target is None else {"qubit": target},
        reuse=reuse_tab_id,
    )
    tab.use_library("modules.readout", readout_ref)
    missing = set_frequency_sweep(
        tab,
        "sweep.freq",
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
        "sweep.gain",
        start=None if gain_range is None else gain_range[0],
        stop=None if gain_range is None else gain_range[1],
        expts=gain_points,
    )
    retain_scalar(tab, "modules.readout.gain")
    tab.set("reps", reps)
    tab.set("rounds", rounds)
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()
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
        SummaryParameter("frequency_sweep", "sweep.freq", "MHz"),
        SummaryParameter("gain", "modules.readout.gain"),
        SummaryParameter("readout_ref", "modules.readout"),
        SummaryParameter("reps", "reps"),
        SummaryParameter("rounds", "rounds"),
        SummaryParameter("gain_sweep", "sweep.gain"),
    ),
    summary_estimates=(),
)
