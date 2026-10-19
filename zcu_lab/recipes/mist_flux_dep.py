"""One native flux/gain MIST survey, separate from the adaptive workflow."""

from math import isfinite

from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    RecipeDefinition,
    RecipeGenerator,
    RecipeSession,
)

from .helpers import set_flux_sweep, use_flux_device


def mist_flux_dep(
    session: RecipeSession,
    *,
    flux_device: str,
    flux_unit: str,
    flux_range: list[float],
    gain_range: list[float],
    points: int = 101,
) -> RecipeGenerator:
    """Acquire and save one MIST map using the current global aliases.

    session is driver-supplied. flux_device names a registered device; flux_unit
    must exactly match its cached unit (A/V/native), without conversion.
    flux_range and gain_range each contain two finite distinct endpoints;
    gain endpoints lie in [-1, 1]. points is an integer >=2 for each axis.
    The GUI's optional preparation, probe, readout and averaging defaults remain
    editable before Run. This recipe does not reuse a tab or change aliases.

    Invalid ranges/count raise ValueError before Open; cached-device/cfg errors
    propagate before Run. Cancellation skips subsequent steps. Save raw IQ then
    the default raw-coordinate Primary map; no writeback is proposed.
    """
    # 1. Reject malformed ranges before opening the map.
    for name, values in (("flux_range", flux_range), ("gain_range", gain_range)):
        if (
            len(values) != 2
            or any(isinstance(value, bool) or not isfinite(value) for value in values)
            or values[0] == values[1]
        ):
            raise ValueError(f"{name} requires two finite distinct endpoints")
    if any(abs(value) > 1 for value in gain_range):
        raise ValueError("gain_range endpoints must lie in [-1, 1]")
    if isinstance(points, bool) or points < 2:
        raise ValueError("points must be an integer >=2")

    # 2. Keep the captured pulse selections and set explicit native axes.
    tab = session.open_tab("mist/flux_dep")
    use_flux_device(tab, flux_device, unit=flux_unit)
    set_flux_sweep(
        tab,
        ("sweep", "flux"),
        start=flux_range[0],
        stop=flux_range[1],
        expts=points,
    )
    tab.set_sweep(
        ("sweep", "gain"),
        start=gain_range[0],
        stop=gain_range[1],
        expts=points,
        sources={"start": "explicit", "stop": "explicit", "expts": "explicit"},
    )

    # 3. Preserve raw response before producing the baseline-difference map.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()
    analysis = yield from run.analyze_step("primary")
    if analysis.status == "cancelled":
        return


DEFINITION = RecipeDefinition(
    name="mist_flux_dep",
    description="Acquire one MIST flux/gain map with explicit native ranges; save raw and Primary without writeback.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["flux_device", "flux_unit", "flux_range", "gain_range"],
        "properties": {
            "flux_device": {"type": "string", "minLength": 1},
            "flux_unit": {
                "type": "string",
                "description": "Exact cached device unit: A, V or native.",
            },
            **{
                name: {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                }
                for name in ("flux_range", "gain_range")
            },
            "points": {
                "type": "integer",
                "description": "At least 2 points per axis; defaults to 101.",
            },
        },
    },
    run=mist_flux_dep,
    adapter_name="mist/flux_dep",
    summary_parameters=(
        SummaryParameter("flux_device", ("flux_dev",)),
        SummaryParameter("flux_unit", ("flux_unit",)),
        SummaryParameter("flux_sweep", ("sweep", "flux")),
        SummaryParameter("gain_sweep", ("sweep", "gain")),
        SummaryParameter("probe", ("modules", "probe_pulse")),
        SummaryParameter("readout", ("modules", "readout")),
        SummaryParameter("reps", ("reps",)),
        SummaryParameter("rounds", ("rounds",)),
    ),
    summary_estimates=(),
)
