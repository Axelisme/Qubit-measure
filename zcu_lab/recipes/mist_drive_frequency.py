"""MIST probe frequency/gain survey with native save and map analysis."""

from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    RecipeDefinition,
    RecipeGenerator,
    RecipeSession,
)


def mist_drive_frequency(
    session: RecipeSession,
    *,
    target: str | None = None,
    freq_range: list[float],
    gain_range: list[float],
    points: int = 51,
) -> RecipeGenerator:
    """Probe the requested MHz/gain rectangle and save the MIST response map.

    The driver supplies session. target overrides the qubit alias only for this
    Open; None keeps global aliases. freq_range and gain_range each contain two
    endpoints, in MHz and normalized amplitude respectively. points is the
    count on each axis (at least two). Modules and averaging retain GUI defaults.
    Invalid ranges/cfg and save/analysis failures propagate. Cancellation stops
    before save/analysis. This recipe makes no calibration writeback.
    """
    # 1. Reject malformed rectangles before opening a tab.
    if len(freq_range) != 2 or len(gain_range) != 2:
        raise ValueError("freq_range and gain_range must each contain two endpoints")
    if points < 2:
        raise ValueError("points must be at least two")

    # 2. Open with captured module sources and set the requested scan rectangle.
    tab = session.open_tab(
        "mist/power_dep/drive_freq",
        aliases=None if target is None else {"qubit": target},
    )
    tab.set_sweep(
        ("sweep", "freq"), start=freq_range[0], stop=freq_range[1], expts=points
    )
    tab.set_sweep(
        ("sweep", "gain"), start=gain_range[0], stop=gain_range[1], expts=points
    )

    # 3. Acquire and preserve the native complex IQ before plotting its response.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()
    yield from run.analyze_step("primary")
    run.preview()


DEFINITION = RecipeDefinition(
    name="mist_drive_frequency",
    description="Map MIST probe frequency/gain, save native IQ and analyze the response. No writeback.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["freq_range", "gain_range"],
        "properties": {
            "target": {"type": ["string", "null"], "minLength": 1},
            **{
                name: {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                }
                for name in ("freq_range", "gain_range")
            },
            "points": {"type": "integer"},
        },
    },
    run=mist_drive_frequency,
    adapter_name="mist/power_dep/drive_freq",
    summary_parameters=(
        SummaryParameter("frequency_sweep", ("sweep", "freq"), "MHz"),
        SummaryParameter("gain_sweep", ("sweep", "gain")),
        SummaryParameter("probe", ("modules", "probe_pulse")),
        SummaryParameter("readout", ("modules", "readout")),
        SummaryParameter("reps", ("reps",)),
        SummaryParameter("rounds", ("rounds",)),
    ),
    summary_estimates=(),
)
