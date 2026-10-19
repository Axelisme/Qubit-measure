"""Measure CPMG decay across positive refocusing-pulse counts."""

from math import isfinite

from zcu_tools.gui.form_adapter.log_points import PositiveLogIntegerPoints
from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    MissingParameter,
    RecipeDefinition,
    RecipeGenerator,
    RecipeNeedsParameters,
    RecipeSession,
)

from zcu_lab.v2.twotone.time_domain.cpmg.core import CPMG_SweepCfg

from .helpers import apply_frequency, choose_frequency


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
    if minimum is not None and maximum is not None and maximum <= minimum:
        raise ValueError("max_length_us must exceed min_length_us")
    _positive_integer("length_points", points, minimum=2)


def cpmg(  # noqa: PLR0913 - Typed keywords preserve the public tool inputs.
    session: RecipeSession,
    *,
    target: str | None = None,
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    use_reset: str | None = None,
    pi_ref: str | None = None,
    pi2_ref: str | None = None,
    max_n: int | None = None,
    points: int | None = None,
    times: list[int] | None = None,
    min_length_us: float | None = None,
    max_length_us: float | None = None,
    length_points: int | None = None,
    detune_ratio: float | None = None,
    reps: int | None = None,
    rounds: int | None = None,
) -> RecipeGenerator:
    """Measure coherence versus CPMG pulse count and total evolution time.

    Apply pi/2, n refocusing pi pulses, a final pi/2 projection and readout.
    Save raw IQ, then fit T_CPMG for each count. No calibration is written.

    session comes from the recipe driver. target binds the qubit alias without
    changing focus. reuse_tab_id resets an idle twotone/cpmg tab and cannot be
    combined with target. readout_ref/pi_ref/pi2_ref select calibrated modules;
    None retains the GUI choices. use_reset=None disables reset.

    Provide max_n and points together for positive geometrically spaced counts,
    or times as a nonempty list of positive integers. max_n and points must be
    positive integers; max_n>1 requires at least two points to retain 1 and
    max_n. max_n=1 accepts one point and yields [1]. Use times for any other
    single count. Rounding and deduplication may reduce the count. Supplying
    neither retains the GUI counts. min_length_us/max_length_us are positive
    finite shared total-time bounds in us. length_points is at least 2.
    detune_ratio is finite fringes per time index; reps/rounds are positive
    integer acquisition counts. Omitted scalar inputs retain GUI defaults.

    Invalid overrides raise ValueError before opening a tab. Missing calibrated
    pulses/readout raise RecipeNeedsParameters. Acquisition, save and analysis
    failures propagate from their step. Cancelled steps end the recipe.
    """
    # 1. Validate overrides and use the shared Log engine for positive counts.
    for name, value in (
        ("reuse_tab_id", reuse_tab_id),
        ("readout_ref", readout_ref),
        ("use_reset", use_reset),
        ("pi_ref", pi_ref),
        ("pi2_ref", pi2_ref),
    ):
        if value is not None and not value.strip():
            raise ValueError(f"{name} must be non-empty or null")
    counts = _prepare_counts(max_n, points, times)
    _validate_time_window(min_length_us, max_length_us, length_points)
    _positive_integer("reps", reps)
    _positive_integer("rounds", rounds)
    if detune_ratio is not None and not isfinite(detune_ratio):
        raise ValueError("detune_ratio must be finite or null")

    # 2. Select captured calibrated modules and optional reset.
    tab = session.open_tab(
        "twotone/cpmg",
        aliases=None if target is None else {"qubit": target},
        reuse=reuse_tab_id,
    )
    if use_reset is None:
        tab.disable_library(("modules", "reset"))
    else:
        tab.use_library(("modules", "reset"), use_reset)
    tab.use_library(("modules", "readout"), readout_ref)
    missing: list[MissingParameter] = []
    frequency = choose_frequency(
        tab,
        ("modules", "readout", "freq"),
        calibration="$resonator.freq",
        required="readout_ref",
    )
    missing.extend(frequency.missing)
    if frequency.edit is not None:
        tab.use_inline(("modules", "readout"))
        apply_frequency(tab, frequency.edit)
    if (
        parameter := tab.use_library(("modules", "pi_pulse"), pi_ref, required="pi_ref")
    ) is not None:
        missing.append(parameter)
    if (
        parameter := tab.use_library(
            ("modules", "pi2_pulse"), pi2_ref, required="pi2_ref"
        )
    ) is not None:
        missing.append(parameter)
    if missing:
        raise RecipeNeedsParameters(tuple(missing))

    # 3. Set counts, the shared total-time window and averaging.
    tab.set_integer_points(("sweep", "times"), counts)
    tab.set(("extras", "min_length_us"), min_length_us)
    tab.set(("extras", "max_length_us"), max_length_us)
    tab.set(("extras", "length_points"), length_points)
    tab.set(("detune_ratio",), detune_ratio)
    tab.set(("reps",), reps)
    tab.set(("rounds",), rounds)

    # 4. Acquire and save the complete raw count/time grid before fitting.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()

    # 5. Fit each decay and display T_CPMG versus count, without writeback.
    analysis = yield from run.analyze_step("primary")
    if analysis.status == "cancelled":
        return


DEFINITION = RecipeDefinition(
    name="cpmg",
    description="Measure CPMG coherence over positive pi-pulse counts and time.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            **{
                name: {"type": ["string", "null"], "minLength": 1}
                for name in (
                    "target",
                    "reuse_tab_id",
                    "readout_ref",
                    "use_reset",
                    "pi_ref",
                    "pi2_ref",
                )
            },
            "times": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "minItems": 1,
                "description": "Positive explicit pi-pulse counts; excludes max_n/points.",
            },
            **{
                name: {"type": ["number", "null"]}
                for name in ("min_length_us", "max_length_us", "detune_ratio")
            },
            **{
                name: {"type": ["integer", "null"]}
                for name in ("max_n", "points", "length_points", "reps", "rounds")
            },
        },
    },
    run=cpmg,
    adapter_name="twotone/cpmg",
    summary_estimates=(),
    summary_parameters=(
        SummaryParameter("counts", ("sweep", "times")),
        SummaryParameter("min_length_us", ("extras", "min_length_us"), "us"),
        SummaryParameter("max_length_us", ("extras", "max_length_us"), "us"),
        SummaryParameter("length_points", ("extras", "length_points")),
        SummaryParameter("detune_ratio", ("detune_ratio",)),
        SummaryParameter("reps", ("reps",)),
        SummaryParameter("rounds", ("rounds",)),
        SummaryParameter("readout_ref", ("modules", "readout")),
        SummaryParameter("pi_ref", ("modules", "pi_pulse")),
        SummaryParameter("pi2_ref", ("modules", "pi2_pulse")),
        SummaryParameter("use_reset", ("modules", "reset")),
    ),
)
