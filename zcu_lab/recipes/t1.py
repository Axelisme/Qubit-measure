"""Measure excited-state relaxation versus delay to estimate T1."""

from math import isfinite

from zcu_tools.mcp.measure.execution_reply import SummaryEstimate, SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    MissingParameter,
    RecipeDefinition,
    RecipeGenerator,
    RecipeNeedsParameters,
    RecipeSession,
    ask,
)

from .helpers import apply_frequency, choose_frequency


def _validate_inputs(
    references: tuple[tuple[str, str | None], ...],
    max_delay_us: float | None,
    points: int | None,
) -> None:
    """Reject blank references, nonpositive delays and undersized sweeps."""
    for name, value in references:
        if value is not None and not value.strip():
            raise ValueError(f"{name} must be non-empty or null")
    if max_delay_us is not None and (not isfinite(max_delay_us) or max_delay_us <= 0):
        raise ValueError("max_delay_us must be positive and finite or null")
    if points is not None and points < 2:
        raise ValueError("points must be at least two")


# tolerated: Public T1 delay keywords must match the handwritten input_schema.
def t1(  # noqa: PLR0913
    session: RecipeSession,
    *,
    target: str | None = None,
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    use_reset: str | None = None,
    pi_ref: str | None = None,
    max_delay_us: float | None = None,
    points: int | None = None,
    reps: int | None = None,
    rounds: int | None = None,
) -> RecipeGenerator:
    """Measure T1 from the decay of the qubit's excited-state population.

    After optional reset, apply a calibrated pi pulse, wait the swept delay,
    and read out the qubit. Fit the decay to estimate the relaxation time, save
    raw IQ, and ask before applying the proposed T1; skipping writes nothing.

    session is supplied by the recipe driver. target selects the qubit alias
    without changing GUI focus; omission uses focus. reuse_tab_id resets an idle
    twotone/t1 tab and must not be combined with target.
    readout_ref/pi_ref select calibrated library modules; None retains the GUI
    selection, which must provide a calibrated pi pulse rather than a custom
    pulse. Readout frequency uses a usable library value, then resonator
    calibration. use_reset selects a library reset module; None disables reset.
    max_delay_us is a positive finite delay sweep stop in us; points is at least
    2. reps/rounds are integer counts. Omitted sweep/scalar values retain GUI
    defaults and expressions.

    Invalid inputs raise ValueError before opening a tab. Missing readout/pi
    calibration raises RecipeNeedsParameters with every missing input; invalid
    cfg or measurement/analysis failures propagate from the failing step.
    """
    # 1. Reject invalid overrides before preparing the measurement.
    _validate_inputs(
        (
            ("reuse_tab_id", reuse_tab_id),
            ("readout_ref", readout_ref),
            ("use_reset", use_reset),
            ("pi_ref", pi_ref),
        ),
        max_delay_us,
        points,
    )

    # 2. Choose the qubit tab, readout and optional reset modules.
    tab = session.open_tab(
        "twotone/t1",
        aliases=None if target is None else {"qubit": target},
        reuse=reuse_tab_id,
    )
    if use_reset is None:
        tab.disable_library(("modules", "reset"))
    else:
        tab.use_library(("modules", "reset"), use_reset)
    tab.use_library(("modules", "readout"), readout_ref)

    # 3. Require calibrated pulses and a readout frequency before acquisition.
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
    if missing:
        raise RecipeNeedsParameters(tuple(missing))

    # 4. Set the waiting-time sweep after the pi pulse.
    tab.set_sweep(
        ("sweep", "length"),
        stop=max_delay_us,
        expts=points,
        sources={"start": "explicit", "stop": "explicit", "expts": "explicit"},
    )
    tab.set(("reps",), reps, source="explicit")
    tab.set(("rounds",), rounds, source="explicit")

    # 5. Acquire the sweep and save raw IQ before fitting.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()

    # 6. Fit the population decay to estimate the relaxation time T1.
    analysis = yield from run.analyze_step("primary")
    if analysis.status == "cancelled":
        return

    # 7. Apply the proposed calibration only after explicit acceptance.
    decision = yield from ask(run.propose_writeback())
    if decision.status == "cancelled":
        return
    if decision.decision == "accepted":
        tab.accept()


DEFINITION = RecipeDefinition(
    name="t1",
    description="Run one calibrated T1 delay sweep and save raw and Primary analysis.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target": {"type": ["string", "null"], "minLength": 1},
            **{
                name: {"type": ["string", "null"], "minLength": 1}
                for name in ("reuse_tab_id", "readout_ref", "pi_ref", "use_reset")
            },
            "max_delay_us": {"type": ["number", "null"]},
            **{
                name: {"type": ["integer", "null"]}
                for name in ("points", "reps", "rounds")
            },
        },
    },
    run=t1,
    adapter_name="twotone/t1",
    summary_parameters=(
        SummaryParameter(
            "readout_frequency_mhz",
            ("modules", "readout", "freq"),
            "MHz",
        ),
        SummaryParameter("readout_ref", ("modules", "readout")),
        SummaryParameter("use_reset", ("modules", "reset")),
        SummaryParameter("reps", ("reps",)),
        SummaryParameter("rounds", ("rounds",)),
        SummaryParameter("delay", ("sweep", "length"), "us"),
        SummaryParameter("pi_ref", ("modules", "pi_pulse")),
    ),
    summary_estimates=(
        SummaryEstimate("t1", ("t1",), ("t1_err",), "us"),
        SummaryEstimate("t1b", ("t1b",), ("t1b_err",), "us"),
    ),
)
