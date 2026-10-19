"""Measure Rabi oscillations versus pulse length to calibrate rotation durations."""

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

from .helpers import apply_frequency, choose_frequency, retain_scalar


def _validate_inputs(
    references: tuple[tuple[str, str | None], ...],
    scalars: tuple[tuple[str, float | None], ...],
    points: int | None,
) -> None:
    """Reject blank references and nonfinite scalar inputs before preparing."""
    for name, value in references:
        if value is not None and not value.strip():
            raise ValueError(f"{name} must be non-empty or null")
    for name, value in scalars:
        if value is not None and not isfinite(value):
            raise ValueError(f"{name} must be finite or null")
    if points is not None and points < 2:
        raise ValueError("points must be at least two")


def time_rabi(  # noqa: PLR0913 - Typed keywords preserve the public tool inputs.
    session: RecipeSession,
    *,
    target: str | None = None,
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    drive_ref: str | None = None,
    use_reset: str | None = None,
    frequency_mhz: float | None = None,
    gain: float | None = None,
    max_length_us: float | None = None,
    points: int | None = None,
    reps: int | None = None,
    rounds: int | None = None,
) -> RecipeGenerator:
    """Sweep pulse length at fixed gain to calibrate pi and pi/2 rotations.

    After optional reset, apply the drive pulse and read out the qubit at each
    length. Fit the Rabi oscillations to estimate pi and pi/2 durations, save raw
    IQ, and ask before applying the proposed calibration; skipping writes nothing.

    session is supplied by the recipe driver. target selects the qubit alias
    without changing GUI focus; omission uses focus. reuse_tab_id resets an idle
    len_rabi tab and must not be combined with target.
    readout_ref/drive_ref select library modules; None retains the GUI selection.
    Drive frequency uses explicit frequency_mhz (MHz), then a usable library
    value, then qubit calibration. Readout uses library frequency, then resonator
    calibration. No pi calibration is needed. use_reset selects a library reset
    module; None disables reset.
    gain is the finite fixed amplitude in native gain units. max_length_us is
    the finite length sweep stop in us; the GUI sweep start is retained.
    points is at least 2. reps/rounds are integer counts. Omitted sweep/scalar
    values retain GUI defaults and expressions.

    Invalid inputs raise ValueError before opening a tab. Missing frequency
    sources raise RecipeNeedsParameters with every missing input; invalid cfg
    or measurement/analysis failures propagate from the failing step.
    """
    # 1. Reject invalid overrides before preparing the measurement.
    _validate_inputs(
        (
            ("reuse_tab_id", reuse_tab_id),
            ("readout_ref", readout_ref),
            ("drive_ref", drive_ref),
            ("use_reset", use_reset),
        ),
        (
            ("frequency_mhz", frequency_mhz),
            ("gain", gain),
            ("max_length_us", max_length_us),
        ),
        points,
    )

    # 2. Choose the qubit tab, readout and optional reset modules.
    tab = session.open_tab(
        "twotone/rabi/len_rabi",
        aliases=None if target is None else {"qubit": target},
        reuse=reuse_tab_id,
    )
    if use_reset is None:
        tab.disable_library(("modules", "reset"))
    else:
        tab.use_library(("modules", "reset"), use_reset)
    tab.use_library(("modules", "readout"), readout_ref)
    tab.use_library(("modules", "qub_pulse"), drive_ref)

    # 3. Resolve drive and readout frequencies; report all missing sources.
    missing: list[MissingParameter] = []
    frequency = choose_frequency(
        tab,
        ("modules", "qub_pulse", "freq"),
        frequency_mhz,
        calibration="$qubit.freq",
        required="frequency_mhz",
    )
    missing.extend(frequency.missing)
    if frequency.edit is not None:
        tab.use_inline(("modules", "qub_pulse"))
        apply_frequency(tab, frequency.edit)
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
    if missing:
        raise RecipeNeedsParameters(tuple(missing))

    # 4. Sweep duration while keeping the drive gain fixed.
    tab.set_sweep(
        ("sweep", "length"),
        stop=max_length_us,
        expts=points,
        sources={"start": "explicit", "stop": "explicit", "expts": "explicit"},
    )
    if gain is not None:
        tab.use_inline(("modules", "qub_pulse"))
        tab.set(("modules", "qub_pulse", "gain"), gain, source="explicit")
    else:
        retain_scalar(tab, ("modules", "qub_pulse", "gain"))
    tab.set(("reps",), reps, source="explicit")
    tab.set(("rounds",), rounds, source="explicit")

    # 5. Acquire the sweep and save raw IQ before fitting.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()

    # 6. Fit Rabi oscillations to estimate pi and pi/2 pulse durations.
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
    name="time_rabi",
    description="Run one length Rabi sweep and save raw data and Primary analysis.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target": {"type": ["string", "null"], "minLength": 1},
            **{
                name: {"type": ["string", "null"], "minLength": 1}
                for name in ("reuse_tab_id", "readout_ref", "drive_ref", "use_reset")
            },
            **{
                name: {"type": ["number", "null"]}
                for name in ("frequency_mhz", "gain", "max_length_us")
            },
            **{
                name: {"type": ["integer", "null"]}
                for name in ("points", "reps", "rounds")
            },
        },
    },
    run=time_rabi,
    adapter_name="twotone/rabi/len_rabi",
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
        SummaryParameter("frequency_mhz", ("modules", "qub_pulse", "freq"), "MHz"),
        SummaryParameter("drive_ref", ("modules", "qub_pulse")),
        SummaryParameter("gain", ("modules", "qub_pulse", "gain")),
        SummaryParameter("length_sweep", ("sweep", "length"), "us"),
    ),
    summary_estimates=(
        SummaryEstimate("pi_len", ("pi_len",), ("pi_len_err",), "us"),
        SummaryEstimate("pi2_len", ("pi2_len",), ("pi2_len_err",), "us"),
        SummaryEstimate("rabi_f", ("rabi_f",), ("rabi_f_err",), "MHz"),
    ),
)
