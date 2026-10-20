"""Measure Rabi oscillations versus drive gain to calibrate rotation amplitudes."""

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
    gain_range: list[float] | None,
    points: int | None,
) -> None:
    """Reject blank references and malformed numeric inputs before preparing."""
    for name, value in references:
        if value is not None and not value.strip():
            raise ValueError(f"{name} must be non-empty or null")
    for name, value in scalars:
        if value is not None and not isfinite(value):
            raise ValueError(f"{name} must be finite or null")
    if gain_range is not None and (
        len(gain_range) != 2 or not all(isfinite(value) for value in gain_range)
    ):
        raise ValueError("gain_range must contain exactly two finite endpoints")
    if points is not None and points < 2:
        raise ValueError("points must be at least two")


# tolerated: Public gain Rabi keywords must match the handwritten input_schema.
def amplitude_rabi(  # noqa: PLR0913
    session: RecipeSession,
    *,
    target: str | None = None,
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    drive_ref: str | None = None,
    use_reset: str | None = None,
    frequency_mhz: float | None = None,
    pulse_length_us: float | None = None,
    gain_range: list[float] | None = None,
    points: int | None = None,
    reps: int | None = None,
    rounds: int | None = None,
) -> RecipeGenerator:
    """Sweep drive gain at fixed pulse length to calibrate pi and pi/2 rotations.

    After optional reset, apply the drive pulse and read out the qubit at each
    gain. Fit the Rabi oscillations to estimate pi and pi/2 gains, save raw IQ,
    and ask before applying the proposed calibration; skipping writes nothing.

    session is supplied by the recipe driver. target selects the qubit alias
    without changing GUI focus; omission uses focus. reuse_tab_id resets an idle
    amp_rabi tab and must not be combined with target.
    readout_ref/drive_ref select library modules; None retains the GUI selection.
    Drive frequency uses explicit frequency_mhz (MHz), then a usable library
    value, then qubit calibration. Readout uses library frequency, then resonator
    calibration. No pi calibration is needed. use_reset selects a library reset
    module; None disables reset.
    pulse_length_us is the fixed length in us. gain_range has two finite
    endpoints in native gain units; points is at least 2. reps/rounds are integer
    counts. Omitted sweep/scalar values retain GUI defaults and expressions.

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
        (("frequency_mhz", frequency_mhz), ("pulse_length_us", pulse_length_us)),
        gain_range,
        points,
    )

    # 2. Choose the qubit tab, readout and optional reset modules.
    tab = session.open_tab(
        "twotone/rabi/amp_rabi",
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

    # 4. Sweep gain while keeping the drive duration fixed.
    tab.set_sweep(
        ("sweep", "gain"),
        start=gain_range[0] if gain_range is not None else None,
        stop=gain_range[1] if gain_range is not None else None,
        expts=points,
        sources={"start": "explicit", "stop": "explicit", "expts": "explicit"},
    )
    if pulse_length_us is not None:
        tab.use_inline(("modules", "qub_pulse"))
        tab.use_inline(("modules", "qub_pulse", "waveform"))
        tab.set(("modules", "qub_pulse", "waveform", "length"), pulse_length_us)
    else:
        retain_scalar(tab, ("modules", "qub_pulse", "waveform", "length"))
    tab.set(("reps",), reps, source="explicit")
    tab.set(("rounds",), rounds, source="explicit")

    # 5. Acquire the sweep and save raw IQ before fitting.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()

    # 6. Fit Rabi oscillations to estimate pi and pi/2 drive gains.
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
    name="amplitude_rabi",
    description="Run one gain Rabi sweep and save raw data and Primary analysis.",
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
                for name in ("frequency_mhz", "pulse_length_us")
            },
            **{
                name: {"type": ["integer", "null"]}
                for name in ("points", "reps", "rounds")
            },
            "gain_range": {
                "type": ["array", "null"],
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
    },
    run=amplitude_rabi,
    adapter_name="twotone/rabi/amp_rabi",
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
        SummaryParameter(
            "pulse_length_us", ("modules", "qub_pulse", "waveform", "length"), "us"
        ),
        SummaryParameter("gain_sweep", ("sweep", "gain")),
    ),
    summary_estimates=(
        SummaryEstimate("pi_gain", ("pi_gain",), ("pi_gain_err",)),
        SummaryEstimate("pi2_gain", ("pi2_gain",), ("pi2_gain_err",)),
    ),
)
