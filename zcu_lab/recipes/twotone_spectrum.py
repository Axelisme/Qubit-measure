"""Qubit drive-frequency sweep, transition fit and reviewed calibration."""

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

from .helpers import (
    FrequencySweepCalibration,
    apply_frequency,
    choose_frequency,
    retain_scalar,
    set_frequency_sweep,
)


def _validate_inputs(
    references: tuple[tuple[str, str | None], ...],
    scalars: tuple[tuple[str, float | None], ...],
    span_mhz: float | None,
    points: int | None,
) -> None:
    """Reject unusable explicit inputs before any GUI preparation."""
    for name, value in references:
        if value is not None and not value.strip():
            raise ValueError(f"{name} must be non-empty or null")
    for name, value in scalars:
        if value is not None and not isfinite(value):
            raise ValueError(f"{name} must be finite or null")
    if span_mhz is not None and span_mhz <= 0:
        raise ValueError("span_mhz must be positive")
    if points is not None and points < 2:
        raise ValueError("points must be at least two")


def twotone_spectrum(  # noqa: PLR0913 - Typed keywords preserve the public tool inputs.
    session: RecipeSession,
    *,
    target: str | None = None,
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    drive_ref: str | None = None,
    use_reset: str | None = None,
    center_mhz: float,
    span_mhz: float,
    gain: float | None = None,
    pulse_length_us: float | None = None,
    points: int,
    reps: int | None = None,
    rounds: int | None = None,
) -> RecipeGenerator:
    """Sweep qubit drive frequency to locate a transition.

    Apply optional reset, a fixed-length/gain qubit pulse, then readout.
    Primary fits the averaged IQ response to estimate transition frequency
    and linewidth. Save raw data and analysis, then ask before applying it.

    session is driver-supplied. target overrides the qubit alias for this Reset;
    None uses global aliases. reuse_tab_id resets an idle twotone/freq tab.
    readout_ref/drive_ref select library modules; None retains GUI selections.
    Readout frequency uses a valid library value, then resonator calibration.
    None disables reset; use_reset selects its module. Required center_mhz and
    positive span_mhz are finite MHz values; required points is an integer >=2.
    gain is native amplitude; pulse_length_us is the fixed duration in us.
    reps/rounds control averaging; other None values retain GUI defaults.
    Blank references and invalid span/points fail before setup; missing
    calibration reports needs_parameters. Invalid inputs/cfg/native failures
    propagate. Cancellation stops later steps; skipped writeback changes nothing.
    """
    # 1. Validate inputs before selecting the measurement sources.
    _validate_inputs(
        (
            ("reuse_tab_id", reuse_tab_id),
            ("readout_ref", readout_ref),
            ("drive_ref", drive_ref),
            ("use_reset", use_reset),
        ),
        (
            ("center_mhz", center_mhz),
            ("span_mhz", span_mhz),
            ("gain", gain),
            ("pulse_length_us", pulse_length_us),
        ),
        span_mhz,
        points,
    )

    # 2. Open/reset the tab and select the pulse libraries.
    tab = session.open_tab(
        "twotone/freq",
        aliases=None if target is None else {"qubit": target},
        reuse=reuse_tab_id,
    )
    if use_reset is None:
        tab.disable_library(("modules", "reset"))
    else:
        tab.use_library(("modules", "reset"), use_reset)
    tab.use_library(("modules", "readout"), readout_ref)
    tab.use_library(("modules", "qub_pulse"), drive_ref)

    # 3. Set the drive sweep and resolve the fixed readout frequency.
    # Report all missing sources together.
    missing: list[MissingParameter] = []
    missing.extend(
        set_frequency_sweep(
            tab,
            ("sweep", "freq"),
            calibration=FrequencySweepCalibration("$qubit.freq", "$qubit.kappa", 3.0),
            center_mhz=center_mhz,
            span_mhz=span_mhz,
            expts=points,
        )
    )
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

    if gain is not None or pulse_length_us is not None:
        tab.use_inline(("modules", "qub_pulse"))
        tab.set(("modules", "qub_pulse", "gain"), gain, source="explicit")
    else:
        retain_scalar(tab, ("modules", "qub_pulse", "gain"))
    if pulse_length_us is not None:
        tab.use_inline(("modules", "qub_pulse", "waveform"))
        tab.set(("modules", "qub_pulse", "waveform", "length"), pulse_length_us)
    else:
        retain_scalar(tab, ("modules", "qub_pulse", "waveform", "length"))
    tab.set(("reps",), reps, source="explicit")
    tab.set(("rounds",), rounds, source="explicit")

    # 4. Acquire and save the raw IQ sweep.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()

    # 5. Fit the transition frequency and linewidth.
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
    name="twotone_spectrum",
    description="Run one two-tone spectrum and save raw data and Primary analysis.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["center_mhz", "span_mhz", "points"],
        "properties": {
            "center_mhz": {"type": "number"},
            "span_mhz": {"type": "number"},
            "points": {"type": "integer"},
            "target": {"type": ["string", "null"], "minLength": 1},
            **{
                name: {"type": ["string", "null"], "minLength": 1}
                for name in ("reuse_tab_id", "readout_ref", "drive_ref", "use_reset")
            },
            **{
                name: {"type": ["number", "null"]}
                for name in ("gain", "pulse_length_us")
            },
            **{name: {"type": ["integer", "null"]} for name in ("reps", "rounds")},
        },
    },
    run=twotone_spectrum,
    adapter_name="twotone/freq",
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
        SummaryParameter(
            "pulse_length_us", ("modules", "qub_pulse", "waveform", "length"), "us"
        ),
        SummaryParameter("gain", ("modules", "qub_pulse", "gain")),
        SummaryParameter("frequency_sweep", ("sweep", "freq"), "MHz"),
        SummaryParameter("drive_ref", ("modules", "qub_pulse")),
    ),
    summary_estimates=(
        SummaryEstimate("freq", ("freq",), ("freq_err",), "MHz"),
        SummaryEstimate("fwhm", ("fwhm",), ("fwhm_err",), "MHz"),
    ),
)
