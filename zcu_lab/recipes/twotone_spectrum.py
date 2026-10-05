"""One calibrated qubit spectrum, with explicit Primary and writeback handoffs."""

from math import isfinite

from zcu_tools.mcp.measure.execution_reply import SummaryEstimate, SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    MissingParameter,
    RecipeDefinition,
    RecipeGenerator,
    RecipeNeedsParameters,
    RecipeRun,
    RecipeSession,
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
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    drive_ref: str | None = None,
    use_reset: str | None = None,
    center_mhz: float | None = None,
    span_mhz: float | None = None,
    gain: float | None = None,
    pulse_length_us: float | None = None,
    points: int | None = None,
    reps: int | None = None,
    rounds: int | None = None,
) -> RecipeGenerator:
    """Save one frequency sweep and yield Primary before asking to write.

    session is driver-supplied. reuse_tab_id resets that idle twotone/freq tab.
    readout_ref/drive_ref select library modules; None retains the GUI selection.
    Readout prefers usable library frequencies, then resonator calibration.
    None disables reset; use_reset selects its library module. center_mhz uses
    qubit calibration when omitted. span_mhz must be positive and defaults to
    the GUI's calibrated linewidth expression. gain is native pulse amplitude;
    pulse_length_us is the fixed pulse length in us. points is at least 2;
    reps/rounds are integer counts. Other omitted values retain GUI defaults.
    The driver rejects invalid schema/scalar inputs before calling this author.
    Blank references and invalid span/points fail before preparing. Missing
    calibration reports all missing parameters. Invalid cfg/native failures
    throw at their author step. Cancelled yields return; finish_early saves usable
    raw data. Primary may hand off for interaction. accepted explicitly writes;
    skipped writes none.
    """
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

    tab = session.open_tab("twotone/freq", reuse=reuse_tab_id)
    if use_reset is None:
        tab.disable_library("modules.reset")
    else:
        tab.use_library("modules.reset", use_reset)
    tab.use_library("modules.readout", readout_ref)
    tab.use_library("modules.qub_pulse", drive_ref)

    # Preserve the complete missing-source list rather than failing on its first item.
    missing: list[MissingParameter] = []
    try:
        tab.set_frequency_sweep(
            "sweep.freq",
            calibration="qubit",
            center_mhz=center_mhz,
            span_mhz=span_mhz,
            expts=points,
        )
    except RecipeNeedsParameters as error:
        missing.extend(error.missing)
    try:
        tab.set_frequency(
            "modules.readout", calibration="resonator", required="readout_ref"
        )
    except RecipeNeedsParameters as error:
        missing.extend(error.missing)
    if missing:
        raise RecipeNeedsParameters(tuple(missing))

    tab.set("modules.qub_pulse.gain", gain, source="gain")
    tab.set(
        "modules.qub_pulse.waveform.length", pulse_length_us, source="pulse_length_us"
    )
    tab.set("reps", reps, source="reps")
    tab.set("rounds", rounds, source="rounds")

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
    name="twotone_spectrum",
    description="Run one two-tone spectrum and save raw data and Primary analysis.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            **{
                name: {"type": ["string", "null"], "minLength": 1}
                for name in ("reuse_tab_id", "readout_ref", "drive_ref", "use_reset")
            },
            **{
                name: {"type": ["number", "null"]}
                for name in ("center_mhz", "span_mhz", "gain", "pulse_length_us")
            },
            **{
                name: {"type": ["integer", "null"]}
                for name in ("points", "reps", "rounds")
            },
        },
    },
    run=twotone_spectrum,
    adapter_name="twotone/freq",
    summary_parameters=(
        SummaryParameter(
            "readout_frequency_mhz", "modules.readout.pulse_cfg.freq", "MHz"
        ),
        SummaryParameter("ro_frequency_mhz", "modules.readout.ro_cfg.ro_freq", "MHz"),
        SummaryParameter("readout_ref", "modules.readout"),
        SummaryParameter("use_reset", "modules.reset"),
        SummaryParameter("reps", "reps"),
        SummaryParameter("rounds", "rounds"),
        SummaryParameter("pulse_length_us", "modules.qub_pulse.waveform.length", "us"),
        SummaryParameter("gain", "modules.qub_pulse.gain"),
        SummaryParameter("frequency_sweep", "sweep.freq", "MHz"),
        SummaryParameter("center_mhz", "center_mhz", "MHz"),
        SummaryParameter("span_mhz", "span_mhz", "MHz"),
        SummaryParameter("drive_ref", "modules.qub_pulse"),
    ),
    summary_estimates=(
        SummaryEstimate("freq", "freq", "freq_err", "MHz"),
        SummaryEstimate("fwhm", "fwhm", "fwhm_err", "MHz"),
    ),
)
