"""One length Rabi sweep, with explicit Primary and writeback handoffs."""

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
    """Save one length sweep and yield Primary before asking to write.

    session is driver-supplied. reuse_tab_id resets that idle len_rabi tab.
    readout_ref/drive_ref select library modules; None retains the GUI selection.
    Drive frequency prefers explicit frequency_mhz, then a usable library, then
    qubit calibration. Readout prefers a usable library, then resonator calibration.
    No pi calibration is required. None disables reset; use_reset selects its
    library module. gain is the finite fixed amplitude in native gain units.
    max_length_us is the finite sweep stop in us; points is at least 2. The GUI
    sweep start is retained. reps/rounds are integer counts. Omitted values retain
    GUI defaults/expressions. Invalid inputs fail before preparing; absent
    frequency sources report all missing parameters. Invalid cfg/native failures
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
            ("frequency_mhz", frequency_mhz),
            ("gain", gain),
            ("max_length_us", max_length_us),
        ),
        points,
    )

    tab = session.open_tab("twotone/rabi/len_rabi", reuse=reuse_tab_id)
    if use_reset is None:
        tab.disable_library("modules.reset")
    else:
        tab.use_library("modules.reset", use_reset)
    tab.use_library("modules.readout", readout_ref)
    tab.use_library("modules.qub_pulse", drive_ref)

    # Preserve the complete missing-source list rather than failing on its first item.
    missing: list[MissingParameter] = []
    try:
        tab.set_frequency(
            "modules.qub_pulse.freq",
            frequency_mhz,
            calibration="qubit",
            required="frequency_mhz",
            source="frequency_mhz",
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

    tab.set_sweep(
        "sweep.length",
        stop=max_length_us,
        expts=points,
        sources={"start": "explicit", "stop": "max_length_us", "expts": "points"},
    )
    tab.set("modules.qub_pulse.gain", gain, source="gain")
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
    name="time_rabi",
    description="Run one length Rabi sweep and save raw data and Primary analysis.",
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
            "readout_frequency_mhz", "modules.readout.pulse_cfg.freq", "MHz"
        ),
        SummaryParameter("ro_frequency_mhz", "modules.readout.ro_cfg.ro_freq", "MHz"),
        SummaryParameter("readout_ref", "modules.readout"),
        SummaryParameter("use_reset", "modules.reset"),
        SummaryParameter("reps", "reps"),
        SummaryParameter("rounds", "rounds"),
        SummaryParameter("frequency_mhz", "modules.qub_pulse.freq", "MHz"),
        SummaryParameter("drive_ref", "modules.qub_pulse"),
        SummaryParameter("gain", "modules.qub_pulse.gain"),
        SummaryParameter("length_sweep", "sweep.length", "us"),
    ),
    summary_estimates=(
        SummaryEstimate("pi_len", "pi_len", "pi_len_err", "us"),
        SummaryEstimate("pi2_len", "pi2_len", "pi2_len_err", "us"),
        SummaryEstimate("rabi_f", "rabi_f", "rabi_f_err", "MHz"),
    ),
)
