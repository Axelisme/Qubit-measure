"""One calibrated T1 delay sweep, with explicit generator handoffs."""

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


def t1(  # noqa: PLR0913 - Typed keywords preserve the public tool inputs.
    session: RecipeSession,
    *,
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    use_reset: str | None = None,
    pi_ref: str | None = None,
    max_delay_us: float | None = None,
    points: int | None = None,
    reps: int | None = None,
    rounds: int | None = None,
) -> RecipeGenerator:
    """Save one T1 delay sweep and yield Primary before asking to write.

    session is driver-supplied. reuse_tab_id resets that idle twotone/t1 tab.
    readout_ref/pi_ref select calibrated library modules; None retains the GUI
    selection, which must use a calibrated pi pulse rather than a custom pulse.
    Readout frequency prefers a usable library, then resonator calibration.
    None disables reset; use_reset selects its library module. max_delay_us is
    a positive finite delay endpoint in us; points is at least 2. reps/rounds
    are integer counts. Omitted values retain GUI defaults and expressions.
    Invalid arguments fail before preparing; missing readout/pi calibration
    reports both names. Invalid cfg/native failures throw at their author step.
    Cancelled yields return; finish_early saves usable raw data. Primary may
    hand off for interaction. accepted explicitly writes; skipped writes none.
    """
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

    tab = session.open_tab("twotone/t1", reuse=reuse_tab_id)
    if use_reset is None:
        tab.disable_library("modules.reset")
    else:
        tab.use_library("modules.reset", use_reset)
    tab.use_library("modules.readout", readout_ref)

    # Report every absent calibration before Run, as the existing tool does.
    missing: list[MissingParameter] = []
    try:
        tab.set_frequency(
            "modules.readout", calibration="resonator", required="readout_ref"
        )
    except RecipeNeedsParameters as error:
        missing.extend(error.missing)
    try:
        tab.use_library("modules.pi_pulse", pi_ref, required="pi_ref")
    except RecipeNeedsParameters as error:
        missing.extend(error.missing)
    if missing:
        raise RecipeNeedsParameters(tuple(missing))

    tab.set_sweep(
        "sweep.length",
        stop=max_delay_us,
        expts=points,
        sources={"start": "explicit", "stop": "max_delay_us", "expts": "points"},
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
    name="t1",
    description="Run one calibrated T1 delay sweep and save raw and Primary analysis.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
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
            "readout_frequency_mhz", "modules.readout.pulse_cfg.freq", "MHz"
        ),
        SummaryParameter("ro_frequency_mhz", "modules.readout.ro_cfg.ro_freq", "MHz"),
        SummaryParameter("readout_ref", "modules.readout"),
        SummaryParameter("use_reset", "modules.reset"),
        SummaryParameter("reps", "reps"),
        SummaryParameter("rounds", "rounds"),
        SummaryParameter("delay", "sweep.length", "us"),
        SummaryParameter("pi_ref", "modules.pi_pulse"),
    ),
    summary_estimates=(
        SummaryEstimate("t1", "t1", "t1_err", "us"),
        SummaryEstimate("t1b", "t1b", "t1b_err", "us"),
    ),
)
