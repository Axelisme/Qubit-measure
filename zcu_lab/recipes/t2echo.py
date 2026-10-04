"""One calibrated Echo total-delay sweep, with explicit generator handoffs."""

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
    detune_ratio: float | None,
    points: int | None,
) -> None:
    """Reject blank references, invalid delays/detuning and undersized sweeps."""
    for name, value in references:
        if value is not None and not value.strip():
            raise ValueError(f"{name} must be non-empty or null")
    if max_delay_us is not None and (not isfinite(max_delay_us) or max_delay_us <= 0):
        raise ValueError("max_delay_us must be positive and finite or null")
    if detune_ratio is not None and not isfinite(detune_ratio):
        raise ValueError("detune_ratio must be finite or null")
    if points is not None and points < 2:
        raise ValueError("points must be at least two")


def t2echo(  # noqa: PLR0913 - Typed keywords preserve the public tool inputs.
    session: RecipeSession,
    *,
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    use_reset: str | None = None,
    pi_ref: str | None = None,
    pi2_ref: str | None = None,
    max_delay_us: float | None = None,
    detune_ratio: float | None = None,
    points: int | None = None,
    reps: int | None = None,
    rounds: int | None = None,
) -> RecipeGenerator:
    """Save one Echo total-delay sweep and yield Primary before asking to write.

    session is driver-supplied. reuse_tab_id resets that idle twotone/t2echo tab.
    readout_ref/pi_ref/pi2_ref select calibrated library modules; None retains
    the GUI selection. Readout frequency prefers a usable library, then
    resonator calibration. None disables reset; use_reset selects its library
    module. max_delay_us is a positive finite total-delay endpoint in us, not
    a half-delay; points is at least 2. detune_ratio is finite and dimensionless.
    reps/rounds are integer counts. Omitted values retain GUI defaults,
    including expressions. Invalid arguments fail before preparing; absent
    readout/pi/pi2 calibration reports every missing name. Invalid cfg/native
    failures throw at their author step. Cancelled yields return; finish_early
    saves usable raw data. Primary may hand off for interaction. accepted
    explicitly writes; skipped writes none.
    """
    _validate_inputs(
        (
            ("reuse_tab_id", reuse_tab_id),
            ("readout_ref", readout_ref),
            ("use_reset", use_reset),
            ("pi_ref", pi_ref),
            ("pi2_ref", pi2_ref),
        ),
        max_delay_us,
        detune_ratio,
        points,
    )

    tab = session.open_tab("twotone/t2echo", reuse=reuse_tab_id)
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
    try:
        tab.use_library("modules.pi2_pulse", pi2_ref, required="pi2_ref")
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
    tab.set("detune_ratio", detune_ratio, source="detune_ratio")

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
    name="t2echo",
    description="Run one calibrated Echo total-delay sweep.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            **{
                name: {"type": ["string", "null"], "minLength": 1}
                for name in (
                    "reuse_tab_id",
                    "readout_ref",
                    "use_reset",
                    "pi_ref",
                    "pi2_ref",
                )
            },
            **{
                name: {"type": ["number", "null"]}
                for name in ("max_delay_us", "detune_ratio")
            },
            **{
                name: {"type": ["integer", "null"]}
                for name in ("points", "reps", "rounds")
            },
        },
    },
    run=t2echo,
    adapter_name="twotone/t2echo",
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
        SummaryParameter("detune_ratio", "detune_ratio"),
        SummaryParameter("pi_ref", "modules.pi_pulse"),
        SummaryParameter("pi2_ref", "modules.pi2_pulse"),
    ),
    summary_estimates=(SummaryEstimate("t2e", "t2e", "t2e_err", "us"),),
)
