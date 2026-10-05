"""GE calibration with explicit Primary, Post and writeback handoffs."""

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
    references: tuple[tuple[str, str | None], ...], shots: int | None
) -> None:
    """Reject blank references and nonpositive shots before opening a tab."""
    for name, value in references:
        if value is not None and not value.strip():
            raise ValueError(f"{name} must be a non-empty string or null")
    if shots is not None and shots <= 0:
        raise ValueError("shots must be a positive integer or null")


def singleshot_ge(  # noqa: PLR0913 - Typed keywords preserve the public tool inputs.
    session: RecipeSession,
    *,
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    pi_ref: str | None = None,
    use_reset: str | None = None,
    init_pulse_ref: str | None = None,
    shots: int | None = None,
) -> RecipeGenerator:
    """Save GE raw data, then yield Primary and Post before asking to write.

    session is driver-supplied. reuse_tab_id resets that idle singleshot/ge tab.
    readout_ref and pi_ref select calibrated library modules; None retains their
    GUI selections. The readout frequency prefers a usable library, then resonator
    calibration. None disables reset/init_pulse; named values select libraries.
    shots is a positive acquisition count; None retains the GUI default.
    Missing readout/pi calibration ends needs_parameters with both missing names.
    Invalid inputs/cfg/native failures throw at their author step. Cancelled yields
    return; finish_early saves usable raw data. Both analyses may hand off for
    interaction. accepted explicitly writes Primary then Post; skipped writes none.
    """
    _validate_inputs(
        (
            ("reuse_tab_id", reuse_tab_id),
            ("readout_ref", readout_ref),
            ("pi_ref", pi_ref),
            ("use_reset", use_reset),
            ("init_pulse_ref", init_pulse_ref),
        ),
        shots,
    )

    tab = session.open_tab("singleshot/ge", reuse=reuse_tab_id)
    if use_reset is None:
        tab.disable_library("modules.reset")
    else:
        tab.use_library("modules.reset", use_reset)
    if init_pulse_ref is None:
        tab.disable_library("modules.init_pulse")
    else:
        tab.use_library("modules.init_pulse", init_pulse_ref)
    tab.use_library("modules.readout", readout_ref)

    # Report both absent calibrations before any Run, as the existing tool does.
    missing: list[MissingParameter] = []
    try:
        tab.set_frequency(
            "modules.readout", calibration="resonator", required="readout_ref"
        )
    except RecipeNeedsParameters as error:
        missing.extend(error.missing)
    try:
        tab.use_library("modules.probe_pulse", pi_ref, required="pi_ref")
    except RecipeNeedsParameters as error:
        missing.extend(error.missing)
    if missing:
        raise RecipeNeedsParameters(tuple(missing))
    tab.set("shots", shots, source="shots")

    run, status = yield tab.run()
    if status == "cancelled":
        return
    assert isinstance(run, RecipeRun)
    run.save_raw()
    _, status = yield run.analyze("primary")
    if status == "cancelled":
        return
    _, status = yield run.analyze("post")
    if status == "cancelled":
        return
    decision, status = yield run.propose_writeback()
    if status == "cancelled":
        return
    if decision == "accepted":
        tab.accept()


DEFINITION = RecipeDefinition(
    name="singleshot_ge",
    description="Run GE single-shot calibration with Primary and Post analysis.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            **{
                name: {"type": ["string", "null"], "minLength": 1}
                for name in (
                    "reuse_tab_id",
                    "readout_ref",
                    "pi_ref",
                    "use_reset",
                    "init_pulse_ref",
                )
            },
            "shots": {"type": ["integer", "null"]},
        },
    },
    run=singleshot_ge,
    adapter_name="singleshot/ge",
    summary_parameters=(
        SummaryParameter(
            "readout_frequency_mhz", "modules.readout.pulse_cfg.freq", "MHz"
        ),
        SummaryParameter("ro_frequency_mhz", "modules.readout.ro_cfg.ro_freq", "MHz"),
        SummaryParameter("readout_ref", "modules.readout"),
        SummaryParameter("use_reset", "modules.reset"),
        SummaryParameter("shots", "shots"),
        SummaryParameter("pi_ref", "modules.probe_pulse"),
        SummaryParameter("init_pulse_ref", "modules.init_pulse"),
    ),
    summary_estimates=(
        SummaryEstimate("fidelity", "fidelity"),
        SummaryEstimate("theta", "theta"),
        SummaryEstimate("threshold", "threshold"),
        SummaryEstimate("ge_s", "ge_s"),
    ),
)
