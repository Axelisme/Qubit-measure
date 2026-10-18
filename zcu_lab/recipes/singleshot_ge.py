"""GE calibration with explicit Primary, Post and writeback handoffs."""

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
    references: tuple[tuple[str, str | None], ...], shots: int | None
) -> None:
    """Reject blank references and nonpositive shots before opening a tab."""
    for name, value in references:
        if value is not None and not value.strip():
            raise ValueError(f"{name} must be a non-empty string or null")
    if shots is not None and shots <= 0:
        raise ValueError("shots must be a positive integer or null")


# tolerated: Public GE single-shot keywords must match the handwritten input_schema.
def singleshot_ge(  # noqa: PLR0913
    session: RecipeSession,
    *,
    target: str | None = None,
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    pi_ref: str | None = None,
    use_reset: str | None = None,
    init_pulse_ref: str | None = None,
    shots: int | None = None,
) -> RecipeGenerator:
    """Save GE raw data, then yield Primary and Post before asking to write.

    target binds the new tab's primary role without changing GUI focus.
    Omission uses focus. Do not pass target when reusing an existing tab.

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

    tab = session.open_tab(
        "singleshot/ge",
        aliases=None if target is None else {"qubit": target},
        reuse=reuse_tab_id,
    )
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
    frequency = choose_frequency(
        tab,
        "modules.readout.freq",
        calibration="$resonator.freq",
        required="readout_ref",
    )
    missing.extend(frequency.missing)
    if frequency.edit is not None:
        tab.use_inline("modules.readout")
        apply_frequency(tab, frequency.edit)
    if (
        parameter := tab.use_library("modules.probe_pulse", pi_ref, required="pi_ref")
    ) is not None:
        missing.append(parameter)
    if missing:
        raise RecipeNeedsParameters(tuple(missing))
    tab.set("shots", shots, source="explicit")

    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()
    analysis = yield from run.analyze_step("primary")
    if analysis.status == "cancelled":
        return
    analysis = yield from run.analyze_step("post")
    if analysis.status == "cancelled":
        return
    decision = yield from ask(run.propose_writeback())
    if decision.status == "cancelled":
        return
    if decision.decision == "accepted":
        tab.accept()


DEFINITION = RecipeDefinition(
    name="singleshot_ge",
    description="Run GE single-shot calibration with Primary and Post analysis.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target": {"type": ["string", "null"], "minLength": 1},
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
            "readout_frequency_mhz",
            "modules.readout.freq",
            "MHz",
        ),
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
