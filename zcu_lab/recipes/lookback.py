"""Readout arrival-time trace and proposed ADC trigger timing."""

from zcu_tools.mcp.measure.execution_reply import SummaryEstimate, SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    RecipeDefinition,
    RecipeGenerator,
    RecipeNeedsParameters,
    RecipeSession,
    RecipeTab,
    ask,
)

from .helpers import apply_frequency, choose_frequency, retain_scalar


def _set_readout_timing(
    tab: RecipeTab,
    readout_length_us: float | None,
    trigger_offset_us: float | None,
) -> None:
    """Keep omitted key values and provenance; convert only for explicit edits."""
    for field, value in (
        (("modules", "readout", "ro_length"), readout_length_us),
        (("modules", "readout", "trig_offset"), trigger_offset_us),
    ):
        if value is not None:
            tab.use_inline(("modules", "readout"))
            tab.set(field, value, source="explicit")
        else:
            retain_scalar(tab, field)


# tolerated: Public lookback keywords must match the handwritten input_schema.
def lookback(  # noqa: PLR0913
    session: RecipeSession,
    *,
    target: str | None = None,
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    use_reset: str | None = None,
    init_pulse_ref: str | None = None,
    frequency_mhz: float | None = None,
    readout_length_us: float | None = None,
    trigger_offset_us: float | None = None,
    rounds: int | None = None,
) -> RecipeGenerator:
    """Capture the readout pulse's arrival and ADC time envelope.

    Apply optional reset/init pulses, then read out at a fixed frequency.
    Primary estimates the trigger offset from the rising signal envelope.
    Save the raw trace and analysis, then ask before applying the timing proposal.

    session is driver-supplied. target overrides the qubit alias for this Reset;
    None uses global aliases. reuse_tab_id resets a named idle Lookback tab.
    readout_ref selects a library module; None retains its GUI selection.
    None disables reset/init_pulse; named references select those modules.
    Explicit frequency_mhz wins, in MHz. Otherwise an explicit readout_ref may
    supply frequency, then resonator calibration. Timing inputs use us:
    readout_length_us sets the window and trigger_offset_us its offset.
    rounds is an averaging count. Other None values retain GUI defaults.
    Missing frequency reports needs_parameters; invalid cfg/native failures
    propagate. Cancellation stops later steps; skipped writeback changes nothing.
    """
    # 1. Select readout and optional state preparation.
    tab = session.open_tab(
        "lookback",
        aliases=None if target is None else {"qubit": target},
        reuse=reuse_tab_id,
    )
    tab.use_library(("modules", "readout"), readout_ref)
    if use_reset is None:
        tab.disable_library(("modules", "reset"))
    else:
        tab.use_library(("modules", "reset"), use_reset)
    if init_pulse_ref is None:
        tab.disable_library(("modules", "init_pulse"))
    else:
        tab.use_library(("modules", "init_pulse"), init_pulse_ref)

    # 2. Resolve readout frequency and set the ADC timing window.
    frequency = choose_frequency(
        tab,
        ("modules", "readout", "freq"),
        frequency_mhz,
        calibration="$resonator.freq",
        prefer_library=readout_ref is not None,
        required="frequency_mhz",
    )
    if frequency.missing:
        raise RecipeNeedsParameters(frequency.missing)
    if frequency.edit is not None:
        tab.use_inline(("modules", "readout"))
        apply_frequency(tab, frequency.edit)
    _set_readout_timing(tab, readout_length_us, trigger_offset_us)
    tab.set(("rounds",), rounds, source="explicit")

    # 3. Acquire and save the raw time trace.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()

    # 4. Estimate the signal arrival time for the ADC trigger.
    analysis = yield from run.analyze_step("primary")
    if analysis.status == "cancelled":
        return

    # 5. Ask before applying the proposed calibration.
    decision = yield from ask(run.propose_writeback())
    if decision.status == "cancelled":
        return
    if decision.decision == "accepted":
        tab.accept()


DEFINITION = RecipeDefinition(
    name="lookback",
    description=(
        "Run lookback once, save raw data, analyze and save figures. "
        "Wait up to 300 seconds; unfinished work continues by execution ID."
    ),
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
                    "use_reset",
                    "init_pulse_ref",
                )
            },
            **{
                name: {"type": ["number", "null"]}
                for name in ("frequency_mhz", "readout_length_us", "trigger_offset_us")
            },
            "rounds": {"type": ["integer", "null"]},
        },
    },
    run=lookback,
    adapter_name="lookback",
    summary_parameters=(
        SummaryParameter("frequency_mhz", ("modules", "readout", "freq"), "MHz"),
        SummaryParameter(
            "readout_length_us", ("modules", "readout", "ro_length"), "us"
        ),
        SummaryParameter(
            "trigger_offset_us", ("modules", "readout", "trig_offset"), "us"
        ),
        SummaryParameter("rounds", ("rounds",)),
        SummaryParameter("readout_ref", ("modules", "readout")),
        SummaryParameter("use_reset", ("modules", "reset")),
        SummaryParameter("init_pulse_ref", ("modules", "init_pulse")),
    ),
    summary_estimates=(
        SummaryEstimate("predict_offset", ("predict_offset",), unit="us"),
    ),
)
