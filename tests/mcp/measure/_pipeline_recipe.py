"""Injected pipeline for transport, worker and captured-reply contracts.

The fixture edits caller-supplied scalars and optionally selects library entries.
It has no calibration fallback, scientific defaults or experiment execution.
"""

from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    MissingParameter,
    RecipeDefinition,
    RecipeFieldEdit,
    RecipeGenerator,
    RecipeNeedsParameters,
    RecipeRun,
    RecipeSession,
    ask,
)


def pipeline(
    session: RecipeSession,
    *,
    frequency_mhz: float | None = None,
    readout_length_us: float | None = None,
    trigger_offset_us: float | None = None,
    rounds: int | None = None,
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    use_reset: str | None = None,
    init_pulse_ref: str | None = None,
    post: bool = False,
) -> RecipeGenerator:
    tab = session.open_tab("lookback", reuse=reuse_tab_id)
    for path, reference in (
        (("modules", "reset"), use_reset),
        (("modules", "init_pulse"), init_pulse_ref),
    ):
        if reference is None:
            tab.disable_library(path)
        else:
            tab.use_library(path, reference)
    if readout_ref is not None:
        tab.use_library(("modules", "readout"), readout_ref)
    if frequency_mhz is None:
        raise RecipeNeedsParameters(
            (MissingParameter("frequency_mhz", "Supply a value"),)
        )
    tab.set_fields(
        (
            RecipeFieldEdit(("modules", "readout", "freq"), frequency_mhz),
            RecipeFieldEdit(("modules", "readout", "ro_length"), readout_length_us),
            RecipeFieldEdit(("modules", "readout", "trig_offset"), trigger_offset_us),
            RecipeFieldEdit(("rounds",), rounds),
        )
    )
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled":
        return
    assert isinstance(run, RecipeRun)
    run.save_raw()
    analysis = yield from run.analyze_step("primary")
    if analysis.status == "cancelled":
        return
    if post:
        analysis = yield from run.analyze_step("post")
        if analysis.status == "cancelled":
            return
    decision = yield from ask(run.propose_writeback())
    if decision.status != "cancelled" and decision.decision == "accepted":
        tab.accept()


PIPELINE = RecipeDefinition(
    "pipeline",
    "Run the recording pipeline",
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            **{
                name: {"type": ["number", "null"]}
                for name in (
                    "frequency_mhz",
                    "readout_length_us",
                    "trigger_offset_us",
                )
            },
            **{
                name: {"type": ["string", "null"], "minLength": 1}
                for name in (
                    "reuse_tab_id",
                    "readout_ref",
                    "use_reset",
                    "init_pulse_ref",
                )
            },
            "rounds": {"type": ["integer", "null"]},
            "post": {"type": "boolean"},
        },
    },
    pipeline,
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
    summary_estimates=(),
)
