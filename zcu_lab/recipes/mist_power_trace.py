"""Acquire and save an averaged-IQ MIST probe-gain trace."""

from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    RecipeDefinition,
    RecipeGenerator,
    RecipeSession,
)


def mist_power_trace(
    session: RecipeSession,
    *,
    target: str | None = None,
    reuse_tab_id: str | None = None,
    probe_ref: str | None = None,
    readout_ref: str | None = None,
) -> RecipeGenerator:
    """Probe across gain, save mean complex IQ and plot its displacement.

    session is driver-supplied. target optionally overrides the qubit alias;
    None retains global aliases. reuse_tab_id resets an idle matching tab.
    probe_ref/readout_ref optionally select library modules; None retains the
    GUI defaults. Reset/init choices and acquisition defaults come from Form.
    Reuse also discards previous edits; run directly in the GUI when custom
    acquisition settings are needed. No writeback is proposed. Blank references
    raise ValueError. Invalid cfg/native failures propagate, and cancellation
    stops later steps.
    """
    # 1. Select a fixed source capture and optional module libraries.
    for name, value in (
        ("target", target),
        ("reuse_tab_id", reuse_tab_id),
        ("probe_ref", probe_ref),
        ("readout_ref", readout_ref),
    ):
        if value is not None and not value.strip():
            raise ValueError(f"{name} must be a non-empty string or null")
    tab = session.open_tab(
        "mist/power_dep/single_trace",
        aliases=None if target is None else {"qubit": target},
        reuse=reuse_tab_id,
    )
    tab.use_library(("modules", "probe_pulse"), probe_ref)
    tab.use_library(("modules", "readout"), readout_ref)
    tab.retain_source(("sweep", "gain"), "gui_default")
    tab.retain_source(("reps",), "gui_default")
    tab.retain_source(("rounds",), "gui_default")

    # 2. Acquire and save the complex IQ trace before plotting a reference view.
    result = yield from tab.run_step()
    run = result.run
    if result.status == "cancelled" or run is None:
        return
    run.save_raw()

    # 3. Publish the displacement figure without a fit or calibration proposal.
    yield from run.analyze_step("primary")


DEFINITION = RecipeDefinition(
    name="mist_power_trace",
    description="Acquire MIST mean IQ versus probe gain, save raw data and plot.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            name: {"type": ["string", "null"], "minLength": 1}
            for name in ("target", "reuse_tab_id", "probe_ref", "readout_ref")
        },
    },
    run=mist_power_trace,
    adapter_name="mist/power_dep/single_trace",
    summary_parameters=(
        SummaryParameter("probe_ref", ("modules", "probe_pulse")),
        SummaryParameter("readout_ref", ("modules", "readout")),
        SummaryParameter("gain", ("sweep", "gain")),
        SummaryParameter("reps", ("reps",)),
        SummaryParameter("rounds", ("rounds",)),
    ),
    summary_estimates=(),
)
