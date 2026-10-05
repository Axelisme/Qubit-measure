"""One calibrated onetone spectrum and its explicit writeback handoff."""

from zcu_tools.mcp.measure.execution_reply import SummaryEstimate, SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    RecipeDefinition,
    RecipeGenerator,
    RecipeRun,
    RecipeSession,
)


def onetone_spectrum(  # noqa: PLR0913 - Typed keywords preserve the public tool inputs.
    session: RecipeSession,
    *,
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    center_mhz: float | None = None,
    span_mhz: float | None = None,
    gain: float | None = None,
    points: int | None = None,
    reps: int | None = None,
    rounds: int | None = None,
) -> RecipeGenerator:
    """Save one spectrum's raw data and Primary images, then ask before writing.

    session is driver-supplied. reuse_tab_id resets that idle onetone/freq tab.
    readout_ref selects a library; None retains the GUI selection. MHz center/span
    default to resonator/linewidth calibration; other None values retain GUI
    defaults. gain is the native pulse amplitude and points is the sweep count.
    Missing calibration ends needs_parameters. Invalid cfg/native failures throw
    at their author step. Cancelled yields return; finish_early saves usable data.
    The writeback question resumes with accepted/skipped, not an automatic write.
    """
    for name, value in (("reuse_tab_id", reuse_tab_id), ("readout_ref", readout_ref)):
        if value is not None and not value.strip():
            raise ValueError(f"{name} must be a non-empty string or null")
    if span_mhz is not None and span_mhz <= 0:
        raise ValueError("span_mhz must be positive")
    tab = session.open_tab("onetone/freq", reuse=reuse_tab_id)
    tab.use_library("modules.readout", readout_ref)
    tab.set_frequency_sweep(
        "sweep.freq",
        calibration="resonator",
        center_mhz=center_mhz,
        span_mhz=span_mhz,
        expts=points,
    )
    tab.set("modules.readout.pulse_cfg.gain", gain)
    tab.set("reps", reps)
    tab.set("rounds", rounds)
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
    name="onetone_spectrum",
    description="Run one onetone spectrum and save raw data and Primary analysis.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            **{
                name: {"type": ["string", "null"], "minLength": 1}
                for name in ("reuse_tab_id", "readout_ref")
            },
            **{
                name: {"type": ["number", "null"]}
                for name in ("center_mhz", "span_mhz")
            },
            **{name: {"type": ["integer", "null"]} for name in ("reps", "rounds")},
            "gain": {"type": ["number", "null"]},
            "points": {"type": ["integer", "null"]},
        },
    },
    run=onetone_spectrum,
    adapter_name="onetone/freq",
    summary_parameters=(
        SummaryParameter("center_mhz", "center_mhz", "MHz"),
        SummaryParameter("span_mhz", "span_mhz", "MHz"),
        SummaryParameter("frequency_sweep", "sweep.freq", "MHz"),
        SummaryParameter("gain", "modules.readout.pulse_cfg.gain"),
        SummaryParameter("readout_ref", "modules.readout"),
        SummaryParameter("reps", "reps"),
        SummaryParameter("rounds", "rounds"),
    ),
    summary_estimates=(
        SummaryEstimate("freq", "freq", unit="MHz"),
        SummaryEstimate("fwhm", "fwhm", unit="MHz"),
    ),
)
