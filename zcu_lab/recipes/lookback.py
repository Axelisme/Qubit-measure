"""Lookback author workflow; cfg and completion ownership stay in the framework."""

from zcu_tools.mcp.measure.execution_reply import SummaryEstimate, SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    RecipeDefinition,
    RecipeGenerator,
    RecipeRun,
    RecipeSession,
)


def lookback(  # noqa: PLR0913 - Typed keywords preserve the public tool inputs.
    session: RecipeSession,
    *,
    reuse_tab_id: str | None = None,
    readout_ref: str | None = None,
    use_reset: str | None = None,
    init_pulse_ref: str | None = None,
    frequency_mhz: float | None = None,
    readout_length_us: float | None = None,
    trigger_offset_us: float | None = None,
    rounds: int | None = None,
) -> RecipeGenerator:
    """Run once, save raw/Primary images, then ask before explicit writeback.

    session is driver-supplied. Names select observed libraries; nullable reset
    and init_pulse disable those optional modules. reuse_tab_id resets the named
    idle Lookback tab. Frequencies use MHz, lengths/offsets use us, and None retains
    GUI defaults or calibrated frequency. Missing frequency ends needs_parameters;
    invalid cfg/native failures throw at their author step. Cancel returns without
    later saves/analyses/writes; finish_early saves usable partial Run data.
    """
    tab = session.open_tab("lookback", reuse=reuse_tab_id)
    tab.use_library("modules.readout", readout_ref)
    if use_reset is None:
        tab.disable_library("modules.reset")
    else:
        tab.use_library("modules.reset", use_reset)
    if init_pulse_ref is None:
        tab.disable_library("modules.init_pulse")
    else:
        tab.use_library("modules.init_pulse", init_pulse_ref)
    tab.set_frequency(
        "modules.readout",
        frequency_mhz,
        calibration="resonator",
        prefer_library=readout_ref is not None,
        required="frequency_mhz",
        source="frequency_mhz",
    )
    tab.set(
        "modules.readout.ro_cfg.ro_length",
        readout_length_us,
        source="readout_length_us",
    )
    tab.set(
        "modules.readout.ro_cfg.trig_offset",
        trigger_offset_us,
        source="trigger_offset_us",
    )
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
    name="lookback",
    description=(
        "Run lookback once, save raw data, analyze and save figures. "
        "Wait up to 300 seconds; unfinished work continues by execution ID."
    ),
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
        SummaryParameter("frequency_mhz", "modules.readout.pulse_cfg.freq", "MHz"),
        SummaryParameter("ro_frequency_mhz", "modules.readout.ro_cfg.ro_freq", "MHz"),
        SummaryParameter("readout_length_us", "modules.readout.ro_cfg.ro_length", "us"),
        SummaryParameter(
            "trigger_offset_us", "modules.readout.ro_cfg.trig_offset", "us"
        ),
        SummaryParameter("rounds", "rounds"),
        SummaryParameter("readout_ref", "modules.readout"),
        SummaryParameter("use_reset", "modules.reset"),
        SummaryParameter("init_pulse_ref", "modules.init_pulse"),
    ),
    summary_estimates=(SummaryEstimate("predict_offset", "predict_offset", unit="us"),),
)
