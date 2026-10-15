"""Lookback author workflow; cfg and completion ownership stay in the framework."""

from zcu_tools.mcp.measure.execution_reply import SummaryEstimate, SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    RecipeDefinition,
    RecipeGenerator,
    RecipeNeedsParameters,
    RecipeRun,
    RecipeSession,
    RecipeTab,
)

from .helpers import set_frequency


def _set_readout_timing(
    tab: RecipeTab,
    readout_length_us: float | None,
    trigger_offset_us: float | None,
) -> None:
    """Keep omitted key values and provenance; convert only for explicit edits."""
    for field, value, value_path in (
        (
            "modules.readout.module.ro_length",
            readout_length_us,
            ("ro_length",),
        ),
        (
            "modules.readout.module.trig_offset",
            trigger_offset_us,
            ("trig_offset",),
        ),
    ):
        if value is not None:
            tab.use_inline("modules.readout")
        if value is None and tab.read_field("modules.readout").mode == "key":
            tab.retain_module_source(
                "modules.readout", summary_field=field, value_path=value_path
            )
        else:
            tab.set(field, value, source="explicit")


def lookback(  # noqa: PLR0913 - Typed keywords preserve the public tool inputs.
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
    """Run once, save raw/Primary images, then ask before explicit writeback.

    target binds the new tab's primary role without changing GUI focus.
    Omission uses focus. Do not pass target when reusing an existing tab.

    session is driver-supplied. Names select observed libraries; nullable reset
    and init_pulse disable those optional modules. reuse_tab_id resets the named
    idle Lookback tab. Frequencies use MHz, lengths/offsets use us, and None retains
    GUI defaults or calibrated frequency. Missing frequency ends needs_parameters;
    invalid cfg/native failures throw at their author step. Cancel returns without
    later saves/analyses/writes; finish_early saves usable partial Run data.
    """
    tab = session.open_tab(
        "lookback",
        aliases=None if target is None else {"qubit": target},
        reuse=reuse_tab_id,
    )
    tab.use_library("modules.readout", readout_ref)
    if use_reset is None:
        tab.disable_library("modules.reset")
    else:
        tab.use_library("modules.reset", use_reset)
    if init_pulse_ref is None:
        tab.disable_library("modules.init_pulse")
    else:
        tab.use_library("modules.init_pulse", init_pulse_ref)
    missing = set_frequency(
        tab,
        "modules.readout",
        frequency_mhz,
        calibration="$resonator.freq",
        prefer_library=readout_ref is not None,
        required="frequency_mhz",
    )
    if missing is not None:
        raise RecipeNeedsParameters((missing,))
    _set_readout_timing(tab, readout_length_us, trigger_offset_us)
    tab.set("rounds", rounds, source="explicit")
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
        SummaryParameter("frequency_mhz", "modules.readout.module.freq", "MHz"),
        SummaryParameter("readout_length_us", "modules.readout.module.ro_length", "us"),
        SummaryParameter(
            "trigger_offset_us", "modules.readout.module.trig_offset", "us"
        ),
        SummaryParameter("rounds", "rounds"),
        SummaryParameter("readout_ref", "modules.readout"),
        SummaryParameter("use_reset", "modules.reset"),
        SummaryParameter("init_pulse_ref", "modules.init_pulse"),
    ),
    summary_estimates=(SummaryEstimate("predict_offset", "predict_offset", unit="us"),),
)
