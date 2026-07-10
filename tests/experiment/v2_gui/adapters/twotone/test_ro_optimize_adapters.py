from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2.twotone.ro_optimize.auto_optimize import AutoOptCfg
from zcu_tools.experiment.v2.twotone.ro_optimize.freq import FreqCfg
from zcu_tools.experiment.v2.twotone.ro_optimize.freq_gain import FreqGainCfg
from zcu_tools.experiment.v2.twotone.ro_optimize.length import LengthCfg
from zcu_tools.experiment.v2.twotone.ro_optimize.power import PowerCfg
from zcu_tools.experiment.v2_gui.adapters.shared import READOUT_DPM_PULSE_TAIL_US
from zcu_tools.experiment.v2_gui.adapters.twotone.ro_optimize import (
    RoOptAutoAdapter,
    RoOptFreqAdapter,
    RoOptFreqGainAdapter,
    RoOptLengthAdapter,
    RoOptPowerAdapter,
)
from zcu_tools.gui.app.main.adapter import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    LiteralSpec,
    MetaDictWriteback,
    ModuleWriteback,
    ReferenceSpec,
    ReferenceValue,
    RunRequest,
    WritebackRequest,
    describe_analyze_params,
)
from zcu_tools.gui.app.main.adapter import (
    SweepValue as GuiSweepValue,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.program.v2 import DirectReadoutCfg, PulseCfg, PulseReadoutCfg, SweepCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


def _make_ml() -> MagicMock:
    ml = MagicMock()
    ml.modules = {}
    ml.waveforms = {}
    ml.make_cfg.return_value = MagicMock()
    return ml


def _make_ctx(ml: MagicMock | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.ml = ml or _make_ml()
    ctx.md = MetaDict()
    ctx.qub_name = "Q1"
    return ctx


def _make_req(ml: MagicMock | None = None) -> RunRequest:
    return RunRequest(md=MagicMock(), ml=ml or _make_ml(), soc=None, soccfg=None)


def _lower(schema: CfgSchema, req: RunRequest) -> dict[str, object]:
    return schema_to_raw_dict(schema, None, req.ml)


def _ctx_with_md(**md_values: float) -> MagicMock:
    ctx = MagicMock()
    md = MetaDict()
    for key, value in md_values.items():
        setattr(md, key, value)
    ctx.md = md
    return ctx


def _make_pulse_readout(
    *, freq: float = 6000.0, gain: float = 0.05, length: float = 1.0
) -> PulseReadoutCfg:
    return PulseReadoutCfg(
        pulse_cfg=PulseCfg(
            type="pulse",
            waveform=ConstWaveformCfg(
                style="const", length=length + READOUT_DPM_PULSE_TAIL_US
            ),
            ch=0,
            nqz=1,
            freq=freq,
            gain=gain,
        ),
        ro_cfg=DirectReadoutCfg(
            type="readout/direct",
            ro_ch=0,
            ro_length=length,
            ro_freq=freq,
            trig_offset=0.0,
            gen_ch=0,
        ),
    )


def _snapshot_with_readout(readout: object) -> Any:
    modules = MagicMock()
    modules.readout = readout
    cfg = MagicMock()
    cfg.modules = modules
    return cfg


def _analysis_result(**values: float) -> MagicMock:
    result = MagicMock()
    for key, value in values.items():
        setattr(result, key, value)
    return result


def _writeback_request(
    *,
    analyze_result: object,
    cfg_snapshot: object | None,
    ctx: object | None = None,
) -> WritebackRequest[Any, Any]:
    run_result = MagicMock()
    run_result.cfg_snapshot = cfg_snapshot
    return WritebackRequest(
        run_result=run_result,
        analyze_result=analyze_result,
        ctx=cast(Any, ctx if ctx is not None else _ctx_with_md()),
    )


def _module_item(items: Sequence[object]) -> ModuleWriteback | None:
    module_items = [it for it in items if isinstance(it, ModuleWriteback)]
    if not module_items:
        return None
    assert len(module_items) == 1
    return module_items[0]


def _direct_float(value: object) -> float:
    assert isinstance(value, DirectValue)
    direct = cast(DirectValue, value)
    assert isinstance(direct.value, (int, float))
    return float(direct.value)


def _assert_readout_dpm_schema(
    item: ModuleWriteback, *, freq: float, gain: float, length: float
) -> None:
    assert item.target_name == "readout_dpm"
    assert item.description == "Optimized readout (DPM)"
    assert item.role_id == "readout_dpm"
    assert isinstance(item.edit_schema, CfgSchema)

    value = item.edit_schema.value
    assert value.fields["type"] == DirectValue("readout/pulse")
    pulse_cfg = value.fields["pulse_cfg"]
    ro_cfg = value.fields["ro_cfg"]
    assert isinstance(pulse_cfg, CfgSectionValue)
    assert isinstance(ro_cfg, CfgSectionValue)
    assert _direct_float(pulse_cfg.fields["freq"]) == pytest.approx(freq)
    assert _direct_float(ro_cfg.fields["ro_freq"]) == pytest.approx(freq)
    assert _direct_float(pulse_cfg.fields["gain"]) == pytest.approx(gain)
    waveform = pulse_cfg.fields["waveform"]
    assert isinstance(waveform, ReferenceValue)
    assert _direct_float(waveform.value.fields["length"]) == pytest.approx(
        length + READOUT_DPM_PULSE_TAIL_US
    )
    assert _direct_float(ro_cfg.fields["ro_length"]) == pytest.approx(length)

    raw = schema_to_raw_dict(item.edit_schema, MetaDict(), ModuleLibrary())
    assert raw["type"] == "readout/pulse"
    raw_pulse = cast(dict[str, Any], raw["pulse_cfg"])
    raw_ro = cast(dict[str, Any], raw["ro_cfg"])
    raw_waveform = cast(dict[str, Any], raw_pulse["waveform"])
    assert raw_pulse["freq"] == pytest.approx(freq)
    assert raw_pulse["gain"] == pytest.approx(gain)
    assert raw_waveform["length"] == pytest.approx(length + READOUT_DPM_PULSE_TAIL_US)
    assert raw_ro["ro_freq"] == pytest.approx(freq)
    assert raw_ro["ro_length"] == pytest.approx(length)


@pytest.mark.parametrize(
    ("adapter", "cfg_model", "sweep_key"),
    [
        (RoOptFreqAdapter(), FreqCfg, "freq"),
        (RoOptPowerAdapter(), PowerCfg, "gain"),
        (RoOptLengthAdapter(), LengthCfg, "length"),
    ],
)
def test_ro_opt_1d_build_exp_cfg_delegates(
    adapter: Any, cfg_model: type, sweep_key: str
) -> None:
    ml = _make_ml()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    assert "modules" in raw
    assert raw["skew_penalty"] == 0.0
    modules = cast(dict[str, Any], raw["modules"])
    assert "readout" in modules
    assert "qub_pulse" in modules
    sweep = cast(dict[str, Any], raw["sweep"])
    assert isinstance(sweep[sweep_key], SweepCfg)

    cfg = adapter.build_exp_cfg(raw, _make_req(ml))
    assert isinstance(cfg, cfg_model)


def test_ro_opt_freq_gain_build_exp_cfg_delegates() -> None:
    ml = _make_ml()
    adapter = RoOptFreqGainAdapter()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    sweep = cast(dict[str, Any], raw["sweep"])
    assert isinstance(sweep["freq"], SweepCfg)
    assert isinstance(sweep["gain"], SweepCfg)
    assert raw["skew_penalty"] == 0.0

    cfg = adapter.build_exp_cfg(raw, _make_req(ml))
    assert isinstance(cfg, FreqGainCfg)


def test_ro_opt_auto_pops_num_points_before_make_cfg() -> None:
    # num_points is a GUI-only runtime arg (like onetone/power_dep's
    # earlystop_snr): it lives in the cfg spec but must be stripped before
    # ml.make_cfg, then passed to AutoOptExp().run(num_points=...).
    ml = _make_ml()
    adapter = RoOptAutoAdapter()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    assert "num_points" in raw  # present in the lowered cfg
    assert raw["skew_penalty"] == 0.0
    sweep = cast(dict[str, Any], raw["sweep"])
    assert isinstance(sweep["freq"], SweepCfg)
    assert isinstance(sweep["gain"], SweepCfg)
    assert isinstance(sweep["length"], SweepCfg)

    adapter.build_exp_cfg(raw, _make_req(ml))
    passed_raw = ml.make_cfg.call_args.args[0]
    assert "num_points" not in passed_raw  # stripped before make_cfg
    assert passed_raw["skew_penalty"] == 0.0
    assert ml.make_cfg.call_args.args[1] is AutoOptCfg


def test_ro_opt_auto_num_points_validation() -> None:
    adapter = RoOptAutoAdapter()
    with pytest.raises(ValueError):
        adapter._num_points({"num_points": 0})  # non-positive
    with pytest.raises(ValueError):
        adapter._num_points({"num_points": "x"})  # not an int
    assert adapter._num_points({"num_points": 500}) == 500


def test_ro_opt_freq_gain_defaults_center_on_best_ro_freq_and_gain_notebook_range() -> (
    None
):
    ctx = _make_ctx(_make_ml())
    ctx.md.best_ro_freq = 6100.0
    ctx.md.r_f = 6000.0
    ctx.md.rf_w = 20.0

    schema = RoOptFreqGainAdapter().make_default_cfg(ctx)
    sweep = schema.value.fields["sweep"]
    assert isinstance(sweep, CfgSectionValue)
    freq = sweep.fields["freq"]
    gain = sweep.fields["gain"]
    assert isinstance(freq, GuiSweepValue)
    assert isinstance(freq.start, EvalValue)
    assert isinstance(freq.stop, EvalValue)
    assert freq.start.expr == "best_ro_freq - 0.5 * rf_w"
    assert freq.stop.expr == "best_ro_freq + 0.5 * rf_w"
    assert freq.expts == 31
    assert isinstance(gain, GuiSweepValue)
    assert gain.start == 0.0
    assert gain.stop == 0.2
    assert gain.expts == 31


def test_ro_opt_freq_gain_defaults_fall_back_to_r_f_center() -> None:
    ctx = _make_ctx(_make_ml())
    ctx.md.r_f = 6000.0
    ctx.md.rf_w = 20.0

    schema = RoOptFreqGainAdapter().make_default_cfg(ctx)
    sweep = schema.value.fields["sweep"]
    assert isinstance(sweep, CfgSectionValue)
    freq = sweep.fields["freq"]
    assert isinstance(freq, GuiSweepValue)
    assert isinstance(freq.start, EvalValue)
    assert isinstance(freq.stop, EvalValue)
    assert freq.start.expr == "r_f - 0.5 * rf_w"
    assert freq.stop.expr == "r_f + 0.5 * rf_w"


def test_ro_opt_length_relax_defaults_to_proper_relax() -> None:
    ctx = _make_ctx(_make_ml())
    ctx.md.t1 = 12.0

    schema = RoOptLengthAdapter().make_default_cfg(ctx)
    relax = schema.value.fields["relax_delay"]
    assert isinstance(relax, EvalValue)
    assert relax.expr == "5.0 * t1"


def test_ro_opt_auto_defaults_relax_num_points_and_gain_bounds() -> None:
    ctx = _make_ctx(_make_ml())
    ctx.md.t1 = 12.0

    schema = RoOptAutoAdapter().make_default_cfg(ctx)
    assert schema.value.fields["num_points"] == DirectValue(1001)
    relax = schema.value.fields["relax_delay"]
    assert isinstance(relax, EvalValue)
    assert relax.expr == "3.0 * t1"
    sweep = schema.value.fields["sweep"]
    assert isinstance(sweep, CfgSectionValue)
    gain = sweep.fields["gain"]
    assert isinstance(gain, GuiSweepValue)
    assert gain.start == 0.1
    assert gain.stop == 0.25
    assert gain.expts == 51


@pytest.mark.parametrize(
    "adapter",
    [
        RoOptFreqAdapter(),
        RoOptPowerAdapter(),
        RoOptLengthAdapter(),
        RoOptFreqGainAdapter(),
        RoOptAutoAdapter(),
    ],
)
def test_ro_opt_readout_spec_is_pulse_only(adapter: Any) -> None:
    modules = adapter.cfg_spec().fields["modules"]
    assert isinstance(modules, CfgSectionSpec)
    readout = modules.fields["readout"]
    assert isinstance(readout, ReferenceSpec)
    assert len(readout.allowed) == 1
    type_spec = readout.allowed[0].fields["type"]
    assert isinstance(type_spec, LiteralSpec)
    assert type_spec.value == "readout/pulse"


def test_ro_opt_length_analyze_params_use_duration_t0_name() -> None:
    params = describe_analyze_params(RoOptLengthAdapter.analyze_params_cls())

    duration_t0 = next(param for param in params if param["name"] == "duration_t0")
    assert duration_t0 == {
        "name": "duration_t0",
        "type": "float",
        "label": "Duration t0 (us)",
        "optional": True,
        "default": None,
    }
    assert all("penalty" not in param["label"].lower() for param in params)


@pytest.mark.parametrize(
    ("adapter", "wb_keys"),
    [
        (RoOptFreqAdapter(), ["best_ro_freq"]),
        (RoOptPowerAdapter(), ["best_ro_gain"]),
        (RoOptLengthAdapter(), ["best_ro_length"]),
        (RoOptFreqGainAdapter(), ["best_ro_freq", "best_ro_gain"]),
        (
            RoOptAutoAdapter(),
            ["best_ro_freq", "best_ro_gain", "best_ro_length"],
        ),
    ],
)
def test_ro_opt_writeback_targets(adapter: Any, wb_keys: list[str]) -> None:
    # Without a cfg snapshot, module writeback gracefully skips and existing md
    # scalar writeback remains unchanged.
    result = _analysis_result(best_freq=6100.0, best_gain=0.12, best_length=1.5)
    items = list(
        adapter.get_writeback_items(
            _writeback_request(analyze_result=result, cfg_snapshot=None)
        )
    )
    assert all(isinstance(it, MetaDictWriteback) for it in items)
    assert [it.target_name for it in items] == wb_keys


def test_ro_opt_auto_writeback_proposes_readout_dpm_from_current_results() -> None:
    result = _analysis_result(best_freq=6100.0, best_gain=0.12, best_length=1.5)
    items = list(
        RoOptAutoAdapter().get_writeback_items(
            _writeback_request(
                analyze_result=result,
                cfg_snapshot=_snapshot_with_readout(_make_pulse_readout()),
            )
        )
    )

    md_items = [it for it in items if isinstance(it, MetaDictWriteback)]
    assert [it.target_name for it in md_items] == [
        "best_ro_freq",
        "best_ro_gain",
        "best_ro_length",
    ]
    item = _module_item(items)
    assert item is not None
    _assert_readout_dpm_schema(item, freq=6100.0, gain=0.12, length=1.5)


@pytest.mark.parametrize(
    ("adapter", "result_values", "md_values", "expected"),
    [
        (
            RoOptFreqAdapter(),
            {"best_freq": 6101.0},
            {
                "best_ro_freq": 5800.0,
                "best_ro_gain": 0.11,
                "best_ro_length": 1.4,
            },
            {"freq": 6101.0, "gain": 0.11, "length": 1.4},
        ),
        (
            RoOptPowerAdapter(),
            {"best_gain": 0.12},
            {
                "best_ro_freq": 6100.0,
                "best_ro_gain": 0.03,
                "best_ro_length": 1.4,
            },
            {"freq": 6100.0, "gain": 0.12, "length": 1.4},
        ),
        (
            RoOptLengthAdapter(),
            {"best_length": 1.5},
            {
                "best_ro_freq": 6100.0,
                "best_ro_gain": 0.11,
                "best_ro_length": 0.7,
            },
            {"freq": 6100.0, "gain": 0.11, "length": 1.5},
        ),
        (
            RoOptFreqGainAdapter(),
            {"best_freq": 6102.0, "best_gain": 0.13},
            {
                "best_ro_freq": 5800.0,
                "best_ro_gain": 0.03,
                "best_ro_length": 1.4,
            },
            {"freq": 6102.0, "gain": 0.13, "length": 1.4},
        ),
    ],
)
def test_ro_opt_partial_adapters_complete_readout_dpm_from_current_result_and_md(
    adapter: Any,
    result_values: dict[str, float],
    md_values: dict[str, float],
    expected: dict[str, float],
) -> None:
    items = list(
        adapter.get_writeback_items(
            _writeback_request(
                analyze_result=_analysis_result(**result_values),
                cfg_snapshot=_snapshot_with_readout(_make_pulse_readout()),
                ctx=_ctx_with_md(**md_values),
            )
        )
    )

    item = _module_item(items)
    assert item is not None
    _assert_readout_dpm_schema(
        item,
        freq=expected["freq"],
        gain=expected["gain"],
        length=expected["length"],
    )


def test_ro_opt_readout_dpm_not_emitted_until_missing_values_available() -> None:
    result = _analysis_result(best_gain=0.12)
    items = list(
        RoOptPowerAdapter().get_writeback_items(
            _writeback_request(
                analyze_result=result,
                cfg_snapshot=_snapshot_with_readout(_make_pulse_readout()),
                ctx=_ctx_with_md(best_ro_freq=6100.0),
            )
        )
    )

    assert _module_item(items) is None
    assert [it.target_name for it in items] == ["best_ro_gain"]


@pytest.mark.parametrize(
    ("adapter", "result_values", "md_values"),
    [
        (
            RoOptFreqAdapter(),
            {"best_freq": float("nan")},
            {"best_ro_gain": 0.11, "best_ro_length": 1.4},
        ),
        (
            RoOptPowerAdapter(),
            {"best_gain": 0.12},
            {"best_ro_freq": float("inf"), "best_ro_length": 1.4},
        ),
    ],
)
def test_ro_opt_readout_dpm_not_emitted_for_nonfinite_resolved_values(
    adapter: Any,
    result_values: dict[str, float],
    md_values: dict[str, float],
) -> None:
    items = list(
        adapter.get_writeback_items(
            _writeback_request(
                analyze_result=_analysis_result(**result_values),
                cfg_snapshot=_snapshot_with_readout(_make_pulse_readout()),
                ctx=_ctx_with_md(**md_values),
            )
        )
    )

    assert _module_item(items) is None
