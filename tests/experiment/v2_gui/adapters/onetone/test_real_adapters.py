from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.experiment.v2.onetone.flux_dep import FluxDepCfg
from zcu_tools.experiment.v2.onetone.freq import FreqCfg, FreqResult
from zcu_tools.experiment.v2.onetone.power_dep import PowerDepCfg
from zcu_tools.experiment.v2_gui.adapters.onetone.flux_dep import (
    OneToneFluxDepAdapter,
)
from zcu_tools.experiment.v2_gui.adapters.onetone.freq import (
    OneToneFreqAdapter,
    OneToneFreqAnalyzeResult,
)
from zcu_tools.experiment.v2_gui.adapters.onetone.power_dep import (
    OneTonePowerDepAdapter,
)
from zcu_tools.gui.app.main.adapter import CfgSchema, RunRequest, WritebackRequest
from zcu_tools.gui.session.value_lookup import EmptyValueLookup, ValueKey, ValueRegistry
from zcu_tools.meta_tool import MetaDict
from zcu_tools.program.v2 import ModuleCfgFactory, SweepCfg


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
    ctx.res_name = "R1"
    ctx.values = EmptyValueLookup()
    return ctx


def _make_req(
    ml: MagicMock | None = None,
    *,
    with_soc: bool = False,
    md: Any | None = None,
) -> RunRequest:
    return RunRequest(
        md=md if md is not None else MagicMock(),
        ml=ml or _make_ml(),
        soc=MagicMock() if with_soc else None,
        soccfg=MagicMock() if with_soc else None,
    )


def _lower(schema: CfgSchema, req: RunRequest) -> dict[str, object]:
    return schema.to_raw_dict(None, req.ml)


def test_onetone_freq_build_exp_cfg_uses_cfg_assembler() -> None:
    ml = _make_ml()
    adapter = OneToneFreqAdapter()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    assert set(raw) == {
        "modules",
        "reps",
        "rounds",
        "relax_delay",
        "sampling_mode",
        "sweep",
    }
    assert raw["sampling_mode"] == "linear"
    sweep = cast(dict[str, Any], raw["sweep"])
    modules = cast(dict[str, Any], raw["modules"])
    assert isinstance(sweep["freq"], SweepCfg)
    assert "readout" in modules
    assert "reset" not in modules
    readout = cast(dict[str, Any], modules["readout"])
    ro_cfg = cast(dict[str, Any], readout["ro_cfg"])
    assert "gen_ch" not in ro_cfg

    cfg = adapter.build_exp_cfg(raw, _make_req(ml))
    assert isinstance(cfg, FreqCfg)
    assert cfg.sampling_mode == "linear"
    assert cfg.homophasal is None


def test_onetone_freq_homophasal_build_exp_cfg_injects_md_fit_params() -> None:
    ml = _make_ml()
    md = MetaDict()
    md.r_f = 6000.0
    md.rf_w = 20.0
    md.theta0 = 0.35
    adapter = OneToneFreqAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    schema.value.with_field("sampling_mode", "homophasal")
    raw = _lower(schema, _make_req(ml, md=md))

    cfg = adapter.build_exp_cfg(raw, _make_req(ml, md=md))

    assert isinstance(cfg, FreqCfg)
    assert cfg.sampling_mode == "homophasal"
    assert cfg.homophasal is not None
    assert cfg.homophasal.r_f == pytest.approx(6000.0)
    assert cfg.homophasal.rf_w == pytest.approx(20.0)
    assert cfg.homophasal.theta0 == pytest.approx(0.35)


def test_onetone_freq_homophasal_preflight_requires_fit_params() -> None:
    ml = _make_ml()
    adapter = OneToneFreqAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    schema.value.with_field("sampling_mode", "homophasal")
    raw = _lower(schema, _make_req(ml, md=MetaDict()))

    with pytest.raises(
        ValueError,
        match="homophasal sampling requires numeric MetaDict keys",
    ):
        adapter.validate_run_request(_make_req(ml, md=MetaDict()), raw)


def test_onetone_freq_writeback_includes_theta0() -> None:
    adapter = OneToneFreqAdapter()
    analyze_result = OneToneFreqAnalyzeResult(
        freq=6000.0,
        fwhm=20.0,
        params={"theta0": 0.35},
        figure=MagicMock(),
    )
    req = WritebackRequest(
        run_result=FreqResult(
            freqs=np.asarray([], dtype=np.float64),
            signals=np.asarray([], dtype=np.complex128),
        ),
        analyze_result=analyze_result,
        ctx=_make_ctx(_make_ml()),
    )

    items = adapter.get_writeback_items(req)

    assert [item.target_name for item in items] == ["r_f", "rf_w", "theta0"]


@pytest.mark.parametrize(
    ("adapter", "cfg_model"),
    [
        (OneTonePowerDepAdapter(), PowerDepCfg),
        (OneToneFluxDepAdapter(), FluxDepCfg),
    ],
)
def test_onetone_2d_build_exp_cfg_delegates_to_ml_make_cfg(adapter, cfg_model) -> None:
    ml = _make_ml()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    assert "modules" in raw
    assert "sweep" in raw
    modules = cast(dict[str, Any], raw["modules"])
    assert "readout" in modules

    adapter.build_exp_cfg(raw, _make_req(ml))
    assert ml.make_cfg.call_args.args[1] is cfg_model


def test_onetone_power_dep_default_sweep_freq_uses_eval_value() -> None:
    from zcu_tools.gui.app.main.adapter import CfgSectionValue, EvalValue, SweepValue

    ctx = _make_ctx(_make_ml())
    ctx.md.r_f = 6100.0
    ctx.md.rf_w = 25.0
    schema = OneTonePowerDepAdapter().make_default_cfg(ctx)
    sweep = schema.value.fields["sweep"]
    assert isinstance(sweep, CfgSectionValue)
    freq_sweep = sweep.fields["freq"]
    assert isinstance(freq_sweep, SweepValue)
    assert isinstance(freq_sweep.start, EvalValue)
    assert isinstance(freq_sweep.stop, EvalValue)
    assert freq_sweep.start.expr == "r_f - 1.5 * rf_w"
    assert freq_sweep.stop.expr == "r_f + 1.5 * rf_w"


def test_onetone_flux_dep_default_sweep_freq_uses_eval_value() -> None:
    from zcu_tools.gui.app.main.adapter import CfgSectionValue, EvalValue, SweepValue

    ctx = _make_ctx(_make_ml())
    ctx.md.r_f = 6100.0
    ctx.md.rf_w = 25.0
    schema = OneToneFluxDepAdapter().make_default_cfg(ctx)
    sweep = schema.value.fields["sweep"]
    assert isinstance(sweep, CfgSectionValue)
    freq_sweep = sweep.fields["freq"]
    assert isinstance(freq_sweep, SweepValue)
    assert isinstance(freq_sweep.start, EvalValue)
    assert isinstance(freq_sweep.stop, EvalValue)
    assert freq_sweep.start.expr == "r_f - rf_w"
    assert freq_sweep.stop.expr == "r_f + rf_w"


def test_onetone_flux_dep_default_flux_device_uses_named_device_source() -> None:
    from zcu_tools.gui.app.main.adapter import CfgSectionValue, DirectValue

    registry = ValueRegistry()
    registry.register(
        ValueKey("device.flux.name", str),
        lambda: "flux",
        owner="test",
    )
    ctx = _make_ctx(_make_ml())
    ctx.values = registry

    schema = OneToneFluxDepAdapter().make_default_cfg(ctx)

    dev = schema.value.fields["dev"]
    assert isinstance(dev, CfgSectionValue)
    assert dev.fields["flux_dev"] == DirectValue("flux")


def test_power_dep_build_exp_cfg_strips_earlystop_snr() -> None:
    ml = _make_ml()
    adapter = OneTonePowerDepAdapter()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    assert raw["earlystop_snr"] == 0.0
    adapter.build_exp_cfg(raw, _make_req(ml))
    assert "earlystop_snr" not in ml.make_cfg.call_args.args[0]


def test_flux_dep_build_exp_cfg_converts_device_section() -> None:
    ml = _make_ml()
    adapter = OneToneFluxDepAdapter()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    adapter.build_exp_cfg(raw, _make_req(ml))
    cfg_raw = ml.make_cfg.call_args.args[0]
    assert cfg_raw["dev"] == {"flux_yoko": {"label": "flux_dev"}}


@pytest.mark.parametrize(
    "adapter",
    [OneToneFreqAdapter(), OneTonePowerDepAdapter(), OneToneFluxDepAdapter()],
)
def test_real_onetone_run_without_soc_fast_fails(adapter) -> None:
    ml = _make_ml()
    schema = adapter.make_default_cfg(_make_ctx(ml))

    with pytest.raises(RuntimeError, match="soc is required"):
        adapter.run(_make_req(ml), schema)


def test_onetone_freq_default_fallback_uses_direct_values_without_md_keys() -> None:
    from zcu_tools.gui.app.main.adapter import (
        CfgSectionValue,
        DirectValue,
        ModuleRefValue,
        SweepValue,
    )

    adapter = OneToneFreqAdapter()
    schema = adapter.make_default_cfg(_make_ctx(_make_ml()))
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    readout = modules.fields["readout"]
    assert isinstance(readout, ModuleRefValue)
    readout_val = readout.value
    pulse_cfg = readout_val.fields["pulse_cfg"]
    ro_cfg = readout_val.fields["ro_cfg"]
    assert isinstance(pulse_cfg, CfgSectionValue)
    assert isinstance(ro_cfg, CfgSectionValue)
    pulse_freq = pulse_cfg.fields["freq"]
    ro_freq = ro_cfg.fields["ro_freq"]
    assert isinstance(pulse_freq, DirectValue)
    assert isinstance(ro_freq, DirectValue)

    pulse_ch = pulse_cfg.fields["ch"]
    ro_ch = ro_cfg.fields["ro_ch"]
    trig_offset = ro_cfg.fields["trig_offset"]
    assert isinstance(pulse_ch, DirectValue)
    assert isinstance(ro_ch, DirectValue)
    assert isinstance(trig_offset, DirectValue)

    sweep = schema.value.fields["sweep"]
    assert isinstance(sweep, CfgSectionValue)
    freq_sweep = sweep.fields["freq"]
    assert isinstance(freq_sweep, SweepValue)
    assert isinstance(freq_sweep.start, float)
    assert isinstance(freq_sweep.stop, float)


def test_onetone_freq_default_uses_eval_when_md_keys_exist() -> None:
    from zcu_tools.gui.app.main.adapter import (
        CfgSectionValue,
        DirectValue,
        EvalValue,
        ModuleRefValue,
        SweepValue,
    )

    ctx = _make_ctx(_make_ml())
    ctx.md.r_f = 6100.0
    ctx.md.rf_w = 25.0
    ctx.md.res_ch = 3
    ctx.md.ro_ch = 7
    ctx.md.timeFly = 0.8
    schema = OneToneFreqAdapter().make_default_cfg(ctx)
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    readout = modules.fields["readout"]
    assert isinstance(readout, ModuleRefValue)
    readout_val = readout.value
    pulse_cfg = readout_val.fields["pulse_cfg"]
    ro_cfg = readout_val.fields["ro_cfg"]
    assert isinstance(pulse_cfg, CfgSectionValue)
    assert isinstance(ro_cfg, CfgSectionValue)
    # freq / ro_freq are lock_literal'd to 0.0 (the sweep axis owns frequency),
    # so the value carries the locked literal — asserting they track r_f would be
    # asserting on a field that is, by design, not user-meaningful for a freq
    # sweep. Only the non-locked md-derived fields track md here.
    assert pulse_cfg.fields["freq"] == DirectValue(0.0)
    assert ro_cfg.fields["ro_freq"] == DirectValue(0.0)
    assert isinstance(pulse_cfg.fields["ch"], EvalValue)
    assert isinstance(ro_cfg.fields["ro_ch"], EvalValue)
    assert isinstance(ro_cfg.fields["trig_offset"], EvalValue)
    sweep = schema.value.fields["sweep"]
    assert isinstance(sweep, CfgSectionValue)
    freq_sweep = sweep.fields["freq"]
    assert isinstance(freq_sweep, SweepValue)
    assert isinstance(freq_sweep.start, EvalValue)
    assert isinstance(freq_sweep.stop, EvalValue)


def test_onetone_freq_default_ignores_library_readout() -> None:
    from zcu_tools.gui.app.main.adapter import CfgSectionValue, ModuleRefValue
    from zcu_tools.meta_tool import ModuleLibrary

    ml = ModuleLibrary()
    ml.register_module(
        readout_dpm=ModuleCfgFactory.from_raw(
            {
                "type": "readout/pulse",
                "pulse_cfg": {
                    "waveform": {"style": "const", "length": 1.0},
                    "ch": 1,
                    "nqz": 2,
                    "freq": 6111.0,
                    "gain": 0.2,
                },
                "ro_cfg": {
                    "ro_ch": 2,
                    "ro_freq": 6111.0,
                    "ro_length": 1.0,
                    "trig_offset": 0.5,
                },
            },
            ml=ml,
        )
    )
    schema = OneToneFreqAdapter().make_default_cfg(_make_ctx(cast(Any, ml)))
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    readout = modules.fields["readout"]
    assert isinstance(readout, ModuleRefValue)
    assert readout.chosen_key == "<Custom:Pulse Readout>"
