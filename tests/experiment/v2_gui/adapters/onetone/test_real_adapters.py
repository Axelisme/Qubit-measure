from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.experiment.v2.onetone.flux_dep import FluxDepCfg
from zcu_tools.experiment.v2.onetone.freq import FreqCfg
from zcu_tools.experiment.v2.onetone.power_dep import PowerDepCfg
from zcu_tools.experiment.v2_gui.adapters.onetone.flux_dep import (
    OneToneFluxDepAdapter,
    OneToneFluxDepRunResult,
)
from zcu_tools.experiment.v2_gui.adapters.onetone.freq import OneToneFreqAdapter
from zcu_tools.experiment.v2_gui.adapters.onetone.power_dep import (
    OneTonePowerDepAdapter,
    OneTonePowerDepRunResult,
)
from zcu_tools.gui.adapter import AnalyzeRequest, CfgSchema, RunRequest
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
    return ctx


def _make_req(ml: MagicMock | None = None, *, with_soc: bool = False) -> RunRequest:
    return RunRequest(
        md=MagicMock(),
        ml=ml or _make_ml(),
        soc=MagicMock() if with_soc else None,
        soccfg=MagicMock() if with_soc else None,
    )


def _lower(schema: CfgSchema, req: RunRequest) -> dict[str, object]:
    return schema.to_raw_dict(None, req.ml)


def test_onetone_freq_build_exp_cfg_delegates_to_ml_make_cfg() -> None:
    ml = _make_ml()
    adapter = OneToneFreqAdapter()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    assert set(raw) == {"modules", "reps", "rounds", "relax_delay", "sweep"}
    sweep = cast(dict[str, Any], raw["sweep"])
    modules = cast(dict[str, Any], raw["modules"])
    assert isinstance(sweep["freq"], SweepCfg)
    assert "readout" in modules
    assert "reset" not in modules
    readout = cast(dict[str, Any], modules["readout"])
    ro_cfg = cast(dict[str, Any], readout["ro_cfg"])
    assert "gen_ch" not in ro_cfg

    adapter.build_exp_cfg(raw, _make_req(ml))
    ml.make_cfg.assert_called_once_with(raw, FreqCfg)


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
    from zcu_tools.gui.adapter import CfgSectionValue, EvalValue, SweepValue

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
    from zcu_tools.gui.adapter import CfgSectionValue, EvalValue, SweepValue

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
    from zcu_tools.gui.adapter import (
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
    from zcu_tools.gui.adapter import (
        CfgSectionValue,
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
    assert isinstance(pulse_cfg.fields["freq"], EvalValue)
    assert isinstance(ro_cfg.fields["ro_freq"], EvalValue)
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
    from zcu_tools.gui.adapter import CfgSectionValue, ModuleRefValue
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
