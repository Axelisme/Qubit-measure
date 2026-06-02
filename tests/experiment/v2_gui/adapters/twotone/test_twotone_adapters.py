from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2.twotone.fluxdep import FreqFluxCfg
from zcu_tools.experiment.v2.twotone.freq import FreqCfg
from zcu_tools.experiment.v2.twotone.power_dep import PowerCfg
from zcu_tools.experiment.v2.twotone.rabi.amp_rabi import AmpRabiCfg
from zcu_tools.experiment.v2.twotone.rabi.len_rabi import LenRabiCfg
from zcu_tools.experiment.v2.twotone.time_domain.t1 import T1Cfg
from zcu_tools.experiment.v2.twotone.time_domain.t2echo import T2EchoCfg
from zcu_tools.experiment.v2.twotone.time_domain.t2ramsey import T2RamseyCfg
from zcu_tools.experiment.v2_gui.adapters.twotone import (
    AmpRabiAdapter,
    FluxDepAdapter,
    FreqAdapter,
    LenRabiAdapter,
    PowerDepAdapter,
    T1Adapter,
    T2EchoAdapter,
    T2RamseyAdapter,
)
from zcu_tools.gui.adapter import CfgSchema, RunRequest
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
    ctx.qub_name = "Q1"
    return ctx


def _make_req(ml: MagicMock | None = None) -> RunRequest:
    return RunRequest(
        md=MagicMock(),
        ml=ml or _make_ml(),
        soc=None,
        soccfg=None,
    )


def _lower(schema: CfgSchema, req: RunRequest) -> dict[str, object]:
    return schema.to_raw_dict(None, req.ml)


@pytest.mark.parametrize(
    ("adapter", "cfg_model"),
    [
        (FreqAdapter(), FreqCfg),
        (AmpRabiAdapter(), AmpRabiCfg),
        (LenRabiAdapter(), LenRabiCfg),
        (T1Adapter(), T1Cfg),
        (T2RamseyAdapter(), T2RamseyCfg),
        (T2EchoAdapter(), T2EchoCfg),
    ],
)
def test_twotone_build_exp_cfg_delegates_to_ml_make_cfg(
    adapter: Any, cfg_model: type
) -> None:
    ml = _make_ml()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    assert "modules" in raw
    assert "sweep" in raw
    modules = cast(dict[str, Any], raw["modules"])
    assert "readout" in modules
    assert "reset" not in modules  # optional reset disabled by default in tests

    adapter.build_exp_cfg(raw, _make_req(ml))
    ml.make_cfg.assert_called_once_with(raw, cfg_model)


@pytest.mark.parametrize(
    ("adapter", "cfg_model"),
    [
        (PowerDepAdapter(), PowerCfg),
    ],
)
def test_twotone_2d_build_exp_cfg_delegates_to_ml_make_cfg(
    adapter: Any, cfg_model: type
) -> None:
    ml = _make_ml()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    assert "modules" in raw
    assert "sweep" in raw
    sweep = cast(dict[str, Any], raw["sweep"])
    assert isinstance(sweep["gain"], SweepCfg)
    assert isinstance(sweep["freq"], SweepCfg)

    adapter.build_exp_cfg(raw, _make_req(ml))
    assert ml.make_cfg.call_args.args[1] is cfg_model


def test_twotone_freq_sweep_contains_freq() -> None:
    ml = _make_ml()
    adapter = FreqAdapter()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    sweep = cast(dict[str, Any], raw["sweep"])
    assert isinstance(sweep["freq"], SweepCfg)


def test_flux_dep_build_exp_cfg_converts_device_section() -> None:
    ml = _make_ml()
    adapter = FluxDepAdapter()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    adapter.build_exp_cfg(raw, _make_req(ml))
    cfg_raw = ml.make_cfg.call_args.args[0]
    assert cfg_raw["dev"] == {"flux_yoko": {"label": "flux_dev"}}


def test_flux_dep_build_exp_cfg_delegates_to_make_cfg() -> None:
    ml = _make_ml()
    adapter = FluxDepAdapter()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    adapter.build_exp_cfg(raw, _make_req(ml))
    assert ml.make_cfg.call_args.args[1] is FreqFluxCfg


@pytest.mark.parametrize(
    "adapter",
    [
        FreqAdapter(),
        PowerDepAdapter(),
        FluxDepAdapter(),
        AmpRabiAdapter(),
        LenRabiAdapter(),
        T1Adapter(),
        T2RamseyAdapter(),
        T2EchoAdapter(),
    ],
)
def test_twotone_run_without_soc_fast_fails(adapter: Any) -> None:
    ml = _make_ml()
    schema = adapter.make_default_cfg(_make_ctx(ml))

    with pytest.raises(RuntimeError, match="soc is required"):
        adapter.run(_make_req(ml), schema)


def test_t2echo_modules_contain_both_pulses() -> None:
    ml = _make_ml()
    adapter = T2EchoAdapter()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    modules = cast(dict[str, Any], raw["modules"])
    assert "pi2_pulse" in modules
    assert "pi_pulse" in modules
    assert "readout" in modules


@pytest.mark.parametrize(
    "adapter", [FreqAdapter(), PowerDepAdapter(), FluxDepAdapter()]
)
def test_twotone_defaults_ignore_library_readout(adapter: Any) -> None:
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
                    "freq": 6100.0,
                    "gain": 0.2,
                },
                "ro_cfg": {
                    "ro_ch": 2,
                    "ro_freq": 6100.0,
                    "ro_length": 1.0,
                    "trig_offset": 0.5,
                },
            },
            ml=ml,
        )
    )
    schema = adapter.make_default_cfg(_make_ctx(cast(Any, ml)))
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    readout = modules.fields["readout"]
    assert isinstance(readout, ModuleRefValue)
    assert readout.chosen_key == "<Custom:Pulse Readout>"
