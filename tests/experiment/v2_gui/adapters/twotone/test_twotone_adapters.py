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
from zcu_tools.gui.app.main.adapter import CfgSchema, DirectValue, RunRequest
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
        (T2RamseyAdapter(), T2RamseyCfg),
        (T2EchoAdapter(), T2EchoCfg),
    ],
)
def test_t2_detune_in_cfg_spec_default_zero(adapter: Any, cfg_model: type) -> None:
    # detune is a run-only knob carried in the cfg spec (ADR-0011: the lowered
    # value tree is complete and validates) and defaults to 0.0.
    ml = _make_ml()
    schema = adapter.make_default_cfg(_make_ctx(ml))  # validate() runs here
    raw = _lower(schema, _make_req(ml))
    assert raw["detune"] == 0.0


@pytest.mark.parametrize(
    ("adapter", "cfg_model"),
    [
        (T2RamseyAdapter(), T2RamseyCfg),
        (T2EchoAdapter(), T2EchoCfg),
    ],
)
def test_t2_build_exp_cfg_strips_detune(adapter: Any, cfg_model: type) -> None:
    # detune is not part of the lowered ExpCfg, so build_exp_cfg must pop it
    # before ml.make_cfg (mirrors ro_optimize/auto's num_points).
    ml = _make_ml()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))
    assert "detune" in raw  # present in the lowered cfg

    adapter.build_exp_cfg(raw, _make_req(ml))
    passed_raw = ml.make_cfg.call_args.args[0]
    assert "detune" not in passed_raw  # stripped before make_cfg
    assert ml.make_cfg.call_args.args[1] is cfg_model


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


@pytest.mark.parametrize(
    "adapter_cls",
    [T2RamseyAdapter, T2EchoAdapter],
)
def test_t2_run_unpacks_tuple_and_forwards_detune(adapter_cls: type) -> None:
    # Domain T2*/T2echo run returns (result, true_detune); the GUI run override
    # must forward the cfg detune knob as a kwarg and return only the bare result.
    sentinel_result = object()
    captured: dict[str, Any] = {}

    class FakeExp:
        def run(self, soc, soccfg, cfg, *, detune: float):
            captured["detune"] = detune
            return sentinel_result, 0.123  # (result, true_detune)

    adapter = adapter_cls()
    adapter.exp_cls = FakeExp  # type: ignore[assignment]

    ml = _make_ml()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    # set a non-zero detune in the draft to prove it reaches the kwarg
    schema.value.fields["detune"] = DirectValue(1.5)

    req = RunRequest(md=MagicMock(), ml=ml, soc=MagicMock(), soccfg=MagicMock())
    result = adapter.run(req, schema)

    assert result is sentinel_result  # tuple unpacked, only result returned
    assert captured["detune"] == 1.5  # cfg detune forwarded to domain kwarg


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
def test_twotone_readout_prefers_library_module(adapter: Any) -> None:
    """When ml contains readout_dpm, the readout default links to it (not inline)."""
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
    # Should be a library link, not an inline custom readout.
    assert readout.chosen_key == "readout_dpm"


@pytest.mark.parametrize(
    "adapter", [FreqAdapter(), PowerDepAdapter(), FluxDepAdapter()]
)
def test_twotone_readout_fallback_inline_when_ml_empty(adapter: Any) -> None:
    """When ml has no calibrated readout module, fall back to inline pulse readout."""
    from zcu_tools.gui.app.main.adapter import CfgSectionValue, ModuleRefValue
    from zcu_tools.meta_tool import ModuleLibrary

    ml = ModuleLibrary()  # empty — no readout_dpm / readout_rf
    schema = adapter.make_default_cfg(_make_ctx(cast(Any, ml)))
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    readout = modules.fields["readout"]
    assert isinstance(readout, ModuleRefValue)
    assert readout.chosen_key == "<Custom:Pulse Readout>"
