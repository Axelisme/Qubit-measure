"""Observable contracts for injected module/waveform materialization."""

import numpy as np
import pytest
from zcu_tools.cfg_model import (
    ArbWaveformCfg,
    CfgInputMap,
    CfgValidationContext,
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    FlatTopWaveformSpec,
    GaussWaveformCfg,
    ModuleCfgFactory,
    PulseSpec,
    TwoPulseResetCfg,
    TwoPulseResetSpec,
    WaveformCfgFactory,
)

from ._support import AssetInfo, Resolver


def test_inline_flat_top_materializes_isolated_ramp_without_dependencies():
    raw: CfgInputMap = {
        "style": "flat_top",
        "length": 2.0,
        "raise_waveform": {"style": "gauss", "length": 0.5, "sigma": 0.1},
    }

    cfg = WaveformCfgFactory.from_raw(raw)

    assert isinstance(cfg, FlatTopWaveformCfg)
    assert cfg.length == 2.0
    assert cfg.raise_waveform.length == 0.5
    cfg.raise_waveform.set_param("length", 1.0)
    assert raw["raise_waveform"] == {"style": "gauss", "length": 0.5, "sigma": 0.1}


def test_arb_materialization_binds_current_asset_reads_without_serializing_provider():
    assets = AssetInfo()
    raw: CfgInputMap = {"style": "arb", "data": "shape"}
    cfg = WaveformCfgFactory.from_raw(raw, context=CfgValidationContext(assets=assets))

    assert isinstance(cfg, ArbWaveformCfg)
    assert cfg.length == 2.5
    assert cfg.to_dict() == raw
    assert cfg.asset_samples().qdata is None

    assets.current_duration = 3.0
    assert cfg.length == 3.0
    np.testing.assert_array_equal(cfg.asset_samples().time, [0.0, 3.0])


@pytest.mark.parametrize(
    "raw",
    [
        {"style": "arb", "data": "shape"},
        {
            "style": "flat_top",
            "length": 1.0,
            "raise_waveform": {"style": "arb", "data": "shape"},
        },
    ],
)
def test_arb_materialization_requires_explicit_assets(raw: CfgInputMap):
    with pytest.raises(ValueError, match="requires assets"):
        WaveformCfgFactory.from_raw(raw)


def test_runtime_ramp_child_rebinds_assets_and_does_not_mutate_input():
    old_assets = AssetInfo()
    old = WaveformCfgFactory.from_raw(
        {"style": "arb", "data": "shape"},
        context=CfgValidationContext(assets=old_assets),
    )
    assets = AssetInfo()
    assets.current_duration = 7.0
    raw: CfgInputMap = {"style": "flat_top", "length": 1.0, "raise_waveform": old}

    cfg = WaveformCfgFactory.from_raw(raw, context=CfgValidationContext(assets=assets))

    assert isinstance(cfg, FlatTopWaveformCfg)
    assert cfg.raise_waveform.length == 7.0
    assert isinstance(old, ArbWaveformCfg)
    assert old.length == 2.5


def test_waveform_key_materializes_nested_ramp_key_from_fixed_specs():
    resolver = Resolver()
    resolver.waveforms["waveforms/envelope"] = FlatTopWaveformSpec(
        length=2.0, raise_waveform="waveforms/ramp"
    )
    ramp = GaussWaveformCfg(length=0.5, sigma=0.1)
    resolver.waveforms["waveforms/ramp"] = ramp

    cfg = WaveformCfgFactory.from_raw(
        "waveforms/envelope", context=CfgValidationContext(modules=resolver)
    )

    assert isinstance(cfg, FlatTopWaveformCfg)
    assert cfg.length == 2.0
    assert cfg.raise_waveform.length == 0.5
    cfg.raise_waveform.set_param("length", 1.0)
    assert ramp.length == 0.5
    assert (
        resolver.waveforms["waveforms/envelope"].to_dict()["raise_waveform"]
        == "waveforms/ramp"
    )


def test_reset_materializes_shared_pulse_key_and_nested_waveform_without_aliases():
    resolver = Resolver()
    resolver.modules["reset/two"] = TwoPulseResetSpec(
        pulse1_cfg="pulse/drive", pulse2_cfg="pulse/drive"
    )
    resolver.modules["pulse/drive"] = PulseSpec(
        ch=2, nqz=1, freq=4000.0, gain=0.2, waveform="waveforms/ramp"
    )
    resolver.waveforms["waveforms/ramp"] = GaussWaveformCfg(length=0.5, sigma=0.1)

    cfg = ModuleCfgFactory.from_raw(
        "reset/two", context=CfgValidationContext(modules=resolver)
    )

    assert isinstance(cfg, TwoPulseResetCfg)
    assert cfg.pulse1_cfg.ch == cfg.pulse2_cfg.ch == 2
    assert cfg.pulse1_cfg.waveform.length == 0.5
    cfg.pulse1_cfg.set_param("length", 1.0)
    assert cfg.pulse2_cfg.waveform.length == 0.5
    assert resolver.waveforms["waveforms/ramp"].length == 0.5


@pytest.mark.parametrize(
    ("child", "target", "chain"),
    [
        ("waveforms/missing", None, "waveforms/envelope -> waveforms/missing"),
        (
            "waveforms/const",
            ConstWaveformCfg(length=0.5),
            "waveforms/envelope -> waveforms/const",
        ),
        ("waveforms/envelope", None, "waveforms/envelope -> waveforms/envelope"),
    ],
)
def test_waveform_resolution_failures_report_root_chain_and_expected_ramp(
    child: str,
    target: ConstWaveformCfg | None,
    chain: str,
):
    resolver = Resolver()
    resolver.waveforms["waveforms/envelope"] = FlatTopWaveformSpec(
        length=1.0, raise_waveform=child
    )
    if target is not None:
        resolver.waveforms[child] = target

    with pytest.raises(ValueError, match="expected ramp") as failure:
        WaveformCfgFactory.from_raw(
            "waveforms/envelope", context=CfgValidationContext(modules=resolver)
        )

    message = str(failure.value)
    assert "Root waveforms/envelope" in message
    assert f"chain {chain}" in message
    assert "expected ramp" in message


def test_missing_resolver_reports_root_key_and_expected_family():
    with pytest.raises(ValueError, match="Missing modules dependency") as failure:
        WaveformCfgFactory.from_raw("waveforms/envelope")

    message = str(failure.value)
    assert "Root waveforms/envelope" in message
    assert "chain waveforms/envelope" in message
    assert "expected waveform" in message
    assert "Missing modules dependency" in message


def test_module_resolution_keeps_chain_across_module_and_waveform():
    resolver = Resolver()
    resolver.modules["pulse/drive"] = PulseSpec(
        ch=0, nqz=1, freq=4000.0, gain=0.2, waveform="waveforms/envelope"
    )
    resolver.waveforms["waveforms/envelope"] = FlatTopWaveformSpec(
        length=1.0, raise_waveform="waveforms/missing"
    )

    with pytest.raises(ValueError, match="expected ramp") as failure:
        ModuleCfgFactory.from_raw(
            "pulse/drive", context=CfgValidationContext(modules=resolver)
        )

    message = str(failure.value)
    assert "Root pulse/drive" in message
    assert "chain pulse/drive -> waveforms/envelope -> waveforms/missing" in message
    assert "expected ramp" in message


def test_runtime_child_revalidates_fields_instead_of_trusting_prior_cfg():
    bad = GaussWaveformCfg.model_construct(length="invalid", sigma=0.1)
    raw: CfgInputMap = {"style": "flat_top", "length": 1.0, "raise_waveform": bad}

    with pytest.raises(ValueError, match="length"):
        WaveformCfgFactory.from_raw(raw)
