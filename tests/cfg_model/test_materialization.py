"""Observable contracts for injected module/waveform materialization."""

from types import MappingProxyType

import numpy as np
import pytest
from zcu_tools.cfg_model import (
    ArbWaveformCfg,
    CfgInputMap,
    CfgInputValue,
    CfgValidationContext,
    ConstWaveformCfg,
    DirectReadoutCfg,
    FlatTopWaveformCfg,
    FlatTopWaveformSpec,
    GaussWaveformCfg,
    ModuleCfgFactory,
    NoneResetCfg,
    PulseCfg,
    PulseResetCfg,
    PulseSpec,
    TwoPulseResetCfg,
    TwoPulseResetSpec,
    WaveformCfgFactory,
)

from ._support import AssetInfo, Resolver


def test_waveform_adapter_rejects_unknown_style():
    with pytest.raises(ValueError, match="union_tag_invalid") as failure:
        WaveformCfgFactory.from_raw({"style": "no_such_style", "length": 1.0})

    message = str(failure.value)
    assert "Root <inline>" in message
    assert "chain <inline>" in message
    assert "expected waveform" in message


def test_waveform_factory_string_ref_requires_resolver():
    with pytest.raises(ValueError, match="Missing modules dependency") as failure:
        WaveformCfgFactory.from_raw("stored_waveform")

    message = str(failure.value)
    assert "Root stored_waveform" in message
    assert "chain stored_waveform" in message
    assert "expected waveform" in message


def test_waveform_factory_string_ref_uses_typed_context():
    resolver = Resolver()
    resolver.waveforms["stored_waveform"] = ConstWaveformCfg(length=0.5)
    context = CfgValidationContext(modules=resolver)

    cfg = WaveformCfgFactory.from_raw("stored_waveform", context=context)

    assert isinstance(cfg, ConstWaveformCfg)
    assert cfg.length == 0.5
    cfg.set_param("length", 2.0)
    assert resolver.waveforms["stored_waveform"].length == 0.5
    assert WaveformCfgFactory.from_raw("stored_waveform", context=context).length == 0.5


def test_nested_waveform_ref_requires_resolver():
    raw: CfgInputMap = {
        "style": "flat_top",
        "length": 1.0,
        "raise_waveform": "named_waveform",
    }
    with pytest.raises(ValueError, match="Missing modules dependency") as failure:
        WaveformCfgFactory.from_raw(raw)

    message = str(failure.value)
    assert "Root <inline>" in message
    assert "chain named_waveform" in message
    assert "expected ramp" in message
    assert raw == {
        "style": "flat_top",
        "length": 1.0,
        "raise_waveform": "named_waveform",
    }


def test_waveform_factory_resolves_nested_raise_waveform_ref():
    resolver = Resolver()
    resolver.waveforms["gauss_rise"] = GaussWaveformCfg(length=0.2, sigma=0.05)
    raw: CfgInputMap = {
        "style": "flat_top",
        "length": 1.0,
        "raise_waveform": "gauss_rise",
    }

    cfg = WaveformCfgFactory.from_raw(
        raw, context=CfgValidationContext(modules=resolver)
    )

    assert isinstance(cfg, FlatTopWaveformCfg)
    assert cfg.length == 1.0
    assert isinstance(cfg.raise_waveform, GaussWaveformCfg)
    assert cfg.raise_waveform.length == 0.2
    assert cfg.raise_waveform.sigma == 0.05
    cfg.raise_waveform.set_param("only_length", 0.4)
    assert resolver.waveforms["gauss_rise"].length == 0.2
    assert raw == {"style": "flat_top", "length": 1.0, "raise_waveform": "gauss_rise"}


def test_waveform_adapter_dispatches_by_style():
    cfg = WaveformCfgFactory.from_raw({"style": "gauss", "length": 1.0, "sigma": 0.2})
    assert isinstance(cfg, GaussWaveformCfg)


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


@pytest.mark.parametrize("readonly_root", [False, True])
def test_waveform_factory_materializes_readonly_mappings_without_aliases(
    readonly_root: bool,
):
    ramp: dict[str, CfgInputValue] = {
        "style": "gauss",
        "length": 0.5,
        "sigma": 0.1,
    }
    fields: dict[str, CfgInputValue] = {
        "style": "flat_top",
        "length": 2.0,
        "raise_waveform": MappingProxyType(ramp),
    }
    raw: CfgInputMap = MappingProxyType(fields) if readonly_root else fields

    cfg = WaveformCfgFactory.from_raw(raw)

    assert isinstance(cfg, FlatTopWaveformCfg)
    assert cfg.length == 2.0
    assert cfg.raise_waveform.length == 0.5
    ramp["length"] = 0.75
    fields["length"] = 3.0
    assert cfg.length == 2.0
    assert cfg.raise_waveform.length == 0.5
    cfg.raise_waveform.set_param("length", 1.0)
    fresh = WaveformCfgFactory.from_raw(raw)
    assert isinstance(fresh, FlatTopWaveformCfg)
    assert fresh.length == 3.0
    assert fresh.raise_waveform.length == 0.75
    assert ramp["length"] == 0.75


@pytest.mark.parametrize("readonly_root", [False, True])
def test_module_factory_materializes_readonly_mappings_without_aliases(
    readonly_root: bool,
):
    waveform: dict[str, CfgInputValue] = {"style": "const", "length": 0.5}
    fields: dict[str, CfgInputValue] = {
        "type": "pulse",
        "ch": 2,
        "nqz": 1,
        "freq": 4000.0,
        "gain": 0.2,
        "waveform": MappingProxyType(waveform),
    }
    raw: CfgInputMap = MappingProxyType(fields) if readonly_root else fields

    cfg = ModuleCfgFactory.from_raw(raw)

    assert isinstance(cfg, PulseCfg)
    assert cfg.freq == 4000.0
    assert cfg.gain == 0.2
    assert cfg.waveform.length == 0.5
    fields["gain"] = 0.25
    waveform["length"] = 0.75
    assert cfg.gain == 0.2
    assert cfg.waveform.length == 0.5
    cfg.set_param("freq", 4500.0)
    cfg.waveform.set_param("length", 1.0)
    fresh = ModuleCfgFactory.from_raw(raw)
    assert isinstance(fresh, PulseCfg)
    assert fresh.freq == 4000.0
    assert fresh.gain == 0.25
    assert fresh.waveform.length == 0.75
    assert waveform["length"] == 0.75


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


class TestModuleCfgFactoryFromRaw:
    def test_dispatch_pulse(self):
        cfg = ModuleCfgFactory.from_raw(
            {
                "type": "pulse",
                "waveform": {"style": "const", "length": 0.5},
                "ch": 1,
                "nqz": 1,
                "freq": 4000.0,
                "gain": 0.5,
            }
        )
        assert isinstance(cfg, PulseCfg)

    def test_dispatch_readout_direct(self):
        cfg = ModuleCfgFactory.from_raw(
            {"type": "readout/direct", "ro_ch": 0, "ro_length": 1.0, "ro_freq": 5000.0}
        )
        assert isinstance(cfg, DirectReadoutCfg)

    def test_dispatch_reset_none(self):
        cfg = ModuleCfgFactory.from_raw({"type": "reset/none"})
        assert isinstance(cfg, NoneResetCfg)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="union_tag_invalid") as failure:
            ModuleCfgFactory.from_raw({"type": "no_such_module"})
        assert "Root <inline>; chain <inline>; expected module:" in str(failure.value)

    def test_missing_discriminator_raises(self):
        with pytest.raises(ValueError, match="union_tag_not_found") as failure:
            ModuleCfgFactory.from_raw({"ch": 1})
        assert "Root <inline>; chain <inline>; expected module:" in str(failure.value)

    def test_typed_context_materializes_string_ref(self):
        resolver = Resolver()
        pulse = PulseSpec(
            waveform=ConstWaveformCfg(length=0.2),
            ch=1,
            nqz=1,
            freq=4000.0,
            gain=0.5,
        )
        resolver.modules["library_pulse"] = pulse
        expected_pulse = pulse.to_dict()
        raw: CfgInputMap = {"type": "reset/pulse", "pulse_cfg": "library_pulse"}

        cfg = ModuleCfgFactory.from_raw(
            raw, context=CfgValidationContext(modules=resolver)
        )

        assert isinstance(cfg, PulseResetCfg)
        assert cfg.pulse_cfg.to_dict() == expected_pulse
        cfg.pulse_cfg.set_param("gain", 0.9)
        assert cfg.pulse_cfg.gain == 0.9
        assert pulse.to_dict() == expected_pulse
        assert resolver.modules["library_pulse"].to_dict() == expected_pulse
        assert raw == {"type": "reset/pulse", "pulse_cfg": "library_pulse"}

    def test_no_context_when_not_provided(self):
        cfg = ModuleCfgFactory.from_raw({"type": "reset/none"})
        assert isinstance(cfg, NoneResetCfg)


class TestWaveformCfgFactoryFromRaw:
    def test_dispatch_const(self):
        cfg = WaveformCfgFactory.from_raw({"style": "const", "length": 1.0})
        assert isinstance(cfg, ConstWaveformCfg)

    def test_dispatch_gauss(self):
        cfg = WaveformCfgFactory.from_raw(
            {"style": "gauss", "length": 1.0, "sigma": 0.2}
        )
        assert isinstance(cfg, GaussWaveformCfg)
        assert cfg.sigma == 0.2

    def test_dispatch_flat_top_with_nested_raise(self):
        cfg = WaveformCfgFactory.from_raw(
            {
                "style": "flat_top",
                "length": 1.0,
                "raise_waveform": {"style": "gauss", "length": 0.2, "sigma": 0.05},
            }
        )
        assert isinstance(cfg, FlatTopWaveformCfg)
        assert isinstance(cfg.raise_waveform, GaussWaveformCfg)

    def test_unknown_style_raises(self):
        with pytest.raises(ValueError, match="union_tag_invalid") as failure:
            WaveformCfgFactory.from_raw({"style": "no_such_style", "length": 1.0})
        assert "Root <inline>; chain <inline>; expected waveform:" in str(failure.value)

    def test_missing_discriminator_raises(self):
        with pytest.raises(ValueError, match="union_tag_not_found") as failure:
            WaveformCfgFactory.from_raw({"length": 1.0})
        assert "Root <inline>; chain <inline>; expected waveform:" in str(failure.value)
