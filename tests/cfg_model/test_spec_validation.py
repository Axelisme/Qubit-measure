"""Pure persistent Spec validation without library or asset lookups."""

from types import MappingProxyType

import numpy as np
import pytest
from qick.asm_v2 import QickParam
from zcu_tools.cfg_model import (
    ArbWaveformCfg,
    CfgInputMap,
    CfgInputValue,
    CfgValidationContext,
    ConfigBase,
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    FlatTopWaveformSpec,
    GaussWaveformCfg,
    PulseReadoutSpec,
    PulseSpec,
    WaveformCfgFactory,
    module_spec_from_raw,
    waveform_spec_from_raw,
)

from ._support import AssetInfo


def test_waveform_spec_preserves_concrete_ramp_key_and_isolates_input():
    raw: CfgInputMap = {
        "style": "flat_top",
        "length": 2.0,
        "raise_waveform": "waveforms/ramp",
    }

    spec = waveform_spec_from_raw(raw)

    assert isinstance(spec, FlatTopWaveformSpec)
    assert spec.raise_waveform == "waveforms/ramp"
    spec.length = 3.0
    assert raw["length"] == 2.0


def test_module_spec_derives_inline_readout_fields_without_mutating_input():
    raw: CfgInputMap = {
        "type": "readout/pulse",
        "pulse_cfg": {
            "type": "pulse",
            "ch": 2,
            "nqz": 1,
            "freq": 4000.0,
            "gain": 0.2,
            "waveform": "waveforms/envelope",
        },
        "ro_cfg": {"type": "readout/direct", "ro_ch": 0, "ro_length": 1.0},
    }

    spec = module_spec_from_raw(raw)

    assert isinstance(spec, PulseReadoutSpec)
    assert spec.ro_cfg.gen_ch == 2
    assert spec.ro_cfg.ro_freq == 4000.0
    assert raw["ro_cfg"] == {"type": "readout/direct", "ro_ch": 0, "ro_length": 1.0}


@pytest.mark.parametrize("readonly_root", [False, True])
def test_waveform_spec_validates_readonly_mappings_without_aliases(
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

    spec = waveform_spec_from_raw(raw)

    assert isinstance(spec, FlatTopWaveformSpec)
    assert isinstance(spec.raise_waveform, GaussWaveformCfg)
    assert spec.length == 2.0
    assert spec.raise_waveform.length == 0.5
    ramp["length"] = 0.75
    fields["length"] = 3.0
    assert spec.length == 2.0
    assert spec.raise_waveform.length == 0.5
    spec.raise_waveform.length = 1.0
    fresh = waveform_spec_from_raw(raw)
    assert isinstance(fresh, FlatTopWaveformSpec)
    assert isinstance(fresh.raise_waveform, GaussWaveformCfg)
    assert fresh.length == 3.0
    assert fresh.raise_waveform.length == 0.75
    assert ramp["length"] == 0.75


@pytest.mark.parametrize("readonly_root", [False, True])
def test_module_spec_validates_readonly_mappings_without_aliases(readonly_root: bool):
    pulse: dict[str, CfgInputValue] = {
        "type": "pulse",
        "ch": 2,
        "nqz": 1,
        "freq": 4000.0,
        "gain": 0.2,
        "waveform": "C1.waveforms.envelope",
    }
    readout: dict[str, CfgInputValue] = {
        "type": "readout/direct",
        "ro_ch": 0,
        "ro_length": 1.0,
    }
    fields: dict[str, CfgInputValue] = {
        "type": "readout/pulse",
        "pulse_cfg": MappingProxyType(pulse),
        "ro_cfg": MappingProxyType(readout),
    }
    raw: CfgInputMap = MappingProxyType(fields) if readonly_root else fields

    spec = module_spec_from_raw(raw)

    assert isinstance(spec, PulseReadoutSpec)
    assert isinstance(spec.pulse_cfg, PulseSpec)
    assert spec.pulse_cfg.freq == 4000.0
    assert spec.pulse_cfg.waveform == "C1.waveforms.envelope"
    assert spec.ro_cfg.gen_ch == 2
    assert spec.ro_cfg.ro_freq == 4000.0
    assert readout == {"type": "readout/direct", "ro_ch": 0, "ro_length": 1.0}
    pulse["freq"] = 5000.0
    assert spec.pulse_cfg.freq == 4000.0
    assert spec.ro_cfg.ro_freq == 4000.0
    spec.pulse_cfg.gain = 0.6
    fresh = module_spec_from_raw(raw)
    assert isinstance(fresh, PulseReadoutSpec)
    assert isinstance(fresh.pulse_cfg, PulseSpec)
    assert fresh.pulse_cfg.gain == 0.2
    assert fresh.pulse_cfg.freq == 5000.0
    assert fresh.ro_cfg.ro_freq == 5000.0
    assert fresh.pulse_cfg.waveform == "C1.waveforms.envelope"
    assert pulse["gain"] == 0.2


@pytest.mark.parametrize(
    "length",
    [
        QickParam(start=1.0, spans={"length": 0.5}),
        1.0 + 2.0j,
        np.array([1.0, 2.0]),
        (1.0, 2.0),
        MappingProxyType({7: 1.0}),
    ],
)
def test_spec_keeps_persistent_value_rejections_inside_readonly_mapping(length: object):
    raw: CfgInputMap = MappingProxyType(
        {
            "style": "flat_top",
            "length": 2.0,
            "raise_waveform": ConstWaveformCfg.model_construct(length=length),
        }
    )

    with pytest.raises(ValueError, match="persistent"):
        waveform_spec_from_raw(raw)


def test_waveform_spec_rejects_qick_sweep_without_lossy_serialization():
    raw: CfgInputMap = {
        "style": "const",
        "length": QickParam(start=1.0, spans={"length": 0.5}),
    }
    with pytest.raises(ValueError, match="persistent"):
        waveform_spec_from_raw(raw)


def test_module_spec_rejects_inline_composite_runtime_cfg():
    raw: CfgInputMap = {
        "type": "pulse",
        "ch": 0,
        "nqz": 1,
        "freq": 4000.0,
        "gain": 0.2,
        "waveform": FlatTopWaveformCfg(
            length=1.0, raise_waveform=GaussWaveformCfg(length=0.5, sigma=0.1)
        ),
    }
    with pytest.raises(ValueError, match="persistent"):
        module_spec_from_raw(raw)


def test_waveform_spec_rejects_bound_arb_instead_of_discarding_asset_binding():
    cfg = WaveformCfgFactory.from_raw(
        {"style": "arb", "data": "shape"},
        context=CfgValidationContext(assets=AssetInfo()),
    )
    assert isinstance(cfg, ArbWaveformCfg)
    with pytest.raises(ValueError, match="persistent"):
        waveform_spec_from_raw(cfg)


@pytest.mark.parametrize(
    "raw",
    [
        {"style": "const", "length": 1.0 + 2.0j},
        ConstWaveformCfg.model_construct(length=np.array([1.0, 2.0])),
    ],
)
def test_waveform_spec_rejects_complex_and_array_values(
    raw: CfgInputMap | ConstWaveformCfg,
):
    with pytest.raises(ValueError, match="persistent"):
        waveform_spec_from_raw(raw)


def test_shared_scalar_leaf_remains_valid_as_an_unbound_persistent_spec():
    raw = ArbWaveformCfg(data="not-read-at-spec-validation")
    spec = waveform_spec_from_raw(raw)

    assert isinstance(spec, ArbWaveformCfg)
    assert spec.data == raw.data
    with pytest.raises(ValueError, match="requires assets"):
        spec.asset_samples()


def test_spec_rejects_an_unrecognized_model_instead_of_dumping_its_runtime_fields():
    class RuntimeEnvelope(ConfigBase):
        style: str = "const"
        length: float = 1.0

    raw = PulseSpec.model_construct(
        ch=0, nqz=1, freq=4000.0, gain=0.2, waveform=RuntimeEnvelope()
    )

    with pytest.raises(ValueError, match="persistent"):
        module_spec_from_raw(raw)
