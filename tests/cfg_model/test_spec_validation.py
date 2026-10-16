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
    ModuleCfgFactory,
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
    waveform: dict[str, CfgInputValue] = {"style": "const", "length": 0.5}
    fields: dict[str, CfgInputValue] = {
        "type": "readout/pulse",
        "gen_ch": 2,
        "nqz": 1,
        "freq": 400.0,
        "gain": 0.2,
        "ro_ch": 0,
        "ro_length": 1.0,
        "waveform": MappingProxyType(waveform),
    }
    raw: CfgInputMap = MappingProxyType(fields) if readonly_root else fields
    spec = module_spec_from_raw(raw)
    assert isinstance(spec, PulseReadoutSpec)
    assert isinstance(spec.waveform, ConstWaveformCfg)
    assert spec.freq == 400.0
    waveform["length"] = 0.75
    fields["freq"] = 500.0
    assert spec.freq == 400.0
    assert spec.waveform.length == 0.5
    spec.gain = 0.6
    fresh = module_spec_from_raw(raw)
    assert isinstance(fresh, PulseReadoutSpec)
    assert isinstance(fresh.waveform, ConstWaveformCfg)
    assert (fresh.freq, fresh.gain, fresh.waveform.length) == (500.0, 0.2, 0.75)


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


@pytest.mark.parametrize("count", [1, 8])
def test_mux_persistent_spec_roundtrips_numeric_tones(count: int):
    from zcu_tools import cfg_model

    from ._support import mux_raw

    raw = mux_raw(count)
    spec = module_spec_from_raw(raw)
    assert isinstance(spec, cfg_model.MuxReadoutSpec)
    assert module_spec_from_raw(spec).to_dict() == spec.to_dict()
    tones = raw["tones"]
    assert isinstance(tones, list)
    expected = []
    for tone in tones:
        assert isinstance(tone, dict)
        expected.append({**tone, "label": ""})
    assert spec.to_dict()["tones"] == expected
    assert ModuleCfgFactory.from_raw(spec).to_dict() == spec.to_dict()


@pytest.mark.parametrize("boundary", ["spec", "cfg"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tones", []),
        ("tones", [{}] * 9),
        ("gen_ch", True),
        ("gen_ch", -1),
        ("gen_ch", 1.5),
        ("gen_ch", "1"),
        ("nqz", 0),
        ("nqz", True),
        ("nqz", 3),
        ("mixer_freq", None),
        ("mixer_freq", float("nan")),
        ("length", 0.0),
        ("length", float("inf")),
        ("ro_length", -1.0),
        ("pre_delay", -0.1),
        ("post_delay", float("inf")),
        ("trig_offset", -0.1),
        ("waveform", {"style": "const", "length": 1.0}),
        ("ro_ch", 0),
        ("ro_chs", [0, 1]),
    ],
)
def test_mux_rejects_invalid_module_fields(
    boundary: str, field: str, value: CfgInputValue
):
    from zcu_tools.cfg_model import ModuleCfgFactory

    from ._support import mux_raw

    validate = module_spec_from_raw if boundary == "spec" else ModuleCfgFactory.from_raw
    raw = mux_raw()
    validate(raw)
    raw[field] = value
    with pytest.raises(ValueError, match=field):
        validate(raw)


@pytest.mark.parametrize("boundary", ["spec", "cfg"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("freq", float("nan")),
        ("freq", float("inf")),
        ("freq", "key"),
        ("freq", QickParam(1.0, {"sweep": 2.0})),
        ("gain", -1.01),
        ("gain", 1.01),
        ("gain", float("inf")),
        ("phase", float("nan")),
        ("phase", float("-inf")),
        ("ro_ch", True),
        ("ro_ch", -1),
        ("ro_ch", 0.5),
        ("ro_ch", "1"),
        ("name", "resonator"),
    ],
)
def test_mux_rejects_invalid_tone_fields(
    boundary: str, field: str, value: CfgInputValue
):
    from zcu_tools.cfg_model import ModuleCfgFactory

    from ._support import mux_raw

    validate = module_spec_from_raw if boundary == "spec" else ModuleCfgFactory.from_raw
    raw = mux_raw()
    validate(raw)
    raw["tones"] = [{"freq": -1.0, "gain": 0.5, "ro_ch": 4, field: value}]
    with pytest.raises(ValueError, match=f"{field}|persistent"):
        validate(raw)


@pytest.mark.parametrize("boundary", ["spec", "cfg"])
@pytest.mark.parametrize(
    "field", ["gen_ch", "nqz", "mixer_freq", "length", "ro_length"]
)
def test_mux_requires_explicit_module_settings(boundary: str, field: str):
    from zcu_tools.cfg_model import ModuleCfgFactory

    from ._support import mux_raw

    validate = module_spec_from_raw if boundary == "spec" else ModuleCfgFactory.from_raw
    raw = mux_raw()
    validate(raw)
    del raw[field]
    with pytest.raises(ValueError, match=field):
        validate(raw)


def test_mux_allows_equal_negative_frequencies_full_gain_and_phase_default():
    from zcu_tools.cfg_model import ModuleCfgFactory

    from ._support import mux_raw

    raw = mux_raw()
    raw["tones"] = [
        {"freq": -100.0, "gain": -1.0, "ro_ch": 4},
        {"freq": -100.0, "gain": 1.0, "ro_ch": 1},
    ]
    spec = module_spec_from_raw(raw)
    assert ModuleCfgFactory.from_raw(spec).to_dict()["tones"] == [
        {"freq": -100.0, "gain": -1.0, "phase": 0.0, "ro_ch": 4, "label": ""},
        {"freq": -100.0, "gain": 1.0, "phase": 0.0, "ro_ch": 1, "label": ""},
    ]
