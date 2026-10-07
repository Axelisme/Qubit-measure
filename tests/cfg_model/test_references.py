"""Persistent reference rewriting through cfg_model's public pure seams."""

import pytest
from zcu_tools.cfg_model import (
    ArbWaveformCfg,
    BathResetSpec,
    CfgInputMap,
    DirectReadoutCfg,
    FlatTopWaveformSpec,
    ModuleSpec,
    NoneResetCfg,
    PulseReadoutSpec,
    PulseResetSpec,
    PulseSpec,
    TwoPulseResetSpec,
    module_spec_from_raw,
    remap_module_references,
    remap_waveform_references,
    waveform_spec_from_raw,
)


def pulse() -> PulseSpec:
    return PulseSpec(
        ch=0,
        nqz=1,
        freq=123.5,
        gain=0.25,
        waveform="wave",
        desc="wave",
        mux_freqs=[12.0],
    )


@pytest.mark.parametrize(
    "spec",
    [
        PulseReadoutSpec(
            pulse_cfg="pulse",
            ro_cfg=DirectReadoutCfg(ro_ch=0, ro_length=2.5, ro_freq=123.5),
        ),
        PulseResetSpec(pulse_cfg="pulse"),
        TwoPulseResetSpec(pulse1_cfg="pulse", pulse2_cfg="pulse"),
        BathResetSpec(cavity_tone_cfg="pulse", qubit_tone_cfg="pulse", pi2_cfg="pulse"),
    ],
)
def test_module_string_children_use_only_module_mapping(spec: ModuleSpec) -> None:
    original = spec.model_dump()
    mapped = remap_module_references(
        spec, modules={"pulse": "Q1.pulses.pi"}, waveforms={"pulse": "wrong"}
    )
    expected = {
        key: "Q1.pulses.pi" if value == "pulse" else value
        for key, value in original.items()
    }
    assert mapped == module_spec_from_raw(expected)
    assert spec.model_dump() == original


@pytest.mark.parametrize(
    "spec",
    [
        PulseReadoutSpec(
            pulse_cfg=pulse(),
            ro_cfg=DirectReadoutCfg(ro_ch=0, ro_length=2.5, ro_freq=123.5),
        ),
        PulseResetSpec(pulse_cfg=pulse()),
        TwoPulseResetSpec(pulse1_cfg=pulse(), pulse2_cfg=pulse()),
        BathResetSpec(cavity_tone_cfg=pulse(), qubit_tone_cfg=pulse(), pi2_cfg=pulse()),
    ],
)
def test_inline_pulses_rewrite_waveforms_without_mutating_input(
    spec: ModuleSpec,
) -> None:
    original = spec.model_dump()
    mapped = remap_module_references(
        spec, modules={}, waveforms={"wave": "Q1.waveforms.base"}
    )
    children = [child for _, child in mapped if isinstance(child, PulseSpec)]
    assert children
    assert all(child.waveform == "Q1.waveforms.base" for child in children)
    assert all(child.desc == "wave" for child in children)
    children[0].mux_freqs = [99.0]
    assert spec.model_dump() == original


def test_nested_inline_flat_top_rewrites_only_its_ramp_reference() -> None:
    spec = pulse()
    spec.waveform = FlatTopWaveformSpec(length=0.125, raise_waveform="ramp")
    mapped = remap_module_references(
        spec, modules={}, waveforms={"ramp": "Q1.waveforms.ramp"}
    )
    assert isinstance(mapped, PulseSpec)
    assert isinstance(mapped.waveform, FlatTopWaveformSpec)
    assert mapped.waveform.raise_waveform == "Q1.waveforms.ramp"
    assert mapped.waveform.length == 0.125
    assert mapped.freq == 123.5
    assert mapped.desc == "wave"
    assert spec.waveform.raise_waveform == "ramp"


@pytest.mark.parametrize(
    "raw",
    [
        {"style": "cosine", "length": 0.05},
        {"style": "gauss", "length": 0.05, "sigma": 0.01},
        {"style": "drag", "length": 0.05, "sigma": 0.01, "delta": 20.0, "alpha": 0.2},
        {"style": "arb", "data": "asset"},
    ],
)
def test_inline_ramps_and_arb_asset_keys_are_not_references(raw: CfgInputMap) -> None:
    ramp = waveform_spec_from_raw(raw)
    flat = waveform_spec_from_raw(
        {"style": "flat_top", "length": 0.125, "raise_waveform": ramp}
    )
    mapped = remap_waveform_references(flat, waveforms={"asset": "do_not_use"})
    assert mapped == flat
    assert mapped is not flat
    if isinstance(mapped, FlatTopWaveformSpec):
        assert mapped.raise_waveform is not ramp


def test_arb_asset_key_is_kept_through_inline_module_rewriting() -> None:
    spec = pulse()
    spec.waveform = ArbWaveformCfg(data="asset")
    mapped = remap_module_references(
        spec, modules={}, waveforms={"asset": "do_not_use"}
    )
    assert isinstance(mapped, PulseSpec)
    assert isinstance(mapped.waveform, ArbWaveformCfg)
    assert mapped.waveform.data == "asset"


@pytest.mark.parametrize(
    "spec", [NoneResetCfg(), DirectReadoutCfg(ro_ch=0, ro_length=0.25, ro_freq=200.0)]
)
def test_reference_free_modules_are_isolated(spec: ModuleSpec) -> None:
    mapped = remap_module_references(spec, modules={}, waveforms={})
    assert mapped == spec
    assert mapped is not spec


@pytest.mark.parametrize("reference", ["wave", "Q1.waveforms.already_canonical"])
def test_unmapped_waveform_reference_has_no_canonical_fallback(reference: str) -> None:
    spec = pulse()
    spec.waveform = reference
    with pytest.raises(KeyError):
        remap_module_references(spec, modules={}, waveforms={})
    with pytest.raises(KeyError):
        remap_waveform_references(
            FlatTopWaveformSpec(length=1.0, raise_waveform=reference), waveforms={}
        )
    assert spec.waveform == reference


def test_unmapped_module_child_is_rejected_without_mutation() -> None:
    spec = PulseResetSpec(pulse_cfg="unknown")
    with pytest.raises(KeyError):
        remap_module_references(spec, modules={}, waveforms={})
    assert spec.pulse_cfg == "unknown"


def test_invalid_persistent_input_is_revalidated() -> None:
    spec = PulseSpec.model_construct(
        ch=0, nqz=1, freq=object(), gain=0.25, waveform="wave"
    )
    with pytest.raises(ValueError, match="persistent"):
        remap_module_references(
            spec, modules={}, waveforms={"wave": "Q1.waveforms.base"}
        )
