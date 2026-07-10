# pyright: reportAttributeAccessIssue=false
"""Tests for cfg_schemas — convert Module/Waveform dicts to GUI representation."""

from __future__ import annotations

from typing import cast

import pytest
from zcu_tools.gui.app.main.adapter import (
    CfgSchema,
    CfgSectionValue,
    DirectValue,
    WaveformRefValue,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.main.cfg_schemas import (
    module_cfg_to_value,
    waveform_cfg_to_value,
)


def test_waveform_cfg_to_value():
    cfg = {"style": "gauss", "length": 2.0, "sigma": 0.5}
    spec, val = waveform_cfg_to_value(cfg)

    assert cast(DirectValue, val.fields["style"]).value == "gauss"
    assert cast(DirectValue, val.fields["length"]).value == 2.0
    assert cast(DirectValue, val.fields["sigma"]).value == 0.5
    # present key → value set (not unset, ADR-0010)
    assert cast(DirectValue, val.fields["sigma"]).value is not None


def test_waveform_cfg_to_value_missing_fields():
    cfg = {"style": "gauss"}
    spec, val = waveform_cfg_to_value(cfg)

    # A missing key is unset (value is None, ADR-0010) — no hard-coded default.
    assert cast(DirectValue, val.fields["length"]).value is None


def test_waveform_cfg_flat_top():
    cfg = {
        "style": "flat_top",
        "length": 3.0,
        "raise_waveform": {"style": "cosine", "length": 0.5},
    }
    spec, val = waveform_cfg_to_value(cfg)
    assert cast(DirectValue, val.fields["style"]).value == "flat_top"

    raise_wav = val.fields["raise_waveform"]
    assert isinstance(raise_wav, WaveformRefValue)
    assert cast(DirectValue, raise_wav.value.fields["length"]).value == 0.5


def test_waveform_cfg_to_value_arb_has_no_length_field():
    cfg = {"style": "arb", "length": 9.0, "data": "asset_a"}
    spec, val = waveform_cfg_to_value(cfg)

    assert "length" not in spec.fields
    assert "length" not in val.fields
    assert cast(DirectValue, val.fields["style"]).value == "arb"
    assert cast(DirectValue, val.fields["data"]).value == "asset_a"


def test_module_cfg_to_value_direct_readout():
    cfg = {"type": "readout/direct", "ro_freq": 7000.0}
    spec, val = module_cfg_to_value(cfg)

    assert cast(DirectValue, val.fields["type"]).value == "readout/direct"
    assert cast(DirectValue, val.fields["ro_freq"]).value == 7000.0
    # missing key → unset (value is None, no hard-coded default; ADR-0010)
    assert cast(DirectValue, val.fields["ro_length"]).value is None


def test_module_cfg_to_value_direct_readout_preserves_gen_ch_round_trip():
    cfg = {
        "type": "readout/direct",
        "ro_ch": 5,
        "ro_freq": 7000.0,
        "ro_length": 2.1,
        "trig_offset": 0.25,
        "gen_ch": 9,
    }
    spec, val = module_cfg_to_value(cfg)

    assert cast(DirectValue, val.fields["gen_ch"]).value == 9

    raw = schema_to_raw_dict(CfgSchema(spec=spec, value=val), md=None, ml=None)
    assert raw["gen_ch"] == 9


def test_module_cfg_to_value_direct_readout_missing_gen_ch_lowers_omitted():
    cfg = {
        "type": "readout/direct",
        "ro_ch": 5,
        "ro_freq": 7000.0,
        "ro_length": 2.1,
        "trig_offset": 0.25,
    }
    spec, val = module_cfg_to_value(cfg)

    assert cast(DirectValue, val.fields["gen_ch"]).value is None

    raw = schema_to_raw_dict(CfgSchema(spec=spec, value=val), md=None, ml=None)
    assert "gen_ch" not in raw


def test_module_cfg_to_value_pulse_reset():
    cfg = {"type": "reset/pulse", "pulse_cfg": {"freq": 5000.0}}
    spec, val = module_cfg_to_value(cfg)

    assert cast(DirectValue, val.fields["type"]).value == "reset/pulse"
    pulse_val = cast(CfgSectionValue, val.fields["pulse_cfg"])
    assert cast(DirectValue, pulse_val.fields["freq"]).value == 5000.0
    assert cast(DirectValue, pulse_val.fields["gain"]).value is None


def test_module_cfg_to_value_unknown_type_raises():
    with pytest.raises(RuntimeError, match="Unsupported module type"):
        module_cfg_to_value({"type": "unknown/custom", "some_key": "value"})


def test_waveform_cfg_to_value_unknown_style_raises():
    with pytest.raises(RuntimeError, match="Unsupported waveform style"):
        waveform_cfg_to_value({"style": "unknown_style", "length": 1.0})


def test_waveform_cfg_to_value_invalid_type():
    with pytest.raises(TypeError, match="Expected dict or AbsWaveformCfg"):
        waveform_cfg_to_value("not_a_dict")


# ---------------------------------------------------------------------------
# "pulse" module spec registration
# ---------------------------------------------------------------------------


def test_pulse_spec_registered():
    from zcu_tools.gui.app.main.cfg_schemas import (
        _MODULE_SPEC_FACTORIES,
        _MODULE_VALUE_BUILDERS,
    )

    assert "pulse" in _MODULE_SPEC_FACTORIES
    assert "pulse" in _MODULE_VALUE_BUILDERS


def test_module_cfg_to_value_pulse_basic():
    cfg = {
        "type": "pulse",
        "ch": 1,
        "freq": 5500.0,
        "gain": 0.8,
        "phase": 90.0,
        "nqz": 1,
        "pre_delay": 0.0,
        "post_delay": 0.0,
        "waveform": {"style": "const", "length": 2.0},
    }
    spec, val = module_cfg_to_value(cfg)

    assert cast(DirectValue, val.fields["type"]).value == "pulse"
    assert cast(DirectValue, val.fields["freq"]).value == 5500.0
    assert cast(DirectValue, val.fields["gain"]).value == 0.8
    assert cast(DirectValue, val.fields["freq"]).value is not None


def test_module_cfg_to_value_pulse_missing_fields():
    cfg = {"type": "pulse"}
    spec, val = module_cfg_to_value(cfg)

    # missing keys → unset (value is None, ADR-0010)
    assert cast(DirectValue, val.fields["freq"]).value is None
    assert cast(DirectValue, val.fields["gain"]).value is None


def test_module_cfg_to_value_pulse_round_trip():
    cfg = {
        "type": "pulse",
        "ch": 0,
        "freq": 5000.0,
        "gain": 0.5,
        "phase": 0.0,
        "nqz": 2,
        "pre_delay": 0.0,
        "post_delay": 0.0,
        "waveform": {"style": "const", "length": 1.0},
    }
    spec, val = module_cfg_to_value(cfg)
    schema = CfgSchema(spec=spec, value=val)
    out = schema_to_raw_dict(schema, md=None, ml=None)

    assert out["freq"] == 5000.0
    assert out["gain"] == 0.5
    assert out["type"] == "pulse"
    assert cast(dict, out["waveform"])["style"] == "const"
