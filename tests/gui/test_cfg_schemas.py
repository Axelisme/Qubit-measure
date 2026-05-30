# pyright: reportAttributeAccessIssue=false
"""Tests for cfg_schemas — convert Module/Waveform dicts to GUI representation."""

from __future__ import annotations

from typing import cast

import pytest
from zcu_tools.gui.adapter import DirectValue, WaveformRefValue
from zcu_tools.gui.cfg_schemas import (
    module_cfg_to_value,
    waveform_cfg_to_value,
)


def test_waveform_cfg_to_value():
    cfg = {"style": "gauss", "length": 2.0, "sigma": 0.5}
    spec, val = waveform_cfg_to_value(cfg)

    assert cast(DirectValue, val.fields["style"]).value == "gauss"
    assert cast(DirectValue, val.fields["length"]).value == 2.0
    assert cast(DirectValue, val.fields["sigma"]).value == 0.5
    assert cast(DirectValue, val.fields["sigma"]).is_unset is False


def test_waveform_cfg_to_value_missing_fields():
    cfg = {"style": "gauss"}
    spec, val = waveform_cfg_to_value(cfg)

    # Defaults should be loaded but marked as unset
    assert cast(DirectValue, val.fields["length"]).value == 1.0
    assert cast(DirectValue, val.fields["length"]).is_unset is True


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


def test_module_cfg_to_value_direct_readout():
    cfg = {"type": "readout/direct", "ro_freq": 7000.0}
    spec, val = module_cfg_to_value(cfg)

    assert cast(DirectValue, val.fields["type"]).value == "readout/direct"
    assert cast(DirectValue, val.fields["ro_freq"]).value == 7000.0
    assert cast(DirectValue, val.fields["ro_length"]).is_unset is True
    assert cast(DirectValue, val.fields["ro_length"]).value == 1.0


def test_module_cfg_to_value_pulse_reset():
    cfg = {"type": "reset/pulse", "pulse_cfg": {"freq": 5000.0}}
    spec, val = module_cfg_to_value(cfg)

    assert cast(DirectValue, val.fields["type"]).value == "reset/pulse"
    pulse_val = val.fields["pulse_cfg"]
    assert cast(DirectValue, pulse_val.fields["freq"]).value == 5000.0
    assert cast(DirectValue, pulse_val.fields["gain"]).is_unset is True


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
    from zcu_tools.gui.cfg_schemas import _MODULE_SPEC_FACTORIES, _MODULE_VALUE_BUILDERS

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
    assert cast(DirectValue, val.fields["freq"]).is_unset is False


def test_module_cfg_to_value_pulse_missing_fields():
    cfg = {"type": "pulse"}
    spec, val = module_cfg_to_value(cfg)

    assert cast(DirectValue, val.fields["freq"]).is_unset is True
    assert cast(DirectValue, val.fields["freq"]).value == 6000.0
    assert cast(DirectValue, val.fields["gain"]).is_unset is True


def test_module_cfg_to_value_pulse_round_trip():

    from zcu_tools.gui.adapter import CfgSchema, schema_to_dict

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
    out = schema_to_dict(schema, ml=None)

    assert out["freq"] == 5000.0
    assert out["gain"] == 0.5
    assert out["type"] == "pulse"
    assert out["waveform"]["style"] == "const"
