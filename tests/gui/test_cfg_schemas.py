# pyright: reportAttributeAccessIssue=false
"""Tests for cfg_schemas — convert Module/Waveform dicts to GUI representation."""

from __future__ import annotations

from typing import cast

import pytest
from zcu_tools.gui.adapter import CfgSectionValue, DirectValue, WaveformRefValue
from zcu_tools.gui.cfg_schemas import (
    _dict_to_spec_value,
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


def test_module_cfg_to_value_fallback():
    cfg = {"unknown_key": "some_value", "nested": {"sub": 42}}
    spec, val = module_cfg_to_value(cfg)

    assert isinstance(val.fields["unknown_key"], DirectValue)
    assert cast(DirectValue, val.fields["unknown_key"]).value == "some_value"
    assert cast(CfgSectionValue, val.fields["nested"]).fields["sub"].value == 42


def test_waveform_cfg_to_value_invalid_type():
    with pytest.raises(TypeError, match="Expected dict or AbsWaveformCfg"):
        waveform_cfg_to_value("not_a_dict")
