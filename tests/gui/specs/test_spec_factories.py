"""Tests for fresh spec factories."""

from __future__ import annotations

from typing import cast

from zcu_tools.gui.app.main.adapter import ReferenceSpec
from zcu_tools.gui.app.main.specs import (
    make_direct_readout_spec,
    make_pulse_readout_spec,
    make_pulse_spec,
    make_waveform_spec_by_style,
)


def test_direct_readout_factory_returns_fresh_specs():
    first = make_direct_readout_spec()
    second = make_direct_readout_spec()

    assert first is not second
    assert first.fields is not second.fields

    first.fields["ro_freq"] = first.fields["ro_ch"]
    assert "ro_freq" in second.fields


def test_pulse_readout_factory_returns_fresh_nested_specs():
    first = make_pulse_readout_spec()
    second = make_pulse_readout_spec()

    first_pulse = first.fields["pulse_cfg"]
    second_pulse = second.fields["pulse_cfg"]
    assert first_pulse is not second_pulse

    first_ro = first.fields["ro_cfg"]
    second_ro = second.fields["ro_cfg"]
    assert first_ro is not second_ro


def test_waveform_style_factory_returns_expected_fresh_shape():
    first = make_waveform_spec_by_style("gauss")
    second = make_waveform_spec_by_style("gauss")

    assert first is not second
    assert first.label == "Gauss"
    assert "sigma" in first.fields
    assert "sigma" in second.fields


def test_pulse_factory_embeds_fresh_waveform_allowed_specs():
    first = make_pulse_spec()
    second = make_pulse_spec()

    first_waveform = cast(ReferenceSpec, first.fields["waveform"])
    second_waveform = cast(ReferenceSpec, second.fields["waveform"])
    assert first_waveform is not second_waveform
    assert first_waveform.allowed[0] is not second_waveform.allowed[0]
