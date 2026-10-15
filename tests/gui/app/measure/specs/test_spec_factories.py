"""Tests for fresh spec factories."""

from __future__ import annotations

from zcu_tools.gui.app.measure.specs import (
    make_pulse_readout_spec,
    make_pulse_spec,
    make_waveform_spec_by_style,
)
from zcu_tools.gui.cfg import ReferenceSpec


def test_pulse_readout_factory_returns_fresh_waveform_specs():
    first = make_pulse_readout_spec()
    second = make_pulse_readout_spec()
    first_waveform = first.fields["waveform"]
    second_waveform = second.fields["waveform"]
    assert isinstance(first_waveform, ReferenceSpec)
    assert isinstance(second_waveform, ReferenceSpec)
    assert first_waveform is not second_waveform
    assert first_waveform.allowed[0] is not second_waveform.allowed[0]


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

    first_waveform = first.fields["waveform"]
    assert isinstance(first_waveform, ReferenceSpec)
    second_waveform = second.fields["waveform"]
    assert isinstance(second_waveform, ReferenceSpec)
    assert first_waveform is not second_waveform
    assert first_waveform.allowed[0] is not second_waveform.allowed[0]
