"""Tests for the per-role default factories (defaults/ — blank + ref each)."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pi2_pulse_ref_default,
    make_pi_pulse_ref_default,
    make_qub_probe_default,
    make_qub_waveform_default,
    make_readout_default,
    make_readout_ref_default,
    make_res_probe_default,
    make_res_waveform_default,
    make_reset_default,
    make_reset_ref_default,
)
from zcu_tools.gui.adapter import ModuleRefValue, WaveformRefValue


def _empty_ctx() -> MagicMock:
    """A ctx with empty md/ml — exercises the blank fallback path."""
    ctx = MagicMock()
    ctx.md.get.side_effect = lambda k, d=None: d
    ctx.md.__contains__ = lambda self, k: False
    ml = MagicMock()
    ml.modules = {}
    ml.waveforms = {}
    ctx.ml = ml
    return ctx


# --- qub_probe (blank, qub_ch / q_f) ----------------------------------------


def test_qub_probe_default_uses_qubit_channel_and_freq():
    ctx = MagicMock()
    ctx.md.get.side_effect = lambda k, d=None: {"qub_ch": 4, "q_f": 4200.0}.get(k, d)
    v = make_qub_probe_default(ctx)
    assert isinstance(v, ModuleRefValue)
    assert v.chosen_key.startswith("<Custom:")
    assert cast(Any, v.value.fields["ch"]).value == 4
    assert cast(Any, v.value.fields["freq"]).value == 4200.0


# --- res_probe (blank, res_ch / r_f, no ro_cfg) -----------------------------


def test_res_probe_default_uses_resonator_channel_and_no_ro_cfg():
    ctx = MagicMock()
    ctx.md.get.side_effect = lambda k, d=None: {"res_ch": 3, "r_f": 6500.0}.get(k, d)
    v = make_res_probe_default(ctx)
    assert cast(Any, v.value.fields["ch"]).value == 3
    assert cast(Any, v.value.fields["freq"]).value == 6500.0
    assert "ro_cfg" not in v.value.fields


# --- pi / pi2 ref (fallback to blank when lib empty) ------------------------


def test_pi_pulse_ref_falls_back_to_blank():
    assert isinstance(make_pi_pulse_ref_default(_empty_ctx()), ModuleRefValue)


def test_pi2_pulse_ref_falls_back_to_blank():
    assert isinstance(make_pi2_pulse_ref_default(_empty_ctx()), ModuleRefValue)


def test_pi_pulse_ref_optional_returns_none_when_lib_empty():
    assert make_pi_pulse_ref_default(_empty_ctx(), optional=True) is None


# --- readout (blank inline pulse+ro_cfg; ref to library) --------------------


def test_readout_default_is_inline_pulse_readout_with_ro_cfg():
    v = make_readout_default(_empty_ctx())
    assert isinstance(v, ModuleRefValue)
    assert v.chosen_key.startswith("<Custom:")
    assert "ro_cfg" in v.value.fields


def test_readout_ref_falls_back_to_blank():
    assert isinstance(make_readout_ref_default(_empty_ctx()), ModuleRefValue)


def test_readout_ref_optional_returns_none_when_lib_empty():
    assert make_readout_ref_default(_empty_ctx(), optional=True) is None


# --- reset ------------------------------------------------------------------


def test_reset_default_returns_blank_pulse_reset():
    v = make_reset_default(_empty_ctx())
    assert isinstance(v, ModuleRefValue)


def test_reset_ref_optional_returns_none_when_lib_empty():
    assert make_reset_ref_default(_empty_ctx(), optional=True) is None


# --- waveforms --------------------------------------------------------------


def test_qub_waveform_default_is_blank_cosine():
    v = make_qub_waveform_default(_empty_ctx())
    assert isinstance(v, WaveformRefValue)
    assert v.chosen_key == "<Custom:Cosine>"


def test_res_waveform_default_is_blank_const():
    v = make_res_waveform_default(_empty_ctx())
    assert isinstance(v, WaveformRefValue)
    assert v.chosen_key == "<Custom:Const>"


# --- composition with value OO ---------------------------------------------


def test_role_factory_composes_with_value_with_field():
    v = cast(Any, make_qub_probe_default(_empty_ctx())).with_field("gain", 0.3)
    assert isinstance(v, ModuleRefValue)
    assert cast(Any, v.value.fields["gain"]).value == 0.3
