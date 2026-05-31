"""Tests for the per-role default factories (defaults/ — blank + ref each)."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_direct_readout_default,
    make_pi2_pulse_ref_default,
    make_pi_pulse_ref_default,
    make_pulse_readout_default,
    make_qub_probe_default,
    make_qub_waveform_default,
    make_readout_default,
    make_readout_ref_default,
    make_res_probe_default,
    make_res_waveform_default,
    make_reset_default,
    make_reset_ref_default,
)
from zcu_tools.gui.adapter import (
    DisabledRefValue,
    EvalValue,
    ModuleRefValue,
    WaveformRefValue,
)


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
    """md keys present → ch/freq are EvalValue carrying the md key expression."""
    ctx = MagicMock()
    ctx.md.get.side_effect = lambda k, d=None: {"qub_ch": 4, "q_f": 4200.0}.get(k, d)
    ctx.md.__contains__ = lambda self, k: k in {"qub_ch", "q_f"}
    v = make_qub_probe_default(ctx)
    assert isinstance(v, ModuleRefValue)
    assert v.chosen_key.startswith("<Custom:")
    ch = cast(Any, v.value.fields["ch"])
    freq = cast(Any, v.value.fields["freq"])
    assert isinstance(ch, EvalValue) and ch.expr == "qub_ch"
    assert isinstance(freq, EvalValue) and freq.expr == "q_f"


def test_qub_probe_default_falls_back_to_direct_when_md_absent():
    """md keys absent → ch/freq are DirectValue fallbacks."""
    v = make_qub_probe_default(_empty_ctx())
    ch = cast(Any, v.value.fields["ch"])
    freq = cast(Any, v.value.fields["freq"])
    assert not isinstance(ch, EvalValue) and ch.value == 0
    assert not isinstance(freq, EvalValue) and freq.value == 4000.0


# --- res_probe (blank, res_ch / r_f, no ro_cfg) -----------------------------


def test_res_probe_default_uses_resonator_channel_and_no_ro_cfg():
    ctx = MagicMock()
    ctx.md.get.side_effect = lambda k, d=None: {"res_ch": 3, "r_f": 6500.0}.get(k, d)
    ctx.md.__contains__ = lambda self, k: k in {"res_ch", "r_f"}
    v = make_res_probe_default(ctx)
    ch = cast(Any, v.value.fields["ch"])
    freq = cast(Any, v.value.fields["freq"])
    assert isinstance(ch, EvalValue) and ch.expr == "res_ch"
    assert isinstance(freq, EvalValue) and freq.expr == "r_f"
    assert "ro_cfg" not in v.value.fields


# --- pi / pi2 ref (fallback to blank when lib empty) ------------------------


def test_pi_pulse_ref_falls_back_to_blank():
    assert isinstance(make_pi_pulse_ref_default(_empty_ctx()), ModuleRefValue)


def test_pi2_pulse_ref_falls_back_to_blank():
    assert isinstance(make_pi2_pulse_ref_default(_empty_ctx()), ModuleRefValue)


def test_pi_pulse_ref_optional_returns_disabled_when_lib_empty():
    assert isinstance(
        make_pi_pulse_ref_default(_empty_ctx(), optional=True), DisabledRefValue
    )


# --- readout (blank inline pulse+ro_cfg; ref to library) --------------------


def test_readout_default_is_inline_pulse_readout_with_ro_cfg():
    v = make_readout_default(_empty_ctx())
    assert isinstance(v, ModuleRefValue)
    assert v.chosen_key.startswith("<Custom:")
    assert "ro_cfg" in v.value.fields


def test_make_readout_default_aliases_pulse_shape():
    """The role blank is the pulse shape (Custom:Pulse Readout)."""
    v = make_readout_default(_empty_ctx())
    assert v.chosen_key == "<Custom:Pulse Readout>"
    assert "pulse_cfg" in v.value.fields


def test_make_pulse_readout_default_shape():
    v = make_pulse_readout_default(_empty_ctx())
    assert isinstance(v, ModuleRefValue)
    assert v.chosen_key == "<Custom:Pulse Readout>"
    assert "pulse_cfg" in v.value.fields
    assert "ro_cfg" in v.value.fields


def test_make_direct_readout_default_shape():
    """Direct readout is a bare ro_cfg — no pulse_cfg."""
    v = make_direct_readout_default(_empty_ctx())
    assert isinstance(v, ModuleRefValue)
    assert v.chosen_key == "<Custom:Direct Readout>"
    assert "pulse_cfg" not in v.value.fields
    assert "ro_freq" in v.value.fields


def test_direct_readout_eval_fields_when_md_present():
    """md keys present → ro_ch/ro_freq are EvalValue carrying the expression."""
    ctx = MagicMock()
    ctx.md.get.side_effect = lambda k, d=None: {"r_f": 6500.0, "ro_ch": 1}.get(k, d)
    ctx.md.__contains__ = lambda self, k: k in {"r_f", "ro_ch"}
    ml = MagicMock()
    ml.modules = {}
    ml.waveforms = {}
    ctx.ml = ml

    v = make_direct_readout_default(ctx)
    ro_freq = cast(Any, v.value.fields["ro_freq"])
    ro_ch = cast(Any, v.value.fields["ro_ch"])
    assert isinstance(ro_freq, EvalValue) and ro_freq.expr == "r_f"
    assert isinstance(ro_ch, EvalValue) and ro_ch.expr == "ro_ch"
    # not pre-resolved — lowering owns resolution
    assert ro_freq.resolved is None and ro_ch.resolved is None


def test_readout_ref_falls_back_to_blank():
    assert isinstance(make_readout_ref_default(_empty_ctx()), ModuleRefValue)


def test_readout_ref_optional_returns_disabled_when_lib_empty():
    assert isinstance(
        make_readout_ref_default(_empty_ctx(), optional=True), DisabledRefValue
    )


# --- reset ------------------------------------------------------------------


def test_reset_default_returns_blank_pulse_reset():
    v = make_reset_default(_empty_ctx())
    assert isinstance(v, ModuleRefValue)


def test_reset_ref_optional_returns_disabled_when_lib_empty():
    assert isinstance(
        make_reset_ref_default(_empty_ctx(), optional=True), DisabledRefValue
    )


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
    v = make_qub_probe_default(_empty_ctx()).with_field("gain", 0.3)
    assert isinstance(v, ModuleRefValue)
    assert cast(Any, v.value.fields["gain"]).value == 0.3
