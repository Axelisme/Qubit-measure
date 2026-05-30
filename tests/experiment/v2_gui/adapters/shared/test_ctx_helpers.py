"""Unit tests for md_writeback / proper_relax / proper_*_freq_range helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.experiment.v2_gui.adapters.shared import (
    md_writeback,
    proper_qub_freq_range,
    proper_relax,
    proper_res_freq_range,
)
from zcu_tools.gui.adapter import (
    DirectValue,
    EvalValue,
    MetaDictWriteback,
    SweepValue,
)


def _ctx_with_md(values: dict) -> MagicMock:
    ctx = MagicMock()
    ctx.md.get.side_effect = lambda k, d=None: values.get(k, d)
    ctx.md.__contains__ = lambda self, k: k in values
    return ctx


# --- proper_relax ----------------------------------------------------------


def test_proper_relax_uses_eval_when_t1_present():
    ctx = _ctx_with_md({"t1": 50.0})
    result = proper_relax(ctx)
    assert isinstance(result, EvalValue)
    assert result.expr == "5.0 * t1"
    assert result.resolved == 250.0


def test_proper_relax_custom_factor():
    ctx = _ctx_with_md({"t1": 10.0})
    result = proper_relax(ctx, factor=3.0)
    assert isinstance(result, EvalValue)
    assert result.expr == "3.0 * t1"
    assert result.resolved == 30.0


def test_proper_relax_falls_back_without_t1():
    ctx = _ctx_with_md({})
    result = proper_relax(ctx)
    assert isinstance(result, DirectValue)
    assert result.value == 100.0


def test_proper_relax_custom_fallback():
    ctx = _ctx_with_md({})
    result = proper_relax(ctx, fallback=42.0)
    assert isinstance(result, DirectValue)
    assert result.value == 42.0


# --- md_writeback ----------------------------------------------------------


def test_md_writeback_collapses_key_and_md_key():
    ctx = _ctx_with_md({"q_f": 4000.0})
    item = md_writeback(ctx, "q_f", "Qubit frequency (MHz)", 4123.456789)
    assert isinstance(item, MetaDictWriteback)
    assert item.key == "q_f"
    assert item.md_key == "q_f"
    assert item.description == "Qubit frequency (MHz)"
    assert item.current_value == 4000.0
    assert item.proposed_value == 4123.4568  # rounded to 4 digits


def test_md_writeback_custom_ndigits():
    ctx = _ctx_with_md({"timeFly": None})
    item = md_writeback(ctx, "timeFly", "Trigger offset", 0.123456789, ndigits=6)
    assert item.proposed_value == 0.123457
    assert item.current_value is None


# --- proper_*_freq_range ----------------------------------------------------


def test_res_freq_range_uses_eval_value_when_md_present():
    ctx = _ctx_with_md({"r_f": 5500.0, "rf_w": 10.0})
    sv = proper_res_freq_range(ctx, 301)
    assert isinstance(sv, SweepValue)
    assert isinstance(sv.start, EvalValue)
    assert isinstance(sv.stop, EvalValue)
    assert sv.start.expr == "r_f - 1.5 * rf_w"
    assert sv.start.resolved == 5485.0
    assert sv.stop.expr == "r_f + 1.5 * rf_w"
    assert sv.stop.resolved == 5515.0
    assert sv.expts == 301


def test_res_freq_range_falls_back_to_scalar_without_md():
    sv = proper_res_freq_range(_ctx_with_md({}), 101)
    # no md → plain float edges (6000 ± 30 default span)
    assert sv.start == 5970.0
    assert sv.stop == 6030.0


def test_freq_range_span_factor_one_omits_coefficient():
    ctx = _ctx_with_md({"r_f": 6000.0, "rf_w": 20.0})
    sv = proper_res_freq_range(ctx, 101, span_factor=1.0)
    assert isinstance(sv.start, EvalValue)
    assert sv.start.expr == "r_f - rf_w"  # not "r_f - 1.0 * rf_w"
    assert sv.start.resolved == 5980.0


def test_qub_freq_range_uses_qubit_md_keys():
    ctx = _ctx_with_md({"q_f": 4200.0, "qf_w": 5.0})
    sv = proper_qub_freq_range(ctx, 201, span_factor=2.0)
    assert isinstance(sv.start, EvalValue)
    assert sv.start.expr == "q_f - 2.0 * qf_w"
    assert sv.start.resolved == 4190.0
    assert sv.expts == 201
