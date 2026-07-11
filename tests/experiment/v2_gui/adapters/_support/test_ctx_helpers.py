"""Unit tests for cross-adapter frequency-range mechanics."""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.experiment.v2_gui.adapters._support.ctx_helpers import (
    proper_flux_range,
    proper_qub_freq_range,
    proper_res_freq_range,
)
from zcu_tools.gui.cfg import (
    EvalValue,
    SweepValue,
)


def _ctx_with_md(values: dict) -> MagicMock:
    ctx = MagicMock()
    ctx.md.get.side_effect = lambda k, d=None: values.get(k, d)
    ctx.md.__contains__ = lambda self, k: k in values
    return ctx


# --- proper_*_freq_range ----------------------------------------------------


def test_res_freq_range_uses_eval_value_when_md_present():
    ctx = _ctx_with_md({"r_f": 5500.0, "rf_w": 10.0})
    sv = proper_res_freq_range(ctx, 301)
    assert isinstance(sv, SweepValue)
    assert isinstance(sv.start, EvalValue)
    assert isinstance(sv.stop, EvalValue)
    assert sv.start.expr == "r_f - 1.5 * rf_w"
    assert sv.start.resolved is None
    assert sv.stop.expr == "r_f + 1.5 * rf_w"
    assert sv.stop.resolved is None
    assert sv.expts == 301


def test_res_freq_range_falls_back_to_scalar_without_md():
    sv = proper_res_freq_range(_ctx_with_md({}), 101)
    # no md → plain float edges (6500 ± 1.5*500 default span)
    assert sv.start == 5750.0
    assert sv.stop == 7250.0


def test_freq_range_span_factor_one_omits_coefficient():
    ctx = _ctx_with_md({"r_f": 6000.0, "rf_w": 20.0})
    sv = proper_res_freq_range(ctx, 101, span_factor=1.0)
    assert isinstance(sv.start, EvalValue)
    assert sv.start.expr == "r_f - rf_w"  # not "r_f - 1.0 * rf_w"
    assert sv.start.resolved is None


def test_qub_freq_range_uses_qubit_md_keys():
    ctx = _ctx_with_md({"q_f": 4200.0, "qf_w": 5.0})
    sv = proper_qub_freq_range(ctx, 201, span_factor=2.0)
    assert isinstance(sv.start, EvalValue)
    assert sv.start.expr == "q_f - 2.0 * qf_w"
    assert sv.start.resolved is None
    assert sv.expts == 201


# --- proper_flux_range ------------------------------------------------------


def test_flux_range_extrapolates_past_calibrated_points():
    ctx = _ctx_with_md({"flx_half": 1e-3, "flx_int": 3e-3})
    sv = proper_flux_range(ctx, 101)
    assert isinstance(sv.start, EvalValue)
    assert isinstance(sv.stop, EvalValue)
    assert sv.start.expr == "1.1 * flx_int - 0.1 * flx_half"
    assert sv.start.resolved is None
    assert sv.stop.expr == "1.1 * flx_half - 0.1 * flx_int"
    assert sv.expts == 101


def test_flux_range_falls_back_when_md_absent():
    sv = proper_flux_range(_ctx_with_md({}), 101)
    assert sv.start == -4e-3
    assert sv.stop == 4e-3
