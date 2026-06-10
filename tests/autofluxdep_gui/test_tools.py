"""Tools tests — the Smoother mechanism and the Tools container.

Smoother is the pure cross-point smoothing mechanism a SmoothingService builds
on (it is NOT injected into Nodes — Nodes report raw). These tests prove the two
smoothing modes match the notebook semantics and that a smoothed value blends
against the *recursively* smoothed history, not the raw previous value (the
``smooth_t1 = 0.5*(prev_smooth + cur)`` subtlety). The orchestrator integration
of smoothing via a SmoothingService lives in ``test_derivation.py``.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.autofluxdep.tools import Smoother, Tools


def test_smoother_first_observation_returns_raw():
    s = Smoother()
    assert s.update("t1", 0, 10.0) == 10.0


def test_smoother_ewma_blends_prev_smoothed_with_cur():
    s = Smoother(ewma_alpha=0.5)
    s.update("t1", 0, 10.0)  # prev = 10
    # 0.5*10 + 0.5*20 = 15 ; the prev term is the SMOOTHED value, not raw
    assert s.update("t1", 1, 20.0) == 15.0
    # next blends against 15 (recursive), not 20: 0.5*15 + 0.5*5 = 10
    assert s.update("t1", 2, 5.0) == 10.0


def test_smoother_step_weighted_decays_with_gap():
    s = Smoother(step_decay=0.7)
    s.update("f", 0, 100.0, mode="step_weighted")  # prev = 100
    # gap of 1 step: w = 0.7**1 = 0.7 → 0.3*200 + 0.7*100 = 130
    assert s.update("f", 1, 200.0, mode="step_weighted") == pytest.approx(130.0)
    # a 3-step gap trusts the new value more: w = 0.7**3 = 0.343
    s2 = Smoother(step_decay=0.7)
    s2.update("f", 0, 100.0, mode="step_weighted")
    got = s2.update("f", 3, 200.0, mode="step_weighted")
    expected = (1 - 0.7**3) * 200.0 + 0.7**3 * 100.0
    assert got == pytest.approx(expected)


def test_smoother_histories_are_independent_per_name():
    s = Smoother(ewma_alpha=0.5)
    s.update("t1", 0, 10.0)
    s.update("t2", 0, 100.0)
    assert s.update("t1", 1, 20.0) == 15.0
    assert s.update("t2", 1, 200.0) == 150.0
    assert s.peek("t1") == 15.0
    assert s.peek("t2") == 150.0


def test_smoother_peek_none_before_first_update():
    assert Smoother().peek("never") is None


def test_tools_default_has_no_predictor():
    # Tools holds only the predictor (Phase B binds a real one). Smoothing is a
    # DerivationService, not a tool — so Tools has no smoother.
    assert Tools().predictor is None
    assert not hasattr(Tools(), "smoother")
