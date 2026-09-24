import numpy as np
import pytest
from zcu_tools.utils.fitting.base import decaycos, dual_expfunc, expfunc
from zcu_tools.utils.fitting.decay import (
    fit_decay,
    fit_decay_fringe,
    fit_dual_decay,
    fit_ge_decay,
)


def test_fit_decay_recovers_T1():
    xs = np.linspace(0, 20, 300)
    true = (0.1, 1.0, 5.0)
    ys = expfunc(xs, *true)
    t1, _, _, (pOpt, _) = fit_decay(xs, ys)
    assert abs(t1 - 5.0) / 5.0 < 1e-3


def test_fit_dual_decay_recovers_two_times():
    xs = np.linspace(0, 60, 600)
    true = (0.0, 0.5, 2.0, 0.5, 12.0)
    ys = dual_expfunc(xs, *true)
    t1, _, t1b, _, _, _ = fit_dual_decay(xs, ys)
    shorter = min(t1, t1b)
    longer = max(t1, t1b)
    assert abs(longer - 12.0) / 12.0 < 5e-2
    assert abs(shorter - 2.0) / 2.0 < 5e-2


def test_fit_ge_decay_shared_t1():
    rng = np.random.default_rng(20260901)
    times = np.linspace(0, 20, 300)
    g_true = (0.1, 0.8, 6.0)
    e_true = (0.9, -0.8, 6.0)
    g_pops = expfunc(times, *g_true) + rng.normal(0.0, 2e-3, times.size)
    e_pops = expfunc(times, *e_true) + rng.normal(0.0, 2e-3, times.size)
    (g_t1, g_t1err, _, _), (e_t1, e_t1err, _, _) = fit_ge_decay(
        times, g_pops, e_pops, share_t1=True
    )
    assert abs(g_t1 - 6.0) / 6.0 < 1e-2
    assert g_t1 == e_t1
    assert np.isfinite(g_t1err)
    assert g_t1err > 0.0
    assert g_t1err == e_t1err


def test_fit_ge_decay_non_shared_t1_remains_independent():
    times = np.linspace(0, 20, 300)
    g_pops = expfunc(times, 0.1, 0.8, 5.0)
    e_pops = expfunc(times, 0.9, -0.8, 7.0)

    (g_t1, _, _, _), (e_t1, _, _, _) = fit_ge_decay(
        times, g_pops, e_pops, share_t1=False
    )

    assert g_t1 == pytest.approx(5.0, rel=1e-3)
    assert e_t1 == pytest.approx(7.0, rel=1e-3)


def test_fit_decay_fringe_recovers_T2_and_detune():
    xs = np.linspace(0, 10, 400)
    true = (0.0, 1.0, 0.5, 0.0, 4.0)
    ys = decaycos(xs, *true)
    t2f, _, detune, _, _, _ = fit_decay_fringe(xs, ys)
    assert abs(detune - 0.5) < 1e-3
    assert abs(t2f - 4.0) / 4.0 < 5e-2
