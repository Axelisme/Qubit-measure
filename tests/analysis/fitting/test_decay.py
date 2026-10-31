import numpy as np
import pytest
from zcu_tools.analysis.fitting import compute_fit_quality, decay_time_bounds
from zcu_tools.analysis.fitting.base import decaycos, dual_expfunc, expfunc
from zcu_tools.analysis.fitting.decay import (
    fit_decay,
    fit_decay_fringe,
    fit_dual_decay,
    fit_ge_decay,
)

from tests.analysis.fitting._fit_contract import assert_frozen_fields


@pytest.mark.parametrize("fixed_offset", [None, 0.2])
def test_decay_named_result_preserves_unpacking_and_quality(
    fixed_offset: float | None,
) -> None:
    xs = np.linspace(0.0, 20.0, 100)
    ys = 0.2 + 0.8 * np.exp(-xs / 5.0)
    ys += np.random.default_rng(42).normal(0.0, 0.001, xs.size)
    result = fit_decay(xs, ys, fixedparams=(fixed_offset, None, None))
    time, error, curve, (params, cov) = result

    assert isinstance(result, tuple)
    assert result.decay_time == time == pytest.approx(5.0, rel=0.01)
    assert result.decay_time_err == error
    assert result.curve is curve
    assert result.params is params
    assert result.cov is cov
    expected = compute_fit_quality(
        ys,
        curve,
        dict(zip(("y0", "yscale", "decay_time"), params, strict=True)),
        cov,
        decay_bounds=decay_time_bounds(xs),
        fixed_parameters=() if fixed_offset is None else ("y0",),
    )
    assert result.quality == expected
    assert result.quality.invalid == ()
    assert result.quality.r2 is not None and result.quality.r2 > 0.999
    assert error > 0.0
    assert_frozen_fields(result, ("decay_time", "params", "cov", "curve", "quality"))


def test_fit_decay_recovers_T1():
    xs = np.linspace(0, 20, 300)
    true = (0.1, 1.0, 5.0)
    ys = expfunc(xs, *true)
    t1, _, _, (_pOpt, _) = fit_decay(xs, ys)
    assert abs(t1 - 5.0) / 5.0 < 1e-3


@pytest.mark.parametrize("initial_fixed", [None, (0.1, None, None)])
def test_fit_decay_reuses_parameters_for_followup_fit(initial_fixed):
    xs = np.linspace(0.0, 20.0, 300)
    ys = expfunc(xs, 0.1, 1.0, 5.0)
    *_, (initial_params, _) = fit_decay(xs, ys, fixedparams=initial_fixed)
    saved_params = np.asarray(initial_params).copy()

    t1, t1_error, fit_signals, (params, covariance) = fit_decay(
        xs, ys, fit_params=initial_params, fixedparams=(0.1, None, None)
    )

    assert t1 == pytest.approx(5.0, rel=1e-3)
    assert np.isfinite(t1_error)
    assert isinstance(params, list)
    np.testing.assert_allclose(params, [0.1, 1.0, 5.0], rtol=1e-3)
    np.testing.assert_allclose(fit_signals, ys, atol=1e-8)
    np.testing.assert_array_equal(initial_params, saved_params)
    assert covariance.shape == (3, 3)
    np.testing.assert_array_equal(covariance[0, :], 0.0)
    np.testing.assert_array_equal(covariance[:, 0], 0.0)


def test_unresolved_fast_decay_stays_at_sampling_lower_bound() -> None:
    xs = np.linspace(0.5, 5.0, 40)
    ys = expfunc(xs, 0.2, 1.0, 0.02)
    t1, _, fitted, (params, covariance) = fit_decay(xs, ys, fit_params=(0.2, 1.0, 0.02))
    assert t1 == pytest.approx(xs[1] - xs[0], rel=1e-5)
    quality = compute_fit_quality(
        ys,
        fitted,
        dict(zip(("y0", "yscale", "decay_time"), params, strict=True)),
        covariance,
        decay_bounds=decay_time_bounds(xs),
    )
    assert any("窗口內無法解析衰減時間" in issue.reason for issue in quality.invalid)


@pytest.mark.parametrize("fringe", [False, True])
def test_decay_time_is_bounded_by_the_observation_window(fringe: bool) -> None:
    xs = np.linspace(0.5, 5.0, 100)
    if fringe:
        ys = decaycos(xs, 0.2, 1.0, 0.5, 0.0, 1e6)
        time, *_ = fit_decay_fringe(xs, ys)
    else:
        ys = expfunc(xs, 0.2, 1.0, 1e6)
        time, *_ = fit_decay(xs, ys)
    assert time <= 10 * (xs[-1] - xs[0])


@pytest.mark.parametrize("fixed_slow_time", [None, 12.0])
def test_dual_decay_named_result_keeps_longer_time_and_fixed_quality(
    fixed_slow_time: float | None,
) -> None:
    xs = np.linspace(0.0, 60.0, 160)
    ys = 0.2 + 0.3 * np.exp(-xs / 2.0) + 0.6 * np.exp(-xs / 12.0)
    ys += np.random.default_rng(42).normal(0.0, 0.001, xs.size)
    result = fit_dual_decay(
        xs,
        ys,
        fit_params=(0.2, 0.3, 2.0, 0.6, 12.0),
        fixedparams=(0.2, None, None, None, fixed_slow_time),
    )
    slow, slow_err, fast, fast_err, curve, (params, cov) = result

    assert isinstance(result, tuple)
    assert result.decay_time1 == slow == pytest.approx(12.0, rel=0.03)
    assert result.decay_time2 == fast == pytest.approx(2.0, rel=0.03)
    assert result.decay_time1_err == slow_err
    assert result.decay_time2_err == fast_err
    assert result.curve is curve
    assert result.params is params
    assert result.cov is cov
    expected = compute_fit_quality(
        ys,
        curve,
        dict(
            zip(
                ("y0", "yscale1", "decay_time1", "yscale2", "decay_time2"),
                params,
                strict=True,
            )
        ),
        cov,
        decay_bounds=decay_time_bounds(xs),
        fixed_parameters=("y0",) if fixed_slow_time is None else ("y0", "decay_time1"),
    )
    assert result.quality == expected
    assert result.quality.invalid == ()
    assert fast_err > 0.0
    assert (slow_err > 0.0) if fixed_slow_time is None else (slow_err == 0.0)
    assert_frozen_fields(
        result, ("decay_time1", "decay_time2", "params", "cov", "curve", "quality")
    )


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


@pytest.mark.parametrize("phase", [0.0, 180.0])
def test_fixed_fringe_phase_does_not_invalidate_a_resolved_decay(phase: float) -> None:
    xs = np.linspace(0.0, 20.0, 300)
    rng = np.random.default_rng(42)
    ys = decaycos(xs, 0.2, 1.0, 0.5, phase, 5.0)
    ys += rng.normal(0.0, 0.001, xs.size)
    *_, fitted, (params, covariance) = fit_decay_fringe(
        xs, ys, fixedparams=(None, None, None, phase, None)
    )
    assert covariance[3, 3] == 0.0
    quality = compute_fit_quality(
        ys,
        fitted,
        dict(zip(("y0", "yscale", "freq", "phase", "decay_time"), params, strict=True)),
        covariance,
        decay_bounds=decay_time_bounds(xs),
        fixed_parameters=("phase",),
    )
    assert quality.invalid == ()
    assert quality.relative_parameter_errors["phase"] == (None if phase == 0 else 0.0)
    assert quality.r2 is not None and quality.r2 > 0.999


@pytest.mark.parametrize("fixed_phase", [None, 0.0, 180.0])
def test_fringe_named_result_preserves_frequency_and_quality(
    fixed_phase: float | None,
) -> None:
    xs = np.linspace(0.0, 20.0, 160)
    phase = 30.0 if fixed_phase is None else fixed_phase
    ys = 0.2 + 0.8 * np.exp(-xs / 5.0) * np.cos(2 * np.pi * (0.5 * xs + phase / 360))
    ys += np.random.default_rng(42).normal(0.0, 0.001, xs.size)
    result = fit_decay_fringe(
        xs,
        ys,
        fit_params=(0.2, 0.8, 0.5, phase, 5.0),
        fixedparams=(None, None, None, fixed_phase, None),
    )
    time, time_err, freq, freq_err, curve, (params, cov) = result

    assert isinstance(result, tuple)
    assert result.decay_time == time == pytest.approx(5.0, rel=0.01)
    assert result.decay_time_err == time_err
    assert result.freq == freq == pytest.approx(0.5, rel=0.01)
    assert result.freq_err == freq_err
    assert result.curve is curve
    assert result.params is params
    assert result.cov is cov
    expected = compute_fit_quality(
        ys,
        curve,
        dict(zip(("y0", "yscale", "freq", "phase", "decay_time"), params, strict=True)),
        cov,
        decay_bounds=decay_time_bounds(xs),
        fixed_parameters=() if fixed_phase is None else ("phase",),
    )
    assert result.quality == expected
    assert result.quality.invalid == ()
    assert time_err > 0.0 and freq_err > 0.0
    assert_frozen_fields(
        result, ("decay_time", "freq", "params", "cov", "curve", "quality")
    )


def test_fit_decay_fringe_recovers_T2_and_detune():
    xs = np.linspace(0, 10, 400)
    true = (0.0, 1.0, 0.5, 0.0, 4.0)
    ys = decaycos(xs, *true)
    t2f, _, detune, _, _, _ = fit_decay_fringe(xs, ys)
    assert abs(detune - 0.5) < 1e-3
    assert abs(t2f - 4.0) / 4.0 < 5e-2
