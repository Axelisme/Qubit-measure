from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from zcu_tools.experiment.v2.twotone.time_domain.t1 import quantize_t1_delays
from zcu_tools.experiment.v2.twotone.time_domain.t1_axis import t1_delay_axis


def _arc_target_residuals(
    axis: np.ndarray, *, start: float, stop: float, model_t1: float
) -> np.ndarray:
    normalized = (axis - start) / (stop - start)
    lambda_value = (stop - start) / model_t1
    denominator = -np.expm1(-lambda_value)

    def speed(x: float) -> float:
        slope = lambda_value * np.exp(-lambda_value * x) / denominator
        return float(np.hypot(1.0, slope))

    total, _error = quad(speed, 0.0, 1.0, epsabs=1e-13, epsrel=1e-13)
    residuals = []
    for index, x in enumerate(normalized):
        cumulative, _error = quad(speed, 0.0, float(x), epsabs=1e-13, epsrel=1e-13)
        target = index * total / (len(axis) - 1)
        residuals.append(abs(cumulative - target) / total)
    return np.asarray(residuals)


@pytest.mark.parametrize(
    ("start", "stop", "expts", "model_t1"),
    [
        (0.0, 80.0, 30, 16.0),
        (10.0, 80.0, 17, 16.0),
        (0.0, 1.001e-4, 9, 1.0),
        (0.0, 0.999e-4, 9, 1.0),
    ],
)
def test_nonuniform_axis_has_equal_normalized_arc_lengths(
    start: float, stop: float, expts: int, model_t1: float
) -> None:
    axis = t1_delay_axis(
        start=start,
        stop=stop,
        expts=expts,
        uniform=False,
        model_t1=model_t1,
    )

    assert len(axis) == expts
    assert axis[0] == start
    assert axis[-1] == stop
    assert np.all(np.isfinite(axis))
    assert np.all(np.diff(axis) > 0.0)
    residuals = _arc_target_residuals(axis, start=start, stop=stop, model_t1=model_t1)
    assert np.max(residuals) <= 1e-10


def test_nonuniform_axis_uses_straight_line_when_lambda_underflows() -> None:
    axis = t1_delay_axis(
        start=0.0,
        stop=1e-300,
        expts=5,
        uniform=False,
        model_t1=1e300,
    )

    np.testing.assert_array_equal(axis, np.linspace(0.0, 1e-300, 5))


def test_nonuniform_axis_rejects_unrepresentable_underflow_window() -> None:
    stop = np.nextafter(0.0, 1.0)

    with pytest.raises(
        ValueError, match="invalid non-uniform T1 sampling domain"
    ) as exc_info:
        t1_delay_axis(
            start=0.0,
            stop=stop,
            expts=3,
            uniform=False,
            model_t1=2.0,
        )

    message = str(exc_info.value)
    assert "start=0.0" in message
    assert f"stop={stop!r}" in message
    assert "expts=3" in message


def test_uniform_axis_preserves_existing_single_point_behavior() -> None:
    axis = t1_delay_axis(
        start=2.0,
        stop=2.0,
        expts=1,
        uniform=True,
        model_t1=np.nan,
    )

    np.testing.assert_array_equal(axis, np.asarray([2.0]))


@pytest.mark.parametrize(
    ("start", "stop", "expts", "model_t1"),
    [
        (-1.0, 1.0, 3, 1.0),
        (0.0, 0.0, 3, 1.0),
        (2.0, 1.0, 3, 1.0),
        (0.0, 1.0, 1, 1.0),
        (0.0, 1.0, 3, 0.0),
        (0.0, 1.0, 3, np.nan),
        (0.0, 6.0, 3, 1.0),
        (0.0, np.finfo(np.float64).max, 3, np.finfo(np.float64).tiny),
        (np.nan, 1.0, 3, 1.0),
        (0.0, np.inf, 3, 1.0),
    ],
)
def test_nonuniform_axis_rejects_invalid_domain(
    start: float, stop: float, expts: int, model_t1: float
) -> None:
    with pytest.raises(ValueError) as exc_info:
        t1_delay_axis(
            start=start,
            stop=stop,
            expts=expts,
            uniform=False,
            model_t1=model_t1,
        )

    message = str(exc_info.value)
    assert "invalid non-uniform T1 sampling domain" in message
    assert "start=" in message
    assert "stop=" in message
    assert "expts=" in message


class _CycleConfig:
    def us2cycles(self, delay: float) -> int:
        return int(np.rint(10.0 * delay))

    def cycles2us(self, cycles: int) -> float:
        return cycles / 10.0


@pytest.mark.parametrize(
    "delays",
    [
        [0.0, 0.11, 0.29, 0.52],
        np.asarray([0.0, 0.11, 0.29, 0.52], dtype=np.float64),
    ],
)
def test_delay_quantization_preserves_count_and_order(
    delays: list[float] | np.ndarray,
) -> None:
    cycles, quantized = quantize_t1_delays(_CycleConfig(), delays)

    np.testing.assert_array_equal(cycles, np.asarray([0, 1, 3, 5], dtype=np.int32))
    np.testing.assert_array_equal(quantized, np.asarray([0.0, 0.1, 0.3, 0.5]))


def test_generated_axis_collision_is_not_silently_deduplicated() -> None:
    generated = t1_delay_axis(
        start=0.0,
        stop=0.04,
        expts=3,
        uniform=False,
        model_t1=0.008,
    )

    with pytest.raises(
        ValueError, match="delay sweep collapsed after cycle quantization"
    ):
        quantize_t1_delays(_CycleConfig(), generated)


@pytest.mark.parametrize(
    "delays",
    [
        [0.0, 0.04, 0.11],
        [0.0, 0.3, 0.2],
    ],
)
def test_direct_delay_quantization_rejects_collisions_or_reordering(
    delays: list[float],
) -> None:
    with pytest.raises(ValueError) as exc_info:
        quantize_t1_delays(_CycleConfig(), delays)

    message = str(exc_info.value)
    assert "delay sweep collapsed after cycle quantization" in message
    assert f"start={delays[0]!r}" in message
    assert f"stop={delays[-1]!r}" in message
    assert f"expts={len(delays)}" in message
    assert "increase the delay span" in message
    assert "reduce the number of points" in message
