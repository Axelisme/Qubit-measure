from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from qick import QickConfig
from scipy.integrate import quad
from scipy.optimize import brentq

from zcu_tools.program.v2 import SweepCfg

from .round_zcu import sweep2array

_DOMAIN_ERROR = "invalid non-uniform T1 sampling domain"
_ANALYTIC_LAMBDA_MIN = 1e-4
_MAX_LAMBDA = 5.0
_ROOT_XTOL = 1e-12
_ROOT_RTOL = 1e-12
_QUAD_EPSABS = 1e-13
_QUAD_EPSREL = 1e-13

T1Sweep = SweepCfg | list[float] | NDArray[np.float64]


class T1CycleConverter(Protocol):
    def us2cycles(self, delay: float) -> int: ...

    def cycles2us(self, cycles: int) -> float: ...


@dataclass(frozen=True)
class T1DelayTable:
    cycles: NDArray[np.int32]
    times_us: NDArray[np.float64]


def _domain_error(*, start: object, stop: object, expts: object) -> ValueError:
    return ValueError(
        f"{_DOMAIN_ERROR}: start={start!r}, stop={stop!r}, expts={expts!r}"
    )


def _validate_nonuniform_domain(
    *, start: float, stop: float, expts: int, model_t1: float
) -> tuple[float, float, int, float]:
    if isinstance(expts, bool) or not isinstance(expts, Integral):
        raise _domain_error(start=start, stop=stop, expts=expts)

    try:
        start_value = float(start)
        stop_value = float(stop)
        count = int(expts)
        model_t1_value = float(model_t1)
    except (TypeError, ValueError, OverflowError):
        raise _domain_error(start=start, stop=stop, expts=expts) from None
    if (
        not np.isfinite(start_value)
        or not np.isfinite(stop_value)
        or start_value < 0.0
        or start_value >= stop_value
        or count < 2
        or not np.isfinite(model_t1_value)
        or model_t1_value <= 0.0
    ):
        raise _domain_error(start=start, stop=stop, expts=expts)

    span = stop_value - start_value
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        lambda_value = float(np.divide(span, model_t1_value))
    if (
        lambda_value < 0.0
        or not np.isfinite(lambda_value)
        or lambda_value > _MAX_LAMBDA
    ):
        raise _domain_error(start=start, stop=stop, expts=expts)

    return start_value, stop_value, count, lambda_value


def _arc_length_function(lambda_value: float) -> Callable[[float], float]:
    y0 = lambda_value / -np.expm1(-lambda_value)

    if lambda_value >= _ANALYTIC_LAMBDA_MIN:
        r0 = np.hypot(1.0, y0)

        def arc_length(x: float) -> float:
            y = y0 * np.exp(-lambda_value * x)
            radius = np.hypot(1.0, y)
            log_ratio = np.log1p((radius - r0) / (1.0 + r0))
            return float(x + (r0 - radius + log_ratio) / lambda_value)

        return arc_length

    def speed(x: float) -> float:
        return float(np.hypot(1.0, y0 * np.exp(-lambda_value * x)))

    def arc_length(x: float) -> float:
        value, _error = quad(
            speed,
            0.0,
            x,
            epsabs=_QUAD_EPSABS,
            epsrel=_QUAD_EPSREL,
        )
        return float(value)

    return arc_length


def t1_delay_axis(
    *,
    start: float,
    stop: float,
    expts: int,
    uniform: bool,
    model_t1: float,
) -> NDArray[np.float64]:
    """Return ideal T1 delay coordinates before hardware quantization."""
    if uniform:
        return np.linspace(start, stop, expts, endpoint=True, dtype=np.float64)

    start_value, stop_value, count, lambda_value = _validate_nonuniform_domain(
        start=start,
        stop=stop,
        expts=expts,
        model_t1=model_t1,
    )
    if lambda_value == 0.0:
        axis = np.linspace(
            start_value, stop_value, count, endpoint=True, dtype=np.float64
        )
    else:
        arc_length = _arc_length_function(lambda_value)
        total_arc_length = arc_length(1.0)
        normalized_points = np.empty(count, dtype=np.float64)
        normalized_points[0] = 0.0
        normalized_points[-1] = 1.0
        for index in range(1, count - 1):
            target = index * total_arc_length / (count - 1)
            normalized_points[index] = cast(
                float,
                brentq(
                    lambda x: arc_length(x) - target,
                    0.0,
                    1.0,
                    xtol=_ROOT_XTOL,
                    rtol=np.float64(_ROOT_RTOL),
                ),
            )

        axis = start_value + (stop_value - start_value) * normalized_points
        axis[0] = start_value
        axis[-1] = stop_value
    if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
        raise _domain_error(start=start, stop=stop, expts=expts)
    return axis


def _ideal_nonuniform_axis(sweep: T1Sweep) -> NDArray[np.float64]:
    if isinstance(sweep, SweepCfg):
        return t1_delay_axis(
            start=sweep.start,
            stop=sweep.stop,
            expts=sweep.expts,
            uniform=False,
            model_t1=0.2 * sweep.stop,
        )
    return _validate_quantized_axis(sweep, sweep)


def _validate_quantized_axis(
    ideal_delays: list[float] | NDArray[np.float64],
    quantized_delays: list[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
    ideal = np.asarray(ideal_delays, dtype=np.float64)
    quantized = np.asarray(quantized_delays, dtype=np.float64)
    start = float(ideal.flat[0]) if ideal.size else None
    stop = float(ideal.flat[-1]) if ideal.size else None
    expts = int(ideal.size)

    if (
        ideal.ndim != 1
        or ideal.size == 0
        or not np.all(np.isfinite(ideal))
        or quantized.shape != ideal.shape
        or not np.all(np.isfinite(quantized))
        or np.any(np.diff(quantized) <= 0)
    ):
        raise ValueError(
            "delay sweep collapsed after cycle quantization: "
            f"start={start!r}, stop={stop!r}, expts={expts}; "
            "increase the delay span or reduce the number of points"
        )

    return quantized


def materialize_nonuniform_t1_delays(
    sweep: T1Sweep,
    *,
    soccfg: T1CycleConverter,
) -> T1DelayTable:
    """Materialize a non-uniform T1 axis on the tProcessor delay grid."""
    ideal_delays = _ideal_nonuniform_axis(sweep)
    cycles = np.asarray(
        [int(soccfg.us2cycles(float(delay))) for delay in ideal_delays],
        dtype=np.int32,
    )
    times_us = np.asarray(
        [soccfg.cycles2us(int(cycle)) for cycle in cycles],
        dtype=np.float64,
    )
    return T1DelayTable(
        cycles=cycles,
        times_us=_validate_quantized_axis(ideal_delays, times_us),
    )


def materialize_nonuniform_t1_pulse_lengths(
    sweep: T1Sweep,
    *,
    soccfg: QickConfig,
    gen_ch: int,
) -> NDArray[np.float64]:
    """Materialize a non-uniform T1 axis on one generator's pulse-length grid."""
    ideal_lengths = _ideal_nonuniform_axis(sweep)
    quantized_lengths = sweep2array(
        ideal_lengths,
        "time",
        {"soccfg": soccfg, "gen_ch": gen_ch},
        allow_array=True,
    )
    return _validate_quantized_axis(ideal_lengths, quantized_lengths)
