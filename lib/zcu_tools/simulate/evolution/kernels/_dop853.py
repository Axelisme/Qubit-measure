"""Reusable complex DOP853 interval integration for numba kernel workers.

The explicit 8th-order tableau and embedded 5th/3rd-order estimators are
Hairer's DOP853 coefficients (also used by SciPy). Dense output is not needed:
callers stop exactly at each control/bath boundary and restart with the last
accepted step. A fresh RHS evaluation at each boundary avoids stale derivatives.
"""

from collections.abc import Callable
from typing import NamedTuple, TypeVar

import numpy as np
from numba import njit
from numpy.typing import NDArray

Context = TypeVar("Context")

# Numerical tableau: only the twelve integration stages are required.
_A = np.zeros((12, 12))
_A[1, :1] = ((0.05260015195876773),)
_A[2, :2] = ((0.0197250569845379), (0.0591751709536137))
_A[3, :3] = ((0.02958758547680685), (0.0), (0.08876275643042054))
_A[4, :4] = ((0.2413651341592667), (0.0), (-0.8845494793282861), (0.924834003261792))
_A[5, :5] = (
    (0.037037037037037035),
    (0.0),
    (0.0),
    (0.17082860872947386),
    (0.12546768756682242),
)
_A[6, :6] = (
    (0.037109375),
    (0.0),
    (0.0),
    (0.17025221101954405),
    (0.06021653898045596),
    (-0.017578125),
)
_A[7, :7] = (
    (0.03709200011850479),
    (0.0),
    (0.0),
    (0.17038392571223998),
    (0.10726203044637328),
    (-0.015319437748624402),
    (0.008273789163814023),
)
_A[8, :8] = (
    (0.6241109587160757),
    (0.0),
    (0.0),
    (-3.3608926294469414),
    (-0.868219346841726),
    (27.59209969944671),
    (20.154067550477894),
    (-43.48988418106996),
)
_A[9, :9] = (
    (0.47766253643826434),
    (0.0),
    (0.0),
    (-2.4881146199716677),
    (-0.590290826836843),
    (21.230051448181193),
    (15.279233632882423),
    (-33.28821096898486),
    (-0.020331201708508627),
)
_A[10, :10] = (
    (-0.9371424300859873),
    (0.0),
    (0.0),
    (5.186372428844064),
    (1.0914373489967295),
    (-8.149787010746927),
    (-18.52006565999696),
    (22.739487099350505),
    (2.4936055526796523),
    (-3.0467644718982196),
)
_A[11, :11] = (
    (2.273310147516538),
    (0.0),
    (0.0),
    (-10.53449546673725),
    (-2.0008720582248625),
    (-17.9589318631188),
    (27.94888452941996),
    (-2.8589982771350235),
    (-8.87285693353063),
    (12.360567175794303),
    (0.6433927460157636),
)
_B = np.array(
    [
        0.054293734116568765,
        0.0,
        0.0,
        0.0,
        0.0,
        4.450312892752409,
        1.8915178993145003,
        -5.801203960010585,
        0.3111643669578199,
        -0.1521609496625161,
        0.20136540080403034,
        0.04471061572777259,
    ]
)
_C = np.array(
    [
        0.0,
        0.05260015195876773,
        0.0789002279381516,
        0.1183503419072274,
        0.2816496580927726,
        0.3333333333333333,
        0.25,
        0.3076923076923077,
        0.6512820512820513,
        0.6,
        0.8571428571428571,
        1.0,
    ]
)
_E3 = np.array(
    [
        -0.18980075407240762,
        0.0,
        0.0,
        0.0,
        0.0,
        4.450312892752409,
        1.8915178993145003,
        -5.801203960010585,
        -0.4226823213237919,
        -0.1521609496625161,
        0.20136540080403034,
        0.02265179219836082,
    ]
)
_E5 = np.array(
    [
        0.01312004499419488,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.2251564463762044,
        -0.4957589496572502,
        1.6643771824549864,
        -0.35032884874997366,
        0.3341791187130175,
        0.08192320648511571,
        -0.022355307863886294,
    ]
)


class Tolerances(NamedTuple):
    """Adaptive controls: positive relative/absolute errors and ns step ceiling."""

    rtol: float
    atol: float
    max_step: float = 0.1


def tolerances(budget: float) -> Tolerances:
    """Return default 1e-5/1e-7 controls, tightened for small absolute budgets.

    budget is a positive finite per-interval population/density allowance.
    The safety margin reserves error for interpolation and interval composition;
    the maximum accepted step is always at most 0.1 ns.
    """
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("DOP853 budget must be positive and finite")
    return Tolerances(min(1e-5, budget * 0.01), min(1e-7, budget * 0.0001))


@njit(nogil=True, cache=True)
def _trial(rhs, context, state, time, step):
    stages = np.empty((12, state.size), dtype=np.complex128)
    stages[0] = rhs(time, state, context)
    for stage in range(1, 12):
        intermediate = state.copy()
        for previous in range(stage):
            intermediate += step * _A[stage, previous] * stages[previous]
        stages[stage] = rhs(time + _C[stage] * step, intermediate, context)
    candidate = state.copy()
    for stage in range(12):
        candidate += step * _B[stage] * stages[stage]
    return candidate, stages


@njit(nogil=True, cache=True)
def _error_norm(stages, state, candidate, step, controls):
    fifth_norm = 0.0
    third_norm = 0.0
    for index in range(state.size):
        scale = controls.atol + controls.rtol * max(
            abs(state[index]), abs(candidate[index])
        )
        fifth = 0j
        third = 0j
        for stage in range(12):
            fifth += _E5[stage] * stages[stage, index]
            third += _E3[stage] * stages[stage, index]
        fifth_norm += abs(fifth / scale) ** 2
        third_norm += abs(third / scale) ** 2
    if fifth_norm == 0 and third_norm == 0:
        return 0.0
    return step * fifth_norm / np.sqrt((fifth_norm + 0.01 * third_norm) * state.size)


@njit(nogil=True, cache=True)
def integrate(
    rhs: Callable[[float, NDArray[np.complex128], Context], NDArray[np.complex128]],
    context: Context,
    state: NDArray[np.complex128],
    duration_ns: float,
    controls: Tolerances,
    first_step: float,
) -> tuple[NDArray[np.complex128], float]:
    """Integrate one smooth interval, returning final state and accepted step.

    rhs(time_ns, flat_state, context) is a numba-compiled autonomous context
    callback, with time local to this interval. state is a complex128 vector;
    it is not mutated. duration_ns is nonnegative. first_step is positive ns,
    usually the previous interval's last accepted step, or max_step initially.
    Zero duration preserves both state and step. No derivative is reused across
    calls, so controls and bath can jump at boundaries. RuntimeError signals
    nonfinite derivatives/errors, step underflow or exhausted iterations.
    """
    time = 0.0
    step = min(first_step, controls.max_step)
    accepted_step = first_step
    rejected = False
    attempts = 0
    while time < duration_ns:
        attempts += 1
        step = min(step, controls.max_step, duration_ns - time)
        if attempts > 10000000 or step <= 0 or time + step == time:
            raise RuntimeError("DOP853 step underflow or iteration limit")
        candidate, stages = _trial(rhs, context, state, time, step)
        error = _error_norm(stages, state, candidate, step, controls)
        if not np.isfinite(error) or not np.isfinite(candidate).all():
            raise RuntimeError("DOP853 nonfinite solution or error estimate")
        if error <= 1:
            state = candidate
            time += step
            accepted_step = step
            factor = 10.0 if error == 0 else min(10.0, 0.9 * error ** (-1 / 8))
            if rejected:
                factor = min(1.0, factor)
            step *= factor
            rejected = False
        else:
            step *= max(0.2, 0.9 * error ** (-1 / 8))
            rejected = True
    return state, accepted_step
