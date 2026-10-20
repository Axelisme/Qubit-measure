"""Reusable complex DOP853 interval integration for numba kernel workers.

The explicit 8th-order tableau and embedded 5th/3rd-order estimators are
Hairer's DOP853 coefficients (also used by SciPy). Interval callers stop
exactly at each control/bath boundary and restart with a deterministic seed.
The dense variant retains the degree-seven extension for Hilbert propagators.
A fresh RHS evaluation at each boundary avoids stale derivatives.
Callback-taking helpers inline into kernel-owned compiled entry points. This
keeps process-specific dispatcher pointers out of those entry points' disk cache.
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
    """Local errors, ns step ceiling and optional bare-phase allowance.

    phase_budget is a positive rad allowance used by the density chain runner
    to tighten lab-frame row controls. Infinity disables that extra control.
    norm_size=0 uses state.size. A compressed exact-zero support may
    supply its original dimension, preserving the dense RMS error denominator.
    Removing identically zero errors must not change adaptive step selection.
    All fields are explicit: Numba does not reliably materialize NamedTuple
    defaults when a compiled caller constructs a shortened positional tuple.
    """

    rtol: float
    atol: float
    max_step: float
    phase_budget: float
    norm_size: int


def tolerances(budget: float) -> Tolerances:
    """Return budget-scaled local controls with no fixed time ceiling.

    Relative and absolute errors scale together with the population budget.
    These empirical margins leave room for interval composition, not a global
    error bound. The runner limits steps by active sampled-control spacing so a
    narrow feature cannot hide between stages of a large accepted step.
    """
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("DOP853 budget must be positive and finite")
    # s=.03 passes the 3-tone isolated-feature budget=1e-8 case; .04 fails it.
    scale = 0.03
    return Tolerances(
        min(1e-4, budget * scale), min(1e-6, budget * scale * 0.01), np.inf, np.inf, 0
    )


@njit(nogil=True, cache=True, inline="always")
def _trial(rhs, context, state, time, step):
    """Accumulate nonzero tableau weights in place, reusing one stage buffer."""
    stages = np.empty((12, state.size), dtype=np.complex128)
    stages[0] = rhs(time, state, context)
    intermediate = np.empty_like(state)
    for stage in range(1, 12):
        intermediate[:] = state
        for previous in range(stage):
            weight = step * _A[stage, previous]
            if weight != 0:
                for index in range(state.size):
                    intermediate[index] += weight * stages[previous, index]
        stages[stage] = rhs(time + _C[stage] * step, intermediate, context)
    candidate = state.copy()
    for stage in range(12):
        weight = step * _B[stage]
        if weight != 0:
            for index in range(state.size):
                candidate[index] += weight * stages[stage, index]
    return candidate, stages


@njit(nogil=True, cache=True)
def _error_norm(stages, state, candidate, step, controls):
    norm_size = controls.norm_size
    if norm_size == 0:
        norm_size = state.size
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
    return step * fifth_norm / np.sqrt((fifth_norm + 0.01 * third_norm) * norm_size)


@njit(nogil=True, cache=True, inline="always")
def integrate(
    rhs: Callable[[float, NDArray[np.complex128], Context], NDArray[np.complex128]],
    context: Context,
    state: NDArray[np.complex128],
    duration_ns: float,
    controls: Tolerances,
    first_step: float,
) -> NDArray[np.complex128]:
    """Integrate one smooth interval and return its final state.

    rhs(time_ns, flat_state, context) is a numba-compiled autonomous context
    callback, with time local to this interval. state is a complex128 vector;
    it is not mutated. controls.norm_size can preserve a compressed support's
    original RMS error dimension; omitted exact zeros still count in the dense
    denominator. This does not change rtol/atol or other callers' controls.
    duration_ns is nonnegative. first_step is positive ns,
    capped by max_step; no accepted-step history is returned. Zero duration
    preserves the state. No derivative is reused across
    calls, so controls and bath can jump at boundaries. RuntimeError signals
    nonfinite derivatives/errors, step underflow or exhausted iterations.
    """
    time = 0.0
    step = min(first_step, controls.max_step)
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
            factor = 10.0 if error == 0 else min(10.0, 0.9 * error ** (-1 / 8))
            if rejected:
                factor = min(1.0, factor)
            step *= factor
            rejected = False
        else:
            step *= max(0.2, 0.9 * error ** (-1 / 8))
            rejected = True
    return state


# Extra stages and weights for the degree-seven continuous extension.
_A_EXTRA = np.array(
    [
        [
            0.056167502283047954,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.25350021021662483,
            -0.2462390374708025,
            -0.12419142326381637,
            0.15329179827876568,
            0.00820105229563469,
            0.007567897660545699,
            -0.008298,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.03183464816350214,
            0.0,
            0.0,
            0.0,
            0.0,
            0.028300909672366776,
            0.053541988307438566,
            -0.05492374857139099,
            0.0,
            0.0,
            -0.00010834732869724932,
            0.0003825710908356584,
            -0.00034046500868740456,
            0.1413124436746325,
            0.0,
            0.0,
        ],
        [
            -0.42889630158379194,
            0.0,
            0.0,
            0.0,
            0.0,
            -4.697621415361164,
            7.683421196062599,
            4.06898981839711,
            0.3567271874552811,
            0.0,
            0.0,
            0.0,
            -0.0013990241651590145,
            2.9475147891527724,
            -9.15095847217987,
            0.0,
        ],
    ]
)

_C_EXTRA = np.array([0.1, 0.2, 0.7777777777777778])

_D = np.array(
    [
        [
            -8.428938276109013,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5667149535193777,
            -3.0689499459498917,
            2.38466765651207,
            2.117034582445028,
            -0.871391583777973,
            2.2404374302607883,
            0.6315787787694688,
            -0.08899033645133331,
            18.148505520854727,
            -9.194632392478356,
            -4.436036387594894,
        ],
        [
            10.427508642579134,
            0.0,
            0.0,
            0.0,
            0.0,
            242.28349177525817,
            165.20045171727028,
            -374.5467547226902,
            -22.113666853125306,
            7.733432668472264,
            -30.674084731089398,
            -9.332130526430229,
            15.697238121770845,
            -31.139403219565178,
            -9.35292435884448,
            35.81684148639408,
        ],
        [
            19.985053242002433,
            0.0,
            0.0,
            0.0,
            0.0,
            -387.0373087493518,
            -189.17813819516758,
            527.8081592054236,
            -11.57390253995963,
            6.8812326946963,
            -1.0006050966910838,
            0.7777137798053443,
            -2.778205752353508,
            -60.19669523126412,
            84.32040550667716,
            11.99229113618279,
        ],
        [
            -25.69393346270375,
            0.0,
            0.0,
            0.0,
            0.0,
            -154.18974869023643,
            -231.5293791760455,
            357.6391179106141,
            93.40532418362432,
            -37.45832313645163,
            104.0996495089623,
            29.8402934266605,
            -43.53345659001114,
            96.32455395918828,
            -39.17726167561544,
            -149.72683625798564,
        ],
    ]
)


@njit(nogil=True, cache=True, inline="always")
def _dense_step(rhs, context, state, trial, time, step):
    """Build an accepted step polynomial with fused, nonzero weighted sums."""
    candidate, stages = trial
    extended = np.empty((16, state.size), dtype=np.complex128)
    extended[:12] = stages
    extended[12] = rhs(time + step, candidate, context)
    for stage in range(13, 16):
        intermediate = state.copy()
        for previous in range(stage):
            weight = step * _A_EXTRA[stage - 13, previous]
            if weight != 0:
                for index in range(state.size):
                    intermediate[index] += weight * extended[previous, index]
        extended[stage] = rhs(time + _C_EXTRA[stage - 13] * step, intermediate, context)
    dense = np.empty((8, state.size), dtype=np.complex128)
    dense[0] = state
    delta = candidate - state
    dense[1] = delta
    dense[2] = step * extended[0] - delta
    dense[3] = 2 * delta - step * (extended[12] + extended[0])
    for degree in range(4):
        dense[degree + 4] = 0.0
        for stage in range(16):
            weight = step * _D[degree, stage]
            if weight != 0:
                for index in range(state.size):
                    dense[degree + 4, index] += weight * extended[stage, index]
    if not np.isfinite(dense).all():
        raise RuntimeError("DOP853 dense nonfinite polynomial")
    return dense


@njit(nogil=True, cache=True, inline="always")
def integrate_dense(
    rhs: Callable[[float, NDArray[np.complex128], Context], NDArray[np.complex128]],
    context: Context,
    state: NDArray[np.complex128],
    duration: float,
    controls: Tolerances,
) -> tuple[NDArray[np.float64], NDArray[np.complex128], NDArray[np.complex128]]:
    """Integrate an interval and retain only finite accepted-step polynomials.

    Each polynomial stores its left state followed by seven nested coefficients.
    The same 8(5,3) step acceptance and degree-seven extension as DOP853 are used.
    All stages, error estimates and dense construction run without the GIL.
    Failed steps are discarded; nonfinite output or step exhaustion raises.
    """
    time = 0.0
    step = controls.max_step
    times = [0.0]
    coefficients = []
    rejected = False
    attempts = 0
    while time < duration:
        attempts += 1
        step = min(step, controls.max_step, duration - time)
        if attempts > 10000000 or step <= 0 or time + step == time:
            raise RuntimeError("DOP853 dense step underflow or iteration limit")
        candidate, stages = _trial(rhs, context, state, time, step)
        error = _error_norm(stages, state, candidate, step, controls)
        if not np.isfinite(error) or not np.isfinite(candidate).all():
            raise RuntimeError("DOP853 dense nonfinite solution or error estimate")
        if error <= 1:
            dense = _dense_step(rhs, context, state, (candidate, stages), time, step)
            coefficients.append(dense)
            time += step
            times.append(time)
            state = candidate
            factor = 10.0 if error == 0 else min(10.0, 0.9 * error ** (-1 / 8))
            if rejected:
                factor = min(1.0, factor)
            step *= factor
            rejected = False
        else:
            step *= max(0.2, 0.9 * error ** (-1 / 8))
            rejected = True
    packed = np.empty((len(coefficients), 8, state.size), dtype=np.complex128)
    for index in range(len(coefficients)):
        packed[index] = coefficients[index]
    return np.array(times), packed, state
