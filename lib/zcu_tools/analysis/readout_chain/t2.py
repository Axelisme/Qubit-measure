"""Ideal-pulse transverse-decay curves for three-state cycles."""

from collections.abc import Mapping

import numpy as np

from .model import Array, CycleModel


def t2_echo_model() -> CycleModel:
    """Return a three-state model for ideal Hahn echo versus total free delay.

    No calibration inputs are required. Axis x is finite, nonnegative total
    free-evolution time (both half delays), in caller-selected time units.
    Fitted parameters ``t2`` and ``tf`` use those same units, with bounds
    [smallest positive normal float, infinity). The g/e curves are
    (1 - exp(-x/t2))/2 and (1 + exp(-x/t2))/2, respectively;
    the third-state curve is exp(-x/tf). These g/e curves are P(e) after
    the sequence starting from g/e (inverse-phase analyzer convention).
    ``tf`` is the lifetime for third-state population leaving the subspace.
    Each row's two free probability endpoints absorb amplitude and sign.

    The pulses are instantaneous ideal pi/2, pi, pi/2 rotations. Longitudinal
    T1 relaxation during the delays does not project onto the measured axis
    after the last pi/2; it is not an extra fitted parameter or fixed input.
    T2 includes the transverse contribution of T1. Third-state decay lands
    incoherently in the qubit subspace. Pulse errors or more complex leakage
    can violate this curve family; fit_cycle reports checks, not corrections.
    Equal t2/tf or vanishing row contrast can lose identification; Wald errors
    then need not be calibrated. Direct curve calls raise ValueError for
    negative/nonfinite delay; parameter validation is owned by CycleModel.
    """

    def curve(axis: Array, parameters: Mapping[str, float]) -> Array:
        return _curves(axis, parameters, 0.0)

    return CycleModel(
        curve,
        ("t2", "tf"),
        ((np.finfo(float).tiny, np.inf),) * 2,
    )


def t2_ramsey_model() -> CycleModel:
    """Return a three-state model for ideal Ramsey versus free-evolution delay.

    No calibration inputs are required. Axis x is finite, nonnegative delay in
    caller-selected time units. Fitted ``t2`` and ``tf`` use those units, with
    bounds [smallest positive normal float, infinity). Fitted ``detuning`` is
    nonnegative, in cycles per time unit, with bounds [0, infinity). Its sign
    is not identifiable from a cosine. The g/e curves are
    (1 - c)/2 and (1 + c)/2, respectively, where
    c = exp(-x/t2)*cos(2*pi*detuning*x). They represent P(e) starting
    from g/e with an inverse-phase analyzer. The third curve is exp(-x/tf).
    ``tf`` is the lifetime for third-state population leaving the subspace.
    Free affine probability endpoints absorb each row's amplitude and sign.

    Instantaneous ideal pi/2 pulses project only transverse coherence onto
    the measured population. T1 relaxation in the delay does not add a
    longitudinal baseline after the analyzer. T2 includes T1's transverse
    contribution; no separate T1 input or parameter is used. Third-state
    decay lands incoherently in the qubit subspace. Pulse errors and more
    complex leakage can cause misfit, reported by fit_cycle's checks.

    ``oscillation_frequency`` is ``detuning`` so CycleFit reports reliability.
    Zero detuning, vanishing contrast and frequency aliases can invalidate
    local Wald errors; fit_cycle searches only the supplied initial basin.
    Direct curve calls raise ValueError for negative/nonfinite delay;
    parameter validation is owned by CycleModel.
    """

    def curve(axis: Array, parameters: Mapping[str, float]) -> Array:
        return _curves(axis, parameters, parameters["detuning"])

    return CycleModel(
        curve,
        ("t2", "detuning", "tf"),
        (
            (np.finfo(float).tiny, np.inf),
            (0.0, np.inf),
            (np.finfo(float).tiny, np.inf),
        ),
        oscillation_frequency="detuning",
    )


def _curves(axis: Array, parameters: Mapping[str, float], frequency: float) -> Array:
    if not np.isfinite(axis).all() or np.any(axis < 0):
        raise ValueError("T2 delay axis must be finite and nonnegative")
    modulation = np.cos(2 * np.pi * frequency * axis)
    transverse = (1 + np.exp(-axis / parameters["t2"]) * modulation) / 2
    return np.column_stack(
        [1 - transverse, transverse, np.exp(-axis / parameters["tf"])]
    )
