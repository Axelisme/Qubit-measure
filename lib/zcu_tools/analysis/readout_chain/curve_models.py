"""Per-state curve models of the Rabi, T1, T2 echo and T2 Ramsey cycles.

Rabi and T2 constructors return CycleModel, with state curves and free affine
endpoints absorbing amplitude, sign and imperfect pulses. T1 instead returns
the detailed-balance RateModel, with a free initial stochastic drive.
"""

from collections.abc import Mapping

import numpy as np

from .model import Array, CycleModel
from .rate_model import RateModel


def rabi_model() -> CycleModel:
    """Return a three-state model for a Rabi drive of frequency f.

    Axis x is the swept drive amount (gain, or pulse length) in caller units.
    The fitted ``f`` is the Rabi frequency in cycles per axis unit, bounds
    [0, infinity), so the π point is x = 1 / (2 f). Each row interpolates
    between two free simplex endpoints using its own curve weight. The g row
    uses sin²(π f x), and the e row uses cos²(π f x). The third row uses the
    same sin² weight as the g row; it is not fixed as undriven. Identical
    endpoints allow a flat response, but the third row need not be flat.
    ``oscillation_frequency`` is ``f``, so CycleFit reports reliability.
    """

    def curve(x: Array, parameters: Mapping[str, float]) -> Array:
        from_g = np.sin(np.pi * parameters["f"] * x) ** 2
        from_e = np.cos(np.pi * parameters["f"] * x) ** 2
        return np.column_stack([from_g, from_e, from_g])

    def derivative(x: Array, parameters: Mapping[str, float]) -> Array:
        slope = np.pi * x * np.sin(2 * np.pi * parameters["f"] * x)
        return np.column_stack([slope, -slope, slope])[:, :, None]

    return CycleModel(
        curve, ("f",), [(0.0, np.inf)], oscillation_frequency="f", derivative=derivative
    )


def t1_model() -> RateModel:
    """Return D(x) = D0 exp(Gx), the detailed-balance T1 model.

    Axis is nonnegative wait time. Three pair rates and the thermal stationary
    distribution define G; D0 is free stochastic. fit_cycle takes
    init={"t1": textbook_time}. It reports both eigen times and selects the
    larger-amplitude one as T1. See RateModel and CycleRelaxation for units,
    the geometric-state labels and the amplitude definition.
    """
    return RateModel()


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
    can produce larger joint residuals; fit_cycle does not correct the model.
    Equal t2/tf or vanishing row contrast can lose identification; Wald errors
    then need not be calibrated. The g curve is affine in exp(-x/t2), the same
    family as the third curve, so with a small third-state population the fit
    can put g in the third row with t2 and tf exchanged; label the rows by
    CycleFit.nuisances.thermal. Direct curve calls raise
    ValueError for negative/nonfinite delay; parameter validation is owned by
    CycleModel.
    """

    def curve(axis: Array, parameters: Mapping[str, float]) -> Array:
        return _curves(axis, parameters, 0.0)

    return CycleModel(
        curve,
        ("t2", "tf"),
        ((np.finfo(float).tiny, np.inf),) * 2,
        derivative=lambda x, p: _derivatives(x, p, ramsey=False),
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
    complex leakage can produce larger joint residuals.

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
        derivative=lambda x, p: _derivatives(x, p, ramsey=True),
    )


def _derivatives(
    axis: Array, parameters: Mapping[str, float], *, ramsey: bool
) -> Array:
    # Multiplying by decay before dividing avoids overflow at the positive time bound.
    curves = _curves(axis, parameters, parameters["detuning"] if ramsey else 0.0)
    decay = np.exp(-axis / parameters["t2"])
    phase = 2 * np.pi * parameters["detuning"] * axis if ramsey else np.zeros_like(axis)
    result = np.zeros((len(axis), 3, 3 if ramsey else 2))
    slope = (axis / parameters["t2"]) * decay / parameters["t2"] * np.cos(phase) / 2
    result[:, 0, 0], result[:, 1, 0] = -slope, slope
    result[:, 2, -1] = (axis / parameters["tf"]) * curves[:, 2] / parameters["tf"]
    if ramsey:
        frequency_slope = -np.pi * axis * decay * np.sin(phase)
        result[:, 0, 1], result[:, 1, 1] = -frequency_slope, frequency_slope
    return result


def _curves(axis: Array, parameters: Mapping[str, float], frequency: float) -> Array:
    if not np.isfinite(axis).all() or np.any(axis < 0):
        raise ValueError("T2 delay axis must be finite and nonnegative")
    modulation = np.cos(2 * np.pi * frequency * axis)
    transverse = (1 + np.exp(-axis / parameters["t2"]) * modulation) / 2
    return np.column_stack(
        [1 - transverse, transverse, np.exp(-axis / parameters["tf"])]
    )
