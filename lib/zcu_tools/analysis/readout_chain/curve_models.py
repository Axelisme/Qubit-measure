"""Per-state curve models of the Rabi, T1, T2 echo and T2 Ramsey cycles.

Each constructor returns a CycleModel whose column i is the curve f_i(x) of
the drive row of state i, in the order g, e, third state. The free affine
endpoints of each row absorb amplitude, sign and imperfect pulses.
"""

from collections.abc import Mapping

import numpy as np

from .model import Array, CycleModel

# t1 and tf stay above 1e-3 axis units, for example 1 ns on a us axis.
_SHORTEST_DECAY = 1e-3


def rabi_model() -> CycleModel:
    """Return a three-state model for a Rabi drive of frequency f.

    Axis x is the swept drive amount (gain, or pulse length) in caller units.
    The fitted ``f`` is the Rabi frequency in cycles per axis unit, bounds
    [0, infinity), so the π point is x = 1 / (2 f). The g curve is
    sin²(π f x), the probability that the drive takes g to e; the e curve is
    cos²(π f x), the probability that e stays in e. The third state is not
    driven: its row's free endpoints absorb a flat response, and its curve
    repeats the g curve. ``oscillation_frequency`` is ``f``, so CycleFit
    reports reliability.
    """

    def curve(x: Array, parameters: Mapping[str, float]) -> Array:
        from_g = np.sin(np.pi * parameters["f"] * x) ** 2
        from_e = np.cos(np.pi * parameters["f"] * x) ** 2
        return np.column_stack([from_g, from_e, from_g])

    return CycleModel(curve, ("f",), [(0.0, np.inf)], oscillation_frequency="f")


def t1_model() -> CycleModel:
    """Return a three-state model for a π pulse followed by a wait x.

    Axis x is the wait in caller time units. Fitted ``t1`` and ``tf`` use the
    same units, with bounds [1e-3, infinity). The π pulse takes g to e,
    which decays back with exp(-x / t1): that is the g curve. From e the π
    pulse lands in g, which does not decay; the e row's free endpoints absorb
    that flat response, and its curve exp(-x / t1) also covers a residual e
    after an imperfect π. The third state decays with its own exp(-x / tf).

    The g and third curves are the same family, so exchanging those two rows
    together with t1 and tf fits the same data when the e row is flat; the
    fit may return either labelling. Callers that know which state is most
    populated, such as g in thermal equilibrium, label the rows by
    CycleFit.nuisances.thermal.
    """

    def curve(x: Array, parameters: Mapping[str, float]) -> Array:
        decay = np.exp(-x / parameters["t1"])
        return np.column_stack([decay, decay, np.exp(-x / parameters["tf"])])

    bounds = [(_SHORTEST_DECAY, np.inf), (_SHORTEST_DECAY, np.inf)]
    return CycleModel(curve, ("t1", "tf"), bounds)


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
    then need not be calibrated. The g curve is affine in exp(-x/t2), the same
    family as the third curve, so with a small third-state population the fit
    can put g in the third row with t2 and tf exchanged; label the rows by
    CycleFit.nuisances.thermal as for t1_model. Direct curve calls raise
    ValueError for negative/nonfinite delay; parameter validation is owned by
    CycleModel.
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
